"""Tests for the window-removal (curriculum) machinery.

These cover the pieces the pre-registered experiments depend on:

  * the quartic schedule the 3.4M curriculum arm starts from,
  * the "full_masked" switch target, which must be attention-identical to plain
    causal attention -- it is the control that separates "the mask changed" from
    "the attention kernel changed" in the seed-256 divergence,
  * the linear window ramp endpoints,
  * remove_windows() at 125M, which must clear BOTH the flash-attn window_size
    and the SDPA attn_bias, since a stale bias would silently keep the window.

All CPU, no data download, no training.
"""
import dataclasses
import json

import torch

import windows
import experiment_mechanism as em


# --------------------------------------------------------------------------
# 3.4M curriculum arm
# --------------------------------------------------------------------------
def test_quartic_windows_depth4():
    assert em._quartic_windows(n_layer=4, seq_len=256) == [8, 23, 86, 256]


def test_window_list_cfg_round_trips():
    widths = em._quartic_windows(n_layer=4, seq_len=256)
    mode = em._window_list_cfg(widths)["window"]
    got = [windows.compute_window_size(i, 4, 256, mode) for i in range(4)]
    assert got == widths


def test_switch_mode_full_clears_arch_cfg():
    # arch_cfg == {} is what routes train_r4.Attention to the fused causal
    # kernel instead of the explicit masked softmax.
    assert em._post_switch_cfg("full") == {}


def test_switch_mode_full_masked_is_attention_identical_to_causal():
    """The kernel control must not change the attention pattern.

    If it did, a divergence surviving the control would be uninterpretable.
    """
    cfg = em._post_switch_cfg("full_masked")
    T = 24
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    for layer in range(4):
        mask = windows.compute_window_mask(T, layer, 4, cfg["window"], "cpu",
                                           dtype=torch.bool)
        assert mask is not None, "full_masked must keep the masked-softmax path"
        assert torch.equal(mask, causal), f"layer {layer} is not plain causal"


def test_unknown_switch_mode_raises():
    try:
        em._post_switch_cfg("sideways")
    except ValueError as e:
        assert "sideways" in str(e)
    else:
        raise AssertionError("expected ValueError for an unknown switch mode")


def test_window_ramp_endpoints():
    """The ramp starts at quartic and ends at full attention."""
    T, L = 256, 4
    quartic = em._quartic_windows(n_layer=L, seq_len=T)
    ramp = lambda frac: [int(round(w + frac * (T - w))) for w in quartic]
    assert ramp(0.0) == quartic
    assert ramp(1.0) == [T] * L
    mid = ramp(0.5)
    # Monotone in the ramp fraction, and never outside [quartic, T].
    for w0, wm, w1 in zip(quartic, mid, [T] * L):
        assert w0 <= wm <= w1


# --------------------------------------------------------------------------
# 125M curriculum arm
# --------------------------------------------------------------------------
def _small_windowed_model():
    import train_125m as t
    # dataclasses.replace, not mutation: CONFIGS holds shared instances.
    cfg = dataclasses.replace(t.CONFIGS["window_power_4.0"],
                              n_layer=4, n_head=4, n_embd=64,
                              max_seq_len=64, vocab_size=128)
    return t, cfg, t.GPT125M(cfg)


def test_remove_windows_clears_size_and_bias():
    t, cfg, model = _small_windowed_model()
    windowed = [b.attn for b in model.blocks
                if b.attn.window_size is not None or b.attn.attn_bias is not None]
    assert windowed, "the fixture must actually have windows to remove"

    n = t.remove_windows(model)
    assert n == len(windowed)
    for b in model.blocks:
        assert b.attn.window_size is None
        assert b.attn.attn_bias is None
        assert b.attn.window_mode == "none"


def test_forward_still_runs_after_removal():
    t, cfg, model = _small_windowed_model()
    t.remove_windows(model)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        logits, loss = model(x, x)
    assert torch.isfinite(loss)
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_remove_windows_is_idempotent():
    t, cfg, model = _small_windowed_model()
    t.remove_windows(model)
    assert t.remove_windows(model) == 0


def test_switch_step_rejected_for_unwindowed_arch():
    """Fails before touching the GPU, so a bad sweep invocation dies fast."""
    import train_125m as t
    try:
        t.train(arch="baseline", switch_step=100, max_steps=1000)
    except ValueError as e:
        assert "no windows" in str(e)
    else:
        raise AssertionError("expected ValueError for baseline + --switch-step")


# --------------------------------------------------------------------------
# Sweep durability
# --------------------------------------------------------------------------
def _stub_training(monkeypatch, tmp_path):
    """Run experiment7 without real data, training cost, or eval cost.

    chdir into tmp_path as well: experiment7 writes its final checkpoint to a
    path relative to the working directory, and tests must not deposit files in
    the repo's checkpoints/.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(em, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(em, "load_data",
                        lambda split: torch.randint(0, 200, (5000,), dtype=torch.uint8))
    monkeypatch.setattr(em, "get_batch",
                        lambda d, b, t, dev: (torch.randint(0, 200, (2, 32), device=dev),
                                              torch.randint(0, 200, (2, 32), device=dev)))
    monkeypatch.setattr(em, "evaluate_val_bpb", lambda *a, **k: 1.234)
    monkeypatch.setattr(em, "BATCH_SIZE", 2)


def test_results_are_written_after_every_seed(monkeypatch, tmp_path):
    """A sweep killed mid-way must not lose the seeds that already finished.

    Regression test: results used to be written only after ALL seeds, so a run
    killed on the last seed lost every completed one.
    """
    _stub_training(monkeypatch, tmp_path)
    seen = []
    real = em._write_exp7_files

    def spy(curve_path, all_results, *a, **k):
        seen.append(sorted(r["seed"] for r in all_results.values()
                           if r["config"] == "F_switch"))
        return real(curve_path, all_results, *a, **k)

    monkeypatch.setattr(em, "_write_exp7_files", spy)
    em.experiment7(seeds=[42, 137], switch_step=4, total_steps=8, tag="dur")

    # Written after seed 42 (before 137 existed), after 137, and once at the end.
    assert [42] in seen, "no write happened before the second seed finished"
    assert seen[-1] == [42, 137]
    written = json.loads((tmp_path / "exp7_curriculum_sw4_dur.json").read_text())
    assert sorted(r["seed"] for r in written["runs"].values()
                  if r["config"] == "F_switch") == [42, 137]


def test_final_f_checkpoint_is_saved_with_post_switch_arch(monkeypatch, tmp_path):
    """The F arm must be re-evaluatable later without retraining.

    The recorded arch_cfg must be the POST-switch one: an F model is
    full-attention at the end, and re-loading it under the quartic config would
    evaluate the wrong operator.
    """
    _stub_training(monkeypatch, tmp_path)
    em.experiment7(seeds=[42], switch_step=4, total_steps=8, tag="ck")

    p = tmp_path / "checkpoints" / "model_F_switch_42_sw4_ck.pt"
    assert p.exists(), "no final F checkpoint was written"
    blob = torch.load(p, map_location="cpu", weights_only=False)
    assert blob["arch"] == "F_switch"
    assert blob["arch_cfg"] == {}, "should record full attention, not quartic"
    assert blob["seed"] == 42 and blob["switch_step"] == 4
    assert "model_state_dict" in blob

    # full_masked records its window list rather than an empty config
    em.experiment7(seeds=[42], switch_step=4, total_steps=8, tag="ckm",
                   switch_mode="full_masked")
    blob2 = torch.load(tmp_path / "checkpoints" / "model_F_switch_42_sw4_ckm.pt",
                       map_location="cpu", weights_only=False)
    assert blob2["arch_cfg"] == em._post_switch_cfg("full_masked")


def test_resume_seeds_skips_completed_and_keeps_them(monkeypatch, tmp_path):
    _stub_training(monkeypatch, tmp_path)
    em.experiment7(seeds=[42], switch_step=4, total_steps=8, tag="res")

    trained = []
    real_gpt = em.GPT
    monkeypatch.setattr(em, "GPT",
                        lambda *a, **k: trained.append(1) or real_gpt(*a, **k))
    r = em.experiment7(seeds=[42, 137], switch_step=4, total_steps=8, tag="res",
                       resume_seeds=True)

    assert len(trained) == 1, "seed 42 was retrained instead of being skipped"
    assert sorted(v["seed"] for v in r.values() if v["config"] == "F_switch") == [42, 137]


def test_no_resume_flag_starts_clean(monkeypatch, tmp_path):
    """Without --resume-seeds, a rerun must not silently reuse stale results."""
    _stub_training(monkeypatch, tmp_path)
    em.experiment7(seeds=[42, 137], switch_step=4, total_steps=8, tag="clean")
    trained = []
    real_gpt = em.GPT
    monkeypatch.setattr(em, "GPT",
                        lambda *a, **k: trained.append(1) or real_gpt(*a, **k))
    em.experiment7(seeds=[42, 137], switch_step=4, total_steps=8, tag="clean")
    assert len(trained) == 2, "stale results were reused without --resume-seeds"


def test_switch_step_out_of_range_rejected():
    import train_125m as t
    for bad in (0, 1000, 5000):
        try:
            t.train(arch="window_power_4.0", switch_step=bad, max_steps=1000)
        except ValueError as e:
            assert "switch-step" in str(e)
        else:
            raise AssertionError(f"expected ValueError for switch_step={bad}")


def test_checkpoint_may_be_resumed_before_the_release_point(monkeypatch, tmp_path):
    """A windowed checkpoint can be resumed at its own step and trained on.

    This is what lets a release-point sweep reuse one windowed prefix per seed
    for every release point at or after the checkpoint.
    """
    _stub_training(monkeypatch, tmp_path)
    ck = tmp_path / "ck"
    em.experiment7(seeds=[42], switch_step=4, total_steps=20, stop_step=6,
                   save_switch_ckpt=str(ck), tag="pre")
    p = ck / "switch_s42_sw4.pt"

    # Resume the step-4 checkpoint for a run that releases later, at step 10.
    r = em.experiment7(seeds=[42], switch_step=10, total_steps=20,
                       load_switch_ckpt=str(p), tag="later")
    run = r["F_switch_s42"]
    assert run["switch_step"] == 10
    assert run["curve"][0][0] == 4, "should resume at the checkpoint's step, not 0"

    # Resuming past the release point is refused: the prefix would be wrong.
    try:
        em.experiment7(seeds=[42], switch_step=2, total_steps=20,
                       load_switch_ckpt=str(p), tag="earlier")
    except ValueError as e:
        assert "after the release point" in str(e)
    else:
        raise AssertionError("expected ValueError resuming past the release point")

    # Seed mismatch is still refused.
    try:
        em.experiment7(seeds=[137], switch_step=10, total_steps=20,
                       load_switch_ckpt=str(p), tag="wrongseed")
    except ValueError as e:
        assert "seed" in str(e)
    else:
        raise AssertionError("expected ValueError on seed mismatch")


def test_tagged_runs_do_not_clobber_each_other_in_the_shared_file(monkeypatch, tmp_path):
    """Two tags at one switch step must not overwrite each other's results.

    Regression test. mechanism_disambiguation.json keyed exp7 blocks by switch
    step alone, and runs are keyed by config+seed WITHIN a block, so the floor
    sweep (sw=250, ramp=250) overwrote the release-point sweep (sw=250,
    ramp=500) value-for-value -- and deleted the two seeds it had no data for,
    because the block is assigned rather than merged into.
    """
    _stub_training(monkeypatch, tmp_path)
    a = em.experiment7(seeds=[42], switch_step=4, total_steps=8, tag="tagA",
                       ramp_steps=0)
    em.print_summary(None, None, None, a, tag="tagA")
    b = em.experiment7(seeds=[42], switch_step=4, total_steps=8, tag="tagB",
                       ramp_steps=2)
    em.print_summary(None, None, None, b, tag="tagB")

    blob = json.loads((tmp_path / "mechanism_disambiguation.json").read_text())
    assert "exp7_sw4_tagA" in blob, "the first tag's block was overwritten"
    assert "exp7_sw4_tagB" in blob
    # Same config, same seed, same stubbed bpb -- only the treatment differs, so
    # ramp_steps is what proves the two blocks are not the same run twice.
    assert blob["exp7_sw4_tagA"]["F_switch_s42"]["ramp_steps"] == 0
    assert blob["exp7_sw4_tagB"]["F_switch_s42"]["ramp_steps"] == 2


def test_untagged_10k_run_keeps_the_canonical_exp7_key():
    """The historical key is preserved, so committed files stay readable."""
    assert em.exp7_key(10000, "") == "exp7"
    assert em.exp7_key(10000, "5seed") == "exp7_sw10000_5seed"
    assert em.exp7_key(250, "") == "exp7_sw250"
    assert em.exp7_key(250, "floor250") == "exp7_sw250_floor250"
