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


def test_switch_step_out_of_range_rejected():
    import train_125m as t
    for bad in (0, 1000, 5000):
        try:
            t.train(arch="window_power_4.0", switch_step=bad, max_steps=1000)
        except ValueError as e:
            assert "switch-step" in str(e)
        else:
            raise AssertionError(f"expected ValueError for switch_step={bad}")
