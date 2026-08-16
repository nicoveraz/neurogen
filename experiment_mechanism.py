"""
NeuroGen: Mechanism Disambiguation Experiments (4-7)

Tests surviving hypotheses after ruling out gradient noise removal,
softmax coupling, and variance reduction.

Exp 4: Train-val gap (implicit regularization)
Exp 5: Gradient covariance rank (parameter coupling)
Exp 6: Gradient stability of trained models (landscape smoothness)
Exp 7: Remove windows mid-training (the load-bearing result)

Usage:
    python experiment_mechanism.py --exp4
    python experiment_mechanism.py --exp5
    python experiment_mechanism.py --exp6

    # Exp 7 at 5 seeds (arms A and B come from validation_results/; only F trains)
    python experiment_mechanism.py --exp7 --seed-list 42,137,256,789,1337

    # Switch-point sweep
    python experiment_mechanism.py --exp7 --seed-list 42,137,256 --switch-step 5000

    # Divergence diagnosis at the switch. --switch-mode full_masked keeps the
    # masked-softmax path so only the mask changes, isolating the kernel switch
    # from stale optimizer state; --trace-window logs per-step loss, pre-clip
    # grad norm, and max Adam update around the switch.
    python experiment_mechanism.py --exp7 --seed-list 256 --total-steps 12000 \
        --switch-mode full_masked --trace-window 100
    python experiment_mechanism.py --exp7 --seed-list 256 --total-steps 12000 \
        --reset-optimizer --trace-window 100

    python experiment_mechanism.py --all
"""

import argparse, json, math, time, sys, os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import windows
from prepare import load_data, get_batch, evaluate_val_bpb, VOCAB_SIZE, MAX_SEQ_LEN
from train_r4 import (GPT, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS, DEVICE,
                       BATCH_SIZE, ARCHS, get_arch_cfg, compute_window_mask, rms_norm)
from ca_rules import block_diagonal_init

LR = 2e-3
WEIGHT_DECAY = 0.05
WARMUP = 200
RESULTS_DIR = Path("gradient_results")


def load_checkpoint(ckpt_path: str) -> GPT:
    """Load a trained model from checkpoint."""
    model = GPT(VOCAB_SIZE, MAX_SEQ_LEN, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS,
                arch_cfg={})
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(DEVICE)


def load_checkpoint_with_arch(ckpt_path: str, arch_cfg: dict) -> GPT:
    """Load checkpoint into model with specific arch config."""
    model = GPT(VOCAB_SIZE, MAX_SEQ_LEN, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS,
                arch_cfg=arch_cfg)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model.to(DEVICE)


def get_lr(step, warmup, max_steps, lr=LR, min_lr=LR/10):
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


# ===========================================================================
# Experiment 4: Train-Val Gap (Implicit Regularization)
# ===========================================================================
def experiment4():
    """Measure train-val gap for baseline vs quartic checkpoints."""
    print("=" * 70)
    print("  EXPERIMENT 4: Train-Val Gap (Implicit Regularization)")
    print("=" * 70)
    print("  Hypothesis: Windows reduce overfitting by constraining capacity")
    print("  Prediction: Baseline should have larger train-val gap\n")

    train_data = load_data("train")
    val_data = load_data("val")

    configs = {
        "baseline": ("checkpoints/model_baseline_42.pt", {}),
        "quartic": ("checkpoints/model_window_power_4.0_42.pt",
                     ARCHS.get("window_power_4.0", {})),
    }

    results = []
    for name, (ckpt, arch_cfg) in configs.items():
        if not os.path.exists(ckpt):
            print(f"EXP4_SKIP: {ckpt} not found")
            continue

        model = load_checkpoint_with_arch(ckpt, arch_cfg)
        model.eval()

        # Compute val loss
        val_bpb = evaluate_val_bpb(model, val_data, BATCH_SIZE, MAX_SEQ_LEN, DEVICE)

        # Compute train loss (same amount of data as val eval)
        model.eval()
        n_batches = max(1, 100_000 // (BATCH_SIZE * MAX_SEQ_LEN))
        train_loss_total = 0
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = get_batch(train_data, BATCH_SIZE, MAX_SEQ_LEN, DEVICE)
                _, loss = model(x, y)
                train_loss_total += loss.item()
        train_bpb = (train_loss_total / n_batches) / math.log(2)

        gap = val_bpb - train_bpb
        gap_pct = gap / train_bpb * 100

        r = {"config": name, "train_bpb": round(train_bpb, 4),
             "val_bpb": round(val_bpb, 4), "gap": round(gap, 4),
             "gap_pct": round(gap_pct, 2)}
        results.append(r)
        print(f"EXP4_RESULT: {name:15s}  train={train_bpb:.4f}  val={val_bpb:.4f}  "
              f"gap={gap:.4f} ({gap_pct:+.2f}%)")

    # Also check seed 137 if available
    for seed in [137]:
        for name, arch_name in [("baseline", "baseline"), ("quartic", "window_power_4.0")]:
            ckpt = f"checkpoints/model_{arch_name}_{seed}.pt"
            if not os.path.exists(ckpt):
                continue
            arch_cfg = ARCHS.get(arch_name, {})
            model = load_checkpoint_with_arch(ckpt, arch_cfg)
            model.eval()
            val_bpb = evaluate_val_bpb(model, val_data, BATCH_SIZE, MAX_SEQ_LEN, DEVICE)
            n_batches = max(1, 100_000 // (BATCH_SIZE * MAX_SEQ_LEN))
            train_loss_total = 0
            with torch.no_grad():
                for _ in range(n_batches):
                    x, y = get_batch(train_data, BATCH_SIZE, MAX_SEQ_LEN, DEVICE)
                    _, loss = model(x, y)
                    train_loss_total += loss.item()
            train_bpb = (train_loss_total / n_batches) / math.log(2)
            gap = val_bpb - train_bpb
            gap_pct = gap / train_bpb * 100
            r = {"config": f"{name}_s{seed}", "train_bpb": round(train_bpb, 4),
                 "val_bpb": round(val_bpb, 4), "gap": round(gap, 4),
                 "gap_pct": round(gap_pct, 2)}
            results.append(r)
            print(f"EXP4_RESULT: {name+'_s'+str(seed):15s}  train={train_bpb:.4f}  "
                  f"val={val_bpb:.4f}  gap={gap:.4f} ({gap_pct:+.2f}%)")

    print(f"\nEXP4_SUMMARY:")
    bl_gaps = [r["gap"] for r in results if "baseline" in r["config"]]
    q4_gaps = [r["gap"] for r in results if "quartic" in r["config"]]
    bl_mean = sum(bl_gaps) / len(bl_gaps) if bl_gaps else 0
    q4_mean = sum(q4_gaps) / len(q4_gaps) if q4_gaps else 0
    print(f"  Baseline mean gap: {bl_mean:.4f}")
    print(f"  Quartic mean gap:  {q4_mean:.4f}")
    if bl_mean > q4_mean * 1.2:
        print(f"  → Baseline gap is {bl_mean/q4_mean:.1f}x larger → implicit regularization IS a factor")
    elif abs(bl_mean - q4_mean) / max(bl_mean, q4_mean, 1e-10) < 0.1:
        print(f"  → Gaps are similar → implicit regularization is NOT the primary mechanism")
    else:
        print(f"  → Small difference — inconclusive")

    return results


# ===========================================================================
# Experiment 5: Gradient Covariance Rank (Parameter Coupling)
# ===========================================================================
def experiment5():
    """Measure effective rank of gradient covariance under different windows."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 5: Gradient Covariance Rank (Parameter Coupling)")
    print("=" * 70)
    print("  Hypothesis: Windows reduce parameter coupling, simplifying landscape")
    print("  Prediction: Fewer large eigenvalues with smaller windows\n")

    model = load_checkpoint("checkpoints/model_baseline_42.pt")
    model.train()
    val_data = load_data("val")

    window_sizes = [8, 32, 64, 128, 256]
    results = []

    for window in window_sizes:
        t0 = time.time()
        grads = []

        for i in range(50):
            model.zero_grad()
            x, y = get_batch(val_data, BATCH_SIZE, MAX_SEQ_LEN, DEVICE)

            # Manual forward with window at layer 0
            B, T = x.size()
            cos, sin = model.cos[:, :T], model.sin[:, :T]
            emb = rms_norm(model.wte(x))
            if T > 1:
                gate = model.smear_lambda * torch.sigmoid(
                    model.smear_gate(emb[:, 1:, :model._smear_ch]))
                emb = torch.cat([emb[:, :1], emb[:, 1:] + gate * emb[:, :-1]], dim=1)
            x_in, x0, prev_attn = emb, emb, None

            for li, block in enumerate(model.blocks):
                x_in = model.resid_lambdas[li] * x_in + model.x0_lambdas[li] * x0
                ve = model.value_embeds[str(li)](x) if str(li) in model.value_embeds else None
                if li == 0 and window < MAX_SEQ_LEN:
                    rows = torch.arange(T, device=DEVICE).unsqueeze(1)
                    cols = torch.arange(T, device=DEVICE).unsqueeze(0)
                    mask = ((cols <= rows) & (cols >= rows - window + 1)).float()
                else:
                    mask = None
                x_in, prev_attn = block(x_in, ve, cos, sin, mask=mask,
                                        prev_attn=prev_attn)

            logits = model.lm_head(rms_norm(x_in)).float()
            logits = 15 * torch.tanh(logits / 15)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()

            # Collect Q/K gradients at layer 0
            q_grad = model.blocks[0].attn.c_q.weight.grad.detach().flatten()
            k_grad = model.blocks[0].attn.c_k.weight.grad.detach().flatten()
            grads.append(torch.cat([q_grad, k_grad]).cpu())

        G = torch.stack(grads).float()  # (50, n_params)

        # SVD for effective rank
        try:
            _, S, _ = torch.linalg.svd(G, full_matrices=False)
            S_norm = S / (S.sum() + 1e-10)
            entropy = -(S_norm * (S_norm + 1e-10).log()).sum().item()
            eff_rank = math.exp(entropy)
            cumvar = (S ** 2).cumsum(0) / (S ** 2).sum()
            var_top1 = cumvar[0].item()
            var_top5 = cumvar[min(4, len(cumvar)-1)].item()
            top_sv = S[0].item()
        except Exception as e:
            print(f"  SVD failed for window={window}: {e}")
            eff_rank = var_top1 = var_top5 = top_sv = float('nan')

        elapsed = time.time() - t0
        r = {"window": window, "eff_rank": round(eff_rank, 2),
             "var_top1": round(var_top1, 4), "var_top5": round(var_top5, 4),
             "top_singular": round(top_sv, 4), "time_s": round(elapsed, 1)}
        results.append(r)
        print(f"EXP5_RESULT: window={window:>3d}  eff_rank={eff_rank:.2f}  "
              f"var_top1={var_top1:.3f}  var_top5={var_top5:.3f}  "
              f"top_sv={top_sv:.4f}  time={elapsed:.0f}s")

    print(f"\nEXP5_SUMMARY:")
    r8 = next((r for r in results if r["window"] == 8), None)
    r256 = next((r for r in results if r["window"] == 256), None)
    if r8 and r256:
        if r8["eff_rank"] < r256["eff_rank"] * 0.8:
            print(f"  Effective rank drops {r256['eff_rank']:.1f} → {r8['eff_rank']:.1f} "
                  f"with smaller windows → parameter coupling IS reduced")
        else:
            print(f"  Effective rank similar ({r256['eff_rank']:.1f} vs {r8['eff_rank']:.1f}) "
                  f"→ parameter coupling is NOT significantly affected")

    return results


# ===========================================================================
# Experiment 6: Gradient Stability of Trained Models
# ===========================================================================
def experiment6():
    """Compare gradient stability between models trained with/without windows."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 6: Gradient Stability of Trained Models")
    print("=" * 70)
    print("  Hypothesis: Windowed training produces a smoother landscape")
    print("  Prediction: Quartic-trained model has more stable gradients\n")
    print("  Note: Exp1 measured stability WITH masks on frozen model.")
    print("  This measures stability of models TRAINED with different configs.\n")

    val_data = load_data("val")

    configs = {
        "baseline_s42": ("checkpoints/model_baseline_42.pt", {}),
        "quartic_s42": ("checkpoints/model_window_power_4.0_42.pt",
                         ARCHS.get("window_power_4.0", {})),
    }

    # Add seed 137 if available
    if os.path.exists("checkpoints/model_baseline_137.pt"):
        configs["baseline_s137"] = ("checkpoints/model_baseline_137.pt", {})
    if os.path.exists("checkpoints/model_window_power_4.0_137.pt"):
        configs["quartic_s137"] = ("checkpoints/model_window_power_4.0_137.pt",
                                     ARCHS.get("window_power_4.0", {}))

    results = []
    for name, (ckpt, arch_cfg) in configs.items():
        if not os.path.exists(ckpt):
            print(f"EXP6_SKIP: {ckpt} not found")
            continue

        t0 = time.time()
        model = load_checkpoint_with_arch(ckpt, arch_cfg)
        model.train()

        grads = []
        for i in range(30):
            model.zero_grad()
            x, y = get_batch(val_data, BATCH_SIZE, MAX_SEQ_LEN, DEVICE)
            _, loss = model(x, y)
            loss.backward()

            all_g = []
            for p in model.parameters():
                if p.grad is not None:
                    all_g.append(p.grad.detach().flatten())
            grads.append(torch.cat(all_g).cpu())

        G = torch.stack(grads).float()

        # Signal and noise
        signal = G.mean(dim=0)
        signal_norm = signal.norm().item()
        noise = G.std(dim=0)
        noise_norm = noise.norm().item()
        snr = signal_norm / (noise_norm + 1e-10)

        # Direction stability
        G_normed = F.normalize(G, dim=1)
        cos_sim = G_normed @ G_normed.T
        mask = torch.triu(torch.ones(30, 30, dtype=torch.bool), diagonal=1)
        stability = cos_sim[mask].mean().item()

        elapsed = time.time() - t0
        r = {"config": name, "snr": round(snr, 4),
             "stability": round(stability, 4),
             "signal_norm": round(signal_norm, 6),
             "noise_norm": round(noise_norm, 6),
             "time_s": round(elapsed, 1)}
        results.append(r)
        print(f"EXP6_RESULT: {name:15s}  snr={snr:.4f}  stability={stability:.4f}  "
              f"signal={signal_norm:.6f}  noise={noise_norm:.6f}  time={elapsed:.0f}s")

    print(f"\nEXP6_SUMMARY:")
    bl_stab = [r["stability"] for r in results if "baseline" in r["config"]]
    q4_stab = [r["stability"] for r in results if "quartic" in r["config"]]
    bl_mean = sum(bl_stab) / len(bl_stab) if bl_stab else 0
    q4_mean = sum(q4_stab) / len(q4_stab) if q4_stab else 0
    print(f"  Baseline mean stability: {bl_mean:.4f}")
    print(f"  Quartic mean stability:  {q4_mean:.4f}")
    if q4_mean > bl_mean * 1.2:
        print(f"  → Quartic-trained model has smoother landscape → optimization landscape IS a factor")
    elif abs(bl_mean - q4_mean) / max(bl_mean, q4_mean, 1e-10) < 0.1:
        print(f"  → Similar stability → landscape smoothness is NOT the differentiator")
    else:
        print(f"  → Small difference — may contribute but not dominant")

    return results


# ===========================================================================
# Experiment 7: Remove Windows Mid-Training (Curriculum vs Structure)
# ===========================================================================
def _quartic_windows(n_layer=DEPTH, seq_len=MAX_SEQ_LEN):
    """Per-layer quartic window widths, e.g. [8, 23, 86, 256] at depth 4."""
    return [windows.compute_window_size(i, n_layer, seq_len, "power_4.0")
            for i in range(n_layer)]


def _window_list_cfg(widths):
    """arch_cfg for explicit per-layer windows (keeps the masked-softmax path)."""
    return {"window": "list:" + ",".join(str(int(w)) for w in widths)}


def _post_switch_cfg(mode):
    """What the model switches TO at the switch point.

    "full" reproduces the original recipe: arch_cfg = {}, which also flips the
    attention implementation from an explicit masked softmax to the fused causal
    kernel (train_r4.Attention gates on `mask is not None`).

    "full_masked" sets every layer's window to the full sequence length instead.
    The attention pattern is identical to "full", but the masked-softmax path is
    retained. Comparing the two isolates "the mask changed" from "the kernel
    changed" as a cause of the divergence seen at seed 256.
    """
    if mode == "full":
        return {}
    if mode == "full_masked":
        return _window_list_cfg([MAX_SEQ_LEN] * DEPTH)
    raise ValueError(f"Unknown switch mode '{mode}' (expected full|full_masked)")


def _max_adam_update(optimizer):
    """Largest single-parameter update Adam would apply this step.

    Diagnostic for the stale-second-moment hypothesis: after an abrupt change in
    the gradient distribution, m/sqrt(v) can be large for parameters whose
    historical v is small. Gradient-norm clipping does not bound this.
    """
    worst = 0.0
    for group in optimizer.param_groups:
        b1, b2 = group["betas"]
        eps, lr = group["eps"], group["lr"]
        for p in group["params"]:
            st = optimizer.state.get(p)
            if not st or "exp_avg" not in st:
                continue
            t = st["step"]
            t = t.item() if torch.is_tensor(t) else t
            if t < 1:
                continue
            m_hat = st["exp_avg"] / (1 - b1 ** t)
            v_hat = st["exp_avg_sq"] / (1 - b2 ** t)
            u = (lr * m_hat / (v_hat.sqrt() + eps)).abs().max().item()
            worst = max(worst, u)
    return worst


def _rng_state():
    st = {"cpu": torch.get_rng_state()}
    if DEVICE == "mps" and hasattr(torch, "mps"):
        try:
            st["mps"] = torch.mps.get_rng_state()
        except Exception:
            pass
    elif DEVICE == "cuda":
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _restore_rng(st):
    # RNG states are ByteTensors and must live on CPU; torch.load(map_location=
    # DEVICE) will have moved them to the accelerator, so move them back.
    torch.set_rng_state(st["cpu"].cpu().to(torch.uint8))
    if "mps" in st:
        try:
            torch.mps.set_rng_state(st["mps"].cpu().to(torch.uint8))
        except Exception:
            pass
    if "cuda" in st:
        torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in st["cuda"]])


def _write_exp7_files(curve_path, all_results, traces, seeds, switch_step,
                      total_steps, stop_step, switch_mode, post_switch_warmup,
                      reset_optimizer, ramp_steps, load_switch_ckpt, suffix):
    """Write the sweep's curves and switch traces. Called after every seed."""
    cfg_blob = {"switch_step": switch_step, "total_steps": total_steps,
                "stop_step": stop_step, "switch_mode": switch_mode,
                "post_switch_warmup": post_switch_warmup,
                "reset_optimizer": reset_optimizer, "ramp_steps": ramp_steps,
                "resumed_from": load_switch_ckpt, "seeds": list(seeds)}
    with open(curve_path, "w") as f:
        json.dump({"config": cfg_blob, "runs": all_results}, f, indent=2)
    for seed, tr in traces.items():
        tp = curve_path.parent / (f"exp7_switch_trace_s{seed}_sw{switch_step}"
                                  f"{suffix}.json")
        with open(tp, "w") as f:
            json.dump({"seed": seed, "config": cfg_blob, "trace": tr}, f, indent=2)


def experiment7(seeds=(42, 137, 256), switch_step: int = 10000,
                total_steps: int = 20000, switch_mode: str = "full",
                post_switch_warmup: int = 0, reset_optimizer: bool = False,
                ramp_steps: int = 0, trace_window: int = 0,
                stop_step: int | None = None,
                save_switch_ckpt: str | None = None,
                load_switch_ckpt: str | None = None,
                tag: str = "", resume_seeds: bool = False):
    """Train with quartic windows for `switch_step` steps, then remove them.

    A: full attention throughout (cached).  B: quartic throughout (cached).
    F: quartic for `switch_step` steps, then full attention. Only F is trained.

    F ≈ B → the hierarchy persists without the mask (curriculum).
    F worse than B → windows are an ongoing constraint.
    F better than B → windows become a ceiling.

    The optional arguments exist to diagnose the divergence observed at seed 256:
    `switch_mode` isolates the kernel change, `post_switch_warmup` and
    `reset_optimizer` test the stale-optimizer-state hypothesis, and `ramp_steps`
    replaces the discontinuity with a linear window ramp.

    `stop_step` ends the run early WITHOUT changing the LR schedule, which stays
    keyed to `total_steps`. Shortening `total_steps` instead would move the whole
    cosine and change the learning rate at the switch, so the divergence would no
    longer be reproduced under its original conditions.

    `save_switch_ckpt` / `load_switch_ckpt` share one pre-switch state across
    treatments. Every intervention then starts from an identical model, optimizer
    and RNG state, which makes them a controlled comparison instead of four
    independent runs (and skips re-training the identical first `switch_step`
    steps each time).
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 7: Remove Windows Mid-Training")
    print("=" * 70)
    print(f"  seeds={list(seeds)}  switch_step={switch_step}  total_steps={total_steps}")
    print(f"  switch_mode={switch_mode}  post_switch_warmup={post_switch_warmup}  "
          f"reset_optimizer={reset_optimizer}  ramp_steps={ramp_steps}")
    if stop_step is not None:
        print(f"  stop_step={stop_step} (LR schedule still keyed to {total_steps})")
    if load_switch_ckpt:
        print(f"  resuming post-switch from {load_switch_ckpt}")
    if save_switch_ckpt:
        print(f"  will save pre-switch state under {save_switch_ckpt}/")
    print()

    train_data = load_data("train")
    val_data = load_data("val")
    eval_interval = 500
    quartic_w = _quartic_windows()
    target_cfg = _post_switch_cfg(switch_mode)

    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    curve_path = RESULTS_DIR / f"exp7_curriculum_sw{switch_step}{suffix}.json"

    all_results = {}
    traces = {}

    # Each seed is ~70-100 min. Persist after every seed and allow resuming, so
    # an interrupted sweep costs at most the in-progress seed rather than all of
    # them.
    done = set()
    if resume_seeds and curve_path.exists():
        prev = json.load(open(curve_path))
        all_results = prev.get("runs", {})
        done = {r["seed"] for r in all_results.values() if r["config"] == "F_switch"}
        print(f"  resuming: {curve_path.name} already has seeds "
              f"{sorted(done)} — those will be skipped\n")

    for seed in seeds:
        if seed in done:
            print(f"  seed {seed}: already complete, skipping")
            continue
        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")

        # Config A & B: load from existing validation results if available
        for label, arch_name in [("A_full", "baseline"), ("B_quartic", "window_power_4.0")]:
            json_path = f"validation_results/{arch_name}_s{seed}.json"
            if os.path.exists(json_path):
                d = json.load(open(json_path))
                s = d["summary"]
                final_bpb = s.get("final_vbpb", s.get("final_val_bpb"))
                curve = [(p["step"], p["val_bpb"]) for p in d["curve"]]
                key = f"{label}_s{seed}"
                all_results[key] = {"config": label, "seed": seed,
                                     "final_bpb": final_bpb, "source": "cached",
                                     "curve": curve}
                print(f"  {label}: loaded from cache → bpb={final_bpb:.4f}")
            else:
                print(f"  {label}: {json_path} not found — skipping")

        # Config F: quartic → full (must train)
        print(f"\n--- F: Quartic {switch_step} → full {total_steps - switch_step} "
              f"(seed={seed}) ---")
        torch.manual_seed(seed)

        arch_cfg_quartic = get_arch_cfg("window_power_4.0")
        model = GPT(VOCAB_SIZE, MAX_SEQ_LEN, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS,
                     arch_cfg=dict(arch_cfg_quartic)).to(DEVICE)

        # Standard init (same as validate.py)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.dim() >= 2 and not any(s in name for s in ("wte", "lm_head", "ve_gate")):
                    if min(p.shape) >= 8:
                        nn.init.xavier_uniform_(p)
                        ca = block_diagonal_init(p.shape, n_blocks=min(4, min(p.shape)),
                                                  target_std=p.std().item() * 0.05)
                        p.data.add_(ca.to(p.device))

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        # Optionally start from a shared pre-switch state instead of retraining
        # the identical first switch_step steps.
        first_step = 0
        if load_switch_ckpt:
            blob = torch.load(load_switch_ckpt, map_location=DEVICE, weights_only=False)
            if blob["seed"] != seed or blob["switch_step"] != switch_step:
                raise ValueError(
                    f"{load_switch_ckpt} is seed={blob['seed']} "
                    f"switch_step={blob['switch_step']}, not seed={seed} "
                    f"switch_step={switch_step}")
            model.load_state_dict(blob["model_state_dict"])
            optimizer.load_state_dict(blob["optimizer_state_dict"])
            _restore_rng(blob["rng_state"])
            first_step = switch_step
            print(f"  loaded pre-switch state from {load_switch_ckpt} "
                  f"(step {switch_step})")

        t0 = time.time()
        curve = []
        trace = []
        diverged_at = None
        trace_lo = switch_step - trace_window
        trace_hi = switch_step + 5 * trace_window
        last_step = total_steps if stop_step is None else min(stop_step, total_steps)

        for step in range(first_step, last_step + 1):
            # --- share the pre-switch state ------------------------------------
            if save_switch_ckpt and step == switch_step:
                d = Path(save_switch_ckpt)
                d.mkdir(parents=True, exist_ok=True)
                p = d / f"switch_s{seed}_sw{switch_step}.pt"
                torch.save({"seed": seed, "switch_step": switch_step,
                            "total_steps": total_steps,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "rng_state": _rng_state()}, p)
                print(f"  >>> saved pre-switch state to {p}")

            # --- architecture schedule ---------------------------------------
            if ramp_steps > 0 and switch_step <= step < switch_step + ramp_steps:
                # Linear per-layer interpolation quartic -> full over ramp_steps.
                frac = (step - switch_step) / ramp_steps
                model.arch_cfg = _window_list_cfg(
                    [int(round(w + frac * (MAX_SEQ_LEN - w))) for w in quartic_w])
                if step == switch_step:
                    print(f"  >>> RAMPING quartic → full over {ramp_steps} steps "
                          f"from step {step}")
            elif step == switch_step + ramp_steps:
                model.arch_cfg = dict(target_cfg)
                print(f"  >>> SWITCHED to '{switch_mode}' attention at step {step}")
                if reset_optimizer:
                    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                                  weight_decay=WEIGHT_DECAY)
                    print("  >>> optimizer state reset at the switch")

            # --- eval ---------------------------------------------------------
            # Always evaluate at the first and last step of the loop, so a
            # resumed or early-stopped run still has endpoints on its curve.
            if step % eval_interval == 0 or step in (first_step, last_step):
                model.eval()
                vbpb = evaluate_val_bpb(model, val_data, BATCH_SIZE, MAX_SEQ_LEN, DEVICE)
                elapsed = time.time() - t0
                phase = "quartic" if step < switch_step else switch_mode
                curve.append((step, round(vbpb, 4)))
                if step % 2000 == 0:
                    print(f"  step:{step:6d}  vbpb:{vbpb:.4f}  phase:{phase}  "
                          f"time:{elapsed:.0f}s")
                model.train()
                if math.isnan(vbpb) and diverged_at is None:
                    diverged_at = step
                    print(f"  !!! val_bpb is NaN at step {step} — run diverged")

            if step >= last_step:
                break

            # --- training step -------------------------------------------------
            x, y = get_batch(train_data, BATCH_SIZE, MAX_SEQ_LEN, DEVICE)
            cur_lr = get_lr(step, WARMUP, total_steps)
            if post_switch_warmup > 0 and 0 <= step - switch_step < post_switch_warmup:
                # Re-warm the LR over the first post_switch_warmup steps after the
                # switch, without altering the underlying cosine.
                cur_lr *= (step - switch_step + 1) / post_switch_warmup
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr
            _, loss = model(x, y, step=step, total_steps=total_steps)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()

            # --- switch-point instrumentation ----------------------------------
            if trace_window and trace_lo <= step <= trace_hi:
                trace.append({"step": step, "lr": round(cur_lr, 8),
                              "loss": round(loss.item(), 6),
                              "grad_norm_preclip": round(gnorm, 6),
                              "max_adam_update": round(_max_adam_update(optimizer), 8)})
            if diverged_at is None and not math.isfinite(loss.item()):
                diverged_at = step
                print(f"  !!! loss is {loss.item()} at step {step} — run diverged")

        final_bpb = curve[-1][1]
        key = f"F_switch_s{seed}"
        all_results[key] = {"config": "F_switch", "seed": seed,
                             "final_bpb": final_bpb,
                             "source": "resumed_from_switch" if load_switch_ckpt
                                       else "trained",
                             "switch_step": switch_step, "switch_mode": switch_mode,
                             "total_steps": total_steps, "stop_step": stop_step,
                             "post_switch_warmup": post_switch_warmup,
                             "reset_optimizer": reset_optimizer,
                             "ramp_steps": ramp_steps,
                             "diverged_at": diverged_at,
                             "curve": curve}
        if trace:
            traces[seed] = trace
        print(f"  FINAL: F_switch seed={seed} bpb={final_bpb:.4f}"
              + (f"  (DIVERGED at step {diverged_at})" if diverged_at is not None else ""))

        # Checkpoint the sweep after every seed, not just at the end.
        _write_exp7_files(curve_path, all_results, traces, seeds, switch_step,
                          total_steps, stop_step, switch_mode, post_switch_warmup,
                          reset_optimizer, ramp_steps, load_switch_ckpt, suffix)
        print(f"  progress saved to {curve_path}")

    # Persist the full curves. print_summary() keeps only endpoints in
    # mechanism_disambiguation.json, which is why nothing survived from the
    # original seed-256 divergence — the curve is the diagnostic.
    _write_exp7_files(curve_path, all_results, traces, seeds, switch_step,
                      total_steps, stop_step, switch_mode, post_switch_warmup,
                      reset_optimizer, ramp_steps, load_switch_ckpt, suffix)
    print(f"\n  Full curves saved to {curve_path}")

    # A partial run (stop_step) ends mid-schedule, so its final bpb is not
    # comparable to the fully-trained A and B arms. Skip the paired verdict.
    if stop_step is None:
        summarize_exp7(all_results)
    else:
        print(f"\n  Partial run (stopped at {stop_step} of {total_steps}) — "
              f"no paired comparison against A/B.")
    return all_results


def summarize_exp7(all_results):
    """Per-seed paired summary against the pre-registered criteria.

    The arms are paired by seed (shared init and data order), so per-seed
    differences are the unit of analysis and their SIGNS are reported, not just
    their mean. Reporting only the mean is what let an earlier draft read
    "F better than B" out of two seeds that disagree in sign.
    """
    from analyze_all import paired_permutation_p, paired_t_p, cohens_dz

    print(f"\n{'='*74}")
    print("  EXPERIMENT 7 RESULTS (paired by seed)")
    print(f"{'='*74}")
    seeds = sorted({r["seed"] for r in all_results.values()})
    print(f"  {'seed':>6} {'A: full':>9} {'B: quartic':>11} {'F: switch':>11} "
          f"{'B vs A':>8} {'F vs A':>8} {'F vs B':>8}")
    dFA, dFB = [], []
    for sd in seeds:
        A = all_results.get(f"A_full_s{sd}", {}).get("final_bpb")
        B = all_results.get(f"B_quartic_s{sd}", {}).get("final_bpb")
        Fr = all_results.get(f"F_switch_s{sd}", {})
        F = Fr.get("final_bpb")
        ok = F is not None and math.isfinite(F)
        pct = lambda ref, v: f"{(ref - v) / ref * 100:>+7.2f}%" if (ref and v is not None
                                                                   and math.isfinite(v)) else f"{'—':>8}"
        num = lambda v, w: f"{v:>{w}.4f}" if v is not None else f"{'—':>{w}}"
        fstr = num(F, 11) if ok else f"{'diverged':>11}"
        print(f"  {sd:>6} {num(A, 9)} {num(B, 11)} {fstr} "
              f"{pct(A, B)} {pct(A, F)} {pct(B, F)}")
        if ok and A is not None and B is not None:
            dFA.append(A - F)
            dFB.append(B - F)

    n_div = sum(1 for r in all_results.values()
                if r["config"] == "F_switch"
                and (r["final_bpb"] is None or not math.isfinite(r["final_bpb"])))
    if n_div:
        print(f"\n  {n_div}/{len(seeds)} F runs diverged. Diverged seeds count in the "
              f"denominator; they are not dropped.")

    def _report(name, diffs, rule):
        if len(diffs) < 2:
            print(f"\n  {name}: n={len(diffs)} — no paired test possible.")
            return None
        p, cnt, tot = paired_permutation_p(diffs)
        npos, n = sum(1 for d in diffs if d > 0), len(diffs)
        print(f"\n  {name}: {npos}/{n} positive, perm {cnt}/{tot}={p:.3f}, "
              f"t_p={paired_t_p(diffs):.4f}, dz={cohens_dz(diffs):.2f}")
        print(f"    floor at n={n} is 1/{2**n}={1/2**n:.3f}"
              + ("  (cannot reach p<0.05)" if 1 / 2 ** n > 0.05 else ""))
        print(f"    {rule(npos, n, paired_t_p(diffs))}")
        return p

    _report("F vs A (does removal preserve the benefit?)", dFA,
            lambda npos, n, tp: (
                "VERDICT: claimed — all paired differences positive."
                if npos == n and n >= 5 else
                "VERDICT: consistent direction, but under-powered at this n."
                if npos == n else
                "VERDICT: NOT claimed — not all seeds positive."))
    _report("F vs B (is removal better than keeping?)", dFB,
            lambda npos, n, tp: (
                "VERDICT: claimed — >=4/5 positive and paired-t p<0.05."
                if n >= 5 and npos >= 4 and tp < 0.05 else
                "VERDICT: NOT claimed — sign-split / under-powered. Report as "
                "'no detectable difference', not as 'removal helps'."))


# ===========================================================================
# Combined Summary
# ===========================================================================
def print_summary(r4, r5, r6, r7):
    print("\n" + "=" * 80)
    print("  MECHANISM DISAMBIGUATION: COMPLETE RESULTS")
    print("=" * 80)

    print("\n  DEAD HYPOTHESES (from experiments 1-3):")
    print("  x Gradient noise removal (noise constant at 0.0053)")
    print("  x Softmax coupling contamination (4-7% noise fraction)")
    print("  x Variance reduction (batch size can't replicate)")

    print("\n  NEW RESULTS:")

    # Exp 4
    if r4:
        bl_gaps = [r["gap"] for r in r4 if "baseline" in r["config"]]
        q4_gaps = [r["gap"] for r in r4 if "quartic" in r["config"]]
        bl_g = sum(bl_gaps)/len(bl_gaps) if bl_gaps else 0
        q4_g = sum(q4_gaps)/len(q4_gaps) if q4_gaps else 0
        status = "YES" if bl_g > q4_g * 1.2 else "NO" if abs(bl_g-q4_g)/max(bl_g,q4_g,1e-10) < 0.1 else "MAYBE"
        print(f"  Exp 4 (Train-val gap):        bl={bl_g:.4f} q4={q4_g:.4f} → Regularization: {status}")

    # Exp 5
    if r5:
        r8 = next((r for r in r5 if r["window"] == 8), None)
        r256 = next((r for r in r5 if r["window"] == 256), None)
        if r8 and r256:
            status = "YES" if r8["eff_rank"] < r256["eff_rank"] * 0.8 else "NO"
            print(f"  Exp 5 (Coupling rank):        w8={r8['eff_rank']:.1f} w256={r256['eff_rank']:.1f} → Coupling: {status}")

    # Exp 6
    if r6:
        bl_s = [r["stability"] for r in r6 if "baseline" in r["config"]]
        q4_s = [r["stability"] for r in r6 if "quartic" in r["config"]]
        bl_sm = sum(bl_s)/len(bl_s) if bl_s else 0
        q4_sm = sum(q4_s)/len(q4_s) if q4_s else 0
        status = "YES" if q4_sm > bl_sm * 1.2 else "NO" if abs(bl_sm-q4_sm)/max(bl_sm,q4_sm,1e-10) < 0.1 else "MAYBE"
        print(f"  Exp 6 (Landscape smooth):     bl={bl_sm:.4f} q4={q4_sm:.4f} → Smoother: {status}")

    # Exp 7 — sign counts, not the mean. See summarize_exp7 for the full report.
    if r7:
        pairs = []
        for sd in sorted({r["seed"] for r in r7.values()}):
            A = r7.get(f"A_full_s{sd}", {}).get("final_bpb")
            B = r7.get(f"B_quartic_s{sd}", {}).get("final_bpb")
            F = r7.get(f"F_switch_s{sd}", {}).get("final_bpb")
            if None not in (A, B, F) and math.isfinite(F):
                pairs.append((A - F, B - F))
        if pairs:
            n = len(pairs)
            nFA = sum(1 for d, _ in pairs if d > 0)
            nFB = sum(1 for _, d in pairs if d > 0)
            print(f"  Exp 7 (Removal test):         F beats A on {nFA}/{n} seeds, "
                  f"F beats B on {nFB}/{n} seeds")

    # Save all results. Merge into any existing file so running a subset of
    # experiments (e.g. --exp4 --exp5 --exp6) does not wipe results from
    # experiments that were not re-run this time (e.g. a cached exp7).
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "mechanism_disambiguation.json"
    combined = {}
    if out_path.exists():
        try:
            combined = json.load(open(out_path))
        except (json.JSONDecodeError, OSError):
            combined = {}
    for key, val in (("exp4", r4), ("exp5", r5), ("exp6", r6)):
        if val is not None:
            combined[key] = val
    if r7:
        # A partial run (stop_step) is a diagnostic, not a result: it stops
        # mid-schedule so its final bpb is not comparable to anything. Never let
        # one overwrite the canonical exp7 block.
        partial = any(v.get("stop_step") is not None for v in r7.values())
        if partial:
            print("  Exp 7: partial run — canonical exp7 block left untouched.")
        else:
            # Endpoints only here (full curves live in exp7_curriculum_sw*.json).
            # A switch-point sweep writes to its own key so it cannot clobber the
            # canonical 10k result.
            r7_save = {}
            sw = 10000
            for k, v in r7.items():
                r7_save[k] = {kk: vv for kk, vv in v.items() if kk != "curve"}
                if "curve" in v:
                    r7_save[k]["curve_start"] = v["curve"][0] if v["curve"] else None
                    r7_save[k]["curve_end"] = v["curve"][-1] if v["curve"] else None
                sw = v.get("switch_step", sw)
            combined["exp7" if sw == 10000 else f"exp7_sw{sw}"] = r7_save
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Results saved to {out_path}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Mechanism Disambiguation Experiments")
    parser.add_argument("--exp4", action="store_true", help="Train-val gap (~5 min)")
    parser.add_argument("--exp5", action="store_true", help="Gradient covariance (~30 min)")
    parser.add_argument("--exp6", action="store_true", help="Landscape smoothness (~10 min)")
    parser.add_argument("--exp7", action="store_true",
                        help="Window-removal test (~1.3 h per seed)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="How many of the default seeds to use for exp7")
    parser.add_argument("--seed-list", type=str, default=None,
                        help="Explicit exp7 seeds, e.g. 42,137,256,789,1337 "
                             "(overrides --seeds)")
    parser.add_argument("--switch-step", type=int, default=10000,
                        help="Step at which exp7 removes the windows")
    parser.add_argument("--total-steps", type=int, default=20000,
                        help="Total steps for the exp7 F arm")
    parser.add_argument("--switch-mode", choices=["full", "full_masked"], default="full",
                        help="What to switch to. 'full' clears arch_cfg (also changes "
                             "the attention kernel); 'full_masked' keeps the "
                             "masked-softmax path with all-full windows, isolating "
                             "the mask change from the kernel change")
    parser.add_argument("--post-switch-warmup", type=int, default=0,
                        help="Re-warm the LR over N steps after the switch")
    parser.add_argument("--reset-optimizer", action="store_true",
                        help="Reset Adam state at the switch (stale second-moment test)")
    parser.add_argument("--ramp-steps", type=int, default=0,
                        help="Ramp windows to full linearly over N steps instead of "
                             "switching discontinuously")
    parser.add_argument("--trace-window", type=int, default=0,
                        help="Log per-step loss / grad norm / max Adam update over "
                             "[switch-N, switch+5N]. Use ~100 to diagnose divergence")
    parser.add_argument("--stop-step", type=int, default=None,
                        help="Stop after this step. The LR schedule stays keyed to "
                             "--total-steps, so the conditions at the switch are "
                             "unchanged (shortening --total-steps would move the "
                             "cosine and change the LR at the switch)")
    parser.add_argument("--save-switch-ckpt", type=str, default=None, metavar="DIR",
                        help="Save model+optimizer+RNG state at the switch, per seed")
    parser.add_argument("--load-switch-ckpt", type=str, default=None, metavar="CKPT",
                        help="Start at the switch from a saved state, so several "
                             "interventions share one identical pre-switch state")
    parser.add_argument("--tag", type=str, default="",
                        help="Suffix for output filenames, so treatments in a "
                             "diagnosis sweep do not overwrite each other")
    parser.add_argument("--resume-seeds", action="store_true",
                        help="Skip seeds already present in the tag's result "
                             "file. Results are written after every seed, so an "
                             "interrupted sweep resumes at the seed it died on")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not any([args.exp4, args.exp5, args.exp6, args.exp7, args.all]):
        parser.print_help()
        return

    if args.seed_list:
        seeds = [int(s) for s in args.seed_list.split(",") if s.strip()]
    else:
        seeds = [42, 137, 256][:args.seeds]

    r4 = experiment4() if (args.exp4 or args.all) else None
    r5 = experiment5() if (args.exp5 or args.all) else None
    r6 = experiment6() if (args.exp6 or args.all) else None
    r7 = experiment7(seeds=seeds, switch_step=args.switch_step,
                     total_steps=args.total_steps, switch_mode=args.switch_mode,
                     post_switch_warmup=args.post_switch_warmup,
                     reset_optimizer=args.reset_optimizer,
                     ramp_steps=args.ramp_steps,
                     trace_window=args.trace_window,
                     stop_step=args.stop_step,
                     save_switch_ckpt=args.save_switch_ckpt,
                     load_switch_ckpt=args.load_switch_ckpt,
                     tag=args.tag, resume_seeds=args.resume_seeds) \
        if (args.exp7 or args.all) else None

    print_summary(r4, r5, r6, r7)


if __name__ == "__main__":
    main()
