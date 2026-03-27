"""
NeuroGen: Mechanism Disambiguation Experiments (4-7)

Tests surviving hypotheses after ruling out gradient noise removal,
softmax coupling, and variance reduction.

Exp 4: Train-val gap (implicit regularization)
Exp 5: Gradient covariance rank (parameter coupling)
Exp 6: Gradient stability of trained models (landscape smoothness)
Exp 7: Remove windows mid-training (curriculum vs structure)

Usage:
    python experiment_mechanism.py --exp4
    python experiment_mechanism.py --exp5
    python experiment_mechanism.py --exp6
    python experiment_mechanism.py --exp7 --seeds 1
    python experiment_mechanism.py --all
"""

import argparse, json, math, time, sys, os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from prepare import load_data, get_batch, evaluate_val_bpb, VOCAB_SIZE, MAX_SEQ_LEN
from train_r4 import (GPT, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS, DEVICE,
                       BATCH_SIZE, ARCHS, compute_window_mask, rms_norm)
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
def experiment7(n_seeds: int = 3):
    """Train with quartic windows for 10k steps, then remove for 10k more."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 7: Remove Windows Mid-Training")
    print("=" * 70)
    print("  Hypothesis: Windows create permanent specialization vs temporary curriculum")
    print("  Configs:")
    print("    A: Full attention for all 20k steps (baseline)")
    print("    B: Quartic windows for all 20k steps")
    print("    F: Quartic 10k steps → full attention 10k steps\n")
    print("  F ≈ B → hierarchy is permanent (structural specialization)")
    print("  F worse than B → windows needed permanently (ongoing constraint)")
    print("  F better than B → windows become ceiling (pure curriculum)\n")

    train_data = load_data("train")
    val_data = load_data("val")
    seeds = [42, 137, 256][:n_seeds]
    total_steps = 20000
    switch_step = 10000
    eval_interval = 500

    all_results = {}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")

        # Config A & B: load from existing validation results if available
        for label, arch_name in [("A_full", "baseline"), ("B_quartic", "window_power_4.0")]:
            json_path = f"validation_results/{arch_name}_s{seed}.json"
            if os.path.exists(json_path):
                d = json.load(open(json_path))
                final_bpb = d["summary"]["final_vbpb"]
                curve = [(p["step"], p["val_bpb"]) for p in d["curve"]]
                key = f"{label}_s{seed}"
                all_results[key] = {"config": label, "seed": seed,
                                     "final_bpb": final_bpb, "source": "cached",
                                     "curve": curve}
                print(f"  {label}: loaded from cache → bpb={final_bpb:.4f}")
            else:
                print(f"  {label}: {json_path} not found — skipping")

        # Config F: quartic 10k → full 10k (must train)
        print(f"\n--- F: Quartic 10k → Full 10k (seed={seed}) ---")
        torch.manual_seed(seed)

        arch_cfg_quartic = ARCHS.get("window_power_4.0", {})
        model = GPT(VOCAB_SIZE, MAX_SEQ_LEN, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS,
                     arch_cfg=arch_cfg_quartic).to(DEVICE)

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
        min_lr = LR / 10
        t0 = time.time()
        curve = []

        for step in range(total_steps + 1):
            # Switch architecture at midpoint
            if step == switch_step:
                print(f"  >>> SWITCHING from quartic to full attention at step {step}")
                model.arch_cfg = {}  # Remove window config

            # Eval
            if step % eval_interval == 0:
                model.eval()
                vbpb = evaluate_val_bpb(model, val_data, BATCH_SIZE, MAX_SEQ_LEN, DEVICE)
                elapsed = time.time() - t0
                phase = "quartic" if step < switch_step else "full"
                curve.append((step, round(vbpb, 4)))
                if step % 2000 == 0:
                    print(f"  step:{step:6d}  vbpb:{vbpb:.4f}  phase:{phase}  "
                          f"time:{elapsed:.0f}s")
                model.train()

            if step >= total_steps:
                break

            # Training step
            x, y = get_batch(train_data, BATCH_SIZE, MAX_SEQ_LEN, DEVICE)
            cur_lr = get_lr(step, WARMUP, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr
            _, loss = model(x, y, step=step, total_steps=total_steps)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        final_bpb = curve[-1][1]
        key = f"F_switch_s{seed}"
        all_results[key] = {"config": "F_switch", "seed": seed,
                             "final_bpb": final_bpb, "source": "trained",
                             "curve": curve}
        print(f"  FINAL: F_switch seed={seed} bpb={final_bpb:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 7 RESULTS")
    print(f"{'='*70}")
    print(f"{'config':<30} {'seed':>5} {'final_bpb':>10} {'vs baseline':>12}")
    print("-" * 60)

    bl_bpbs, q4_bpbs, f_bpbs = [], [], []
    for key in sorted(all_results.keys()):
        r = all_results[key]
        bl_key = f"A_full_s{r['seed']}"
        bl_bpb = all_results.get(bl_key, {}).get("final_bpb", 0)
        delta = (bl_bpb - r["final_bpb"]) / bl_bpb * 100 if bl_bpb else 0
        delta_str = f"{delta:>+11.2f}%" if r["config"] != "A_full" else f"{'—':>12}"
        print(f"{r['config']:<30} {r['seed']:>5} {r['final_bpb']:>10.4f} {delta_str}")
        if "A_full" in r["config"]: bl_bpbs.append(r["final_bpb"])
        elif "B_quartic" in r["config"]: q4_bpbs.append(r["final_bpb"])
        elif "F_switch" in r["config"]: f_bpbs.append(r["final_bpb"])

    if bl_bpbs and q4_bpbs and f_bpbs:
        bl_m = sum(bl_bpbs) / len(bl_bpbs)
        q4_m = sum(q4_bpbs) / len(q4_bpbs)
        f_m = sum(f_bpbs) / len(f_bpbs)
        print(f"\n  Means: A={bl_m:.4f}  B={q4_m:.4f}  F={f_m:.4f}")
        print(f"  B vs A: {(bl_m-q4_m)/bl_m*100:+.2f}%")
        print(f"  F vs A: {(bl_m-f_m)/bl_m*100:+.2f}%")
        print(f"  F vs B: {(q4_m-f_m)/q4_m*100:+.2f}%")

        if abs(f_m - q4_m) / q4_m < 0.003:
            print(f"\n  → F ≈ B: hierarchy is permanent after 10k steps")
            print(f"    Supports: STRUCTURAL SPECIALIZATION")
        elif f_m > q4_m * 1.003:
            print(f"\n  → F worse than B: removing windows hurts")
            print(f"    Supports: ONGOING CONSTRAINT (not just curriculum)")
        elif f_m < q4_m * 0.997:
            print(f"\n  → F better than B: windows become a ceiling after early training")
            print(f"    Supports: CURRICULUM EFFECT")
        else:
            print(f"\n  → Inconclusive (F ≈ B within noise)")

    return all_results


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

    # Exp 7
    if r7:
        bl_bpbs = [r["final_bpb"] for r in r7.values() if "A_full" in r["config"]]
        q4_bpbs = [r["final_bpb"] for r in r7.values() if "B_quartic" in r["config"]]
        f_bpbs = [r["final_bpb"] for r in r7.values() if "F_switch" in r["config"]]
        if bl_bpbs and q4_bpbs and f_bpbs:
            f_m = sum(f_bpbs)/len(f_bpbs)
            q4_m = sum(q4_bpbs)/len(q4_bpbs)
            if abs(f_m - q4_m) / q4_m < 0.003:
                status = "STRUCTURAL (F≈B)"
            elif f_m > q4_m * 1.003:
                status = "ONGOING CONSTRAINT (F<B)"
            elif f_m < q4_m * 0.997:
                status = "CURRICULUM (F>B)"
            else:
                status = "INCONCLUSIVE"
            print(f"  Exp 7 (Curriculum test):      F={f_m:.4f} B={q4_m:.4f} → {status}")

    # Save all results
    RESULTS_DIR.mkdir(exist_ok=True)
    combined = {"exp4": r4, "exp5": r5, "exp6": r6}
    if r7:
        # Convert curves to just endpoints for JSON size
        r7_save = {}
        for k, v in r7.items():
            r7_save[k] = {kk: vv for kk, vv in v.items() if kk != "curve"}
            if "curve" in v:
                r7_save[k]["curve_start"] = v["curve"][0] if v["curve"] else None
                r7_save[k]["curve_end"] = v["curve"][-1] if v["curve"] else None
        combined["exp7"] = r7_save
    with open(RESULTS_DIR / "mechanism_disambiguation.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR / 'mechanism_disambiguation.json'}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Mechanism Disambiguation Experiments")
    parser.add_argument("--exp4", action="store_true", help="Train-val gap (~5 min)")
    parser.add_argument("--exp5", action="store_true", help="Gradient covariance (~30 min)")
    parser.add_argument("--exp6", action="store_true", help="Landscape smoothness (~10 min)")
    parser.add_argument("--exp7", action="store_true", help="Curriculum test (~3 hours)")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds for exp7")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not any([args.exp4, args.exp5, args.exp6, args.exp7, args.all]):
        parser.print_help()
        return

    r4 = experiment4() if (args.exp4 or args.all) else None
    r5 = experiment5() if (args.exp5 or args.all) else None
    r6 = experiment6() if (args.exp6 or args.all) else None
    r7 = experiment7(n_seeds=args.seeds) if (args.exp7 or args.all) else None

    print_summary(r4, r5, r6, r7)


if __name__ == "__main__":
    main()
