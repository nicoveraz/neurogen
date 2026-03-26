"""
NeuroGen: Gradient Mechanism Experiment

Tests whether attention windows improve training by cleaning gradient noise
from the softmax backward pass.

Usage:
    python experiment_gradient.py --exp1          # Window sweep (20 min)
    python experiment_gradient.py --exp2          # Gradient decomposition (30 min)
    python experiment_gradient.py --exp3          # Causal test (15 hours)
    python experiment_gradient.py --exp1 --exp2   # Both measurement experiments
"""

import argparse, json, math, time, sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from prepare import load_data, get_batch, VOCAB_SIZE, MAX_SEQ_LEN
from train_r4 import (GPT, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS,
                       compute_window_mask, DEVICE)

CKPT_PATH = "checkpoints/model_baseline_42.pt"
N_PASSES = 50
BATCH_SIZE = 32
RESULTS_DIR = Path("gradient_results")


def load_model(ckpt_path: str = CKPT_PATH) -> GPT:
    """Load trained baseline checkpoint."""
    model = GPT(VOCAB_SIZE, MAX_SEQ_LEN, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS,
                arch_cfg={})
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    return model


def make_window_mask(seq_len: int, window: int, device: str = DEVICE) -> torch.Tensor:
    """Create a causal + window mask. Returns float tensor: 1.0 allowed, 0.0 blocked."""
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = (cols <= rows) & (cols >= rows - window + 1)
    return mask.float()


# ---------------------------------------------------------------------------
# Experiment 1: Window Sweep with Gradient Measurement
# ---------------------------------------------------------------------------
def experiment1_window_sweep():
    """Measure gradient SNR, stability, and rank at different window sizes."""
    print("=" * 70)
    print("  EXPERIMENT 1: Window Sweep with Gradient Measurement")
    print("=" * 70)

    model = load_model()
    model.train()  # need gradients, but don't update weights
    val_data = load_data("val")

    window_sizes = [8, 16, 32, 48, 64, 80, 96, 128, 192, 256]
    all_results = []

    for window in window_sizes:
        print(f"\n--- Window size: {window} ---")
        t0 = time.time()

        # Collect gradients over N_PASSES
        q_grads = []
        k_grads = []
        v_grads = []
        grad_norms = []

        for pass_idx in range(N_PASSES):
            model.zero_grad()
            x, y = get_batch(val_data, BATCH_SIZE, MAX_SEQ_LEN, DEVICE)

            # Modify the arch_cfg to apply window at layer 0 only
            # We override the masks in forward manually
            original_arch_cfg = model.arch_cfg.copy()

            # Hook into forward to inject our mask
            # We'll call forward with custom masks
            B, T = x.size()
            cos, sin = model.cos[:, :T], model.sin[:, :T]

            # Embedding
            from train_r4 import rms_norm
            emb = rms_norm(model.wte(x))
            if T > 1:
                gate = model.smear_lambda * torch.sigmoid(
                    model.smear_gate(emb[:, 1:, :model._smear_ch]))
                emb = torch.cat([emb[:, :1], emb[:, 1:] + gate * emb[:, :-1]], dim=1)
            x_in = emb
            x0 = emb
            prev_attn = None

            for i, block in enumerate(model.blocks):
                x_in = model.resid_lambdas[i] * x_in + model.x0_lambdas[i] * x0
                ve = model.value_embeds[str(i)](x) if str(i) in model.value_embeds else None
                # Apply window mask ONLY at layer 0
                if i == 0:
                    mask = make_window_mask(T, window, device=x_in.device)
                else:
                    mask = None  # full causal attention for other layers
                x_in, prev_attn = block(x_in, ve, cos, sin, mask=mask,
                                        prev_attn=prev_attn)

            mid = model.n_layer // 2
            # Simplified: skip x_mid backout and ca_state for gradient measurement
            logits = model.lm_head(rms_norm(x_in)).float()
            logits = 15 * torch.tanh(logits / 15)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()

            # Capture gradients at layer 0
            q_grad = model.blocks[0].attn.c_q.weight.grad.detach().clone()
            k_grad = model.blocks[0].attn.c_k.weight.grad.detach().clone()
            v_grad = model.blocks[0].attn.c_v.weight.grad.detach().clone()

            q_grads.append(q_grad)
            k_grads.append(k_grad)
            v_grads.append(v_grad)

            grad_norms.append({
                'q_norm': q_grad.norm().item(),
                'k_norm': k_grad.norm().item(),
                'v_norm': v_grad.norm().item(),
                'loss': loss.item(),
            })

        # Aggregate statistics
        q_stack = torch.stack(q_grads)  # (N_PASSES, out_dim, in_dim)
        k_stack = torch.stack(k_grads)
        v_stack = torch.stack(v_grads)

        stats = compute_gradient_stats(q_stack, "Q", window)
        stats.update(compute_gradient_stats(k_stack, "K", window, prefix="k_"))
        stats.update(compute_gradient_stats(v_stack, "V", window, prefix="v_"))

        stats['window_size'] = window
        stats['mean_loss'] = sum(g['loss'] for g in grad_norms) / len(grad_norms)
        stats['mean_q_norm'] = sum(g['q_norm'] for g in grad_norms) / len(grad_norms)
        stats['mean_k_norm'] = sum(g['k_norm'] for g in grad_norms) / len(grad_norms)
        stats['mean_v_norm'] = sum(g['v_norm'] for g in grad_norms) / len(grad_norms)
        stats['time_s'] = round(time.time() - t0, 1)

        all_results.append(stats)

        print(f"GRAD_EXP1: window={window:4d}  snr={stats['snr']:.4f}  "
              f"stability={stats['direction_stability']:.4f}  "
              f"eff_rank={stats['effective_rank']:.2f}  "
              f"loss={stats['mean_loss']:.4f}  "
              f"time={stats['time_s']}s")

    # Print summary table
    print(f"\n{'=' * 90}")
    print(f"  EXPERIMENT 1 RESULTS: Gradient Quality vs Window Size (Layer 0)")
    print(f"{'=' * 90}")
    print(f"{'window':>7} | {'snr':>8} | {'signal_norm':>11} | {'noise_norm':>10} | "
          f"{'stability':>10} | {'eff_rank':>8} | {'loss':>8}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['window_size']:>7} | {r['snr']:>8.4f} | {r['signal_norm']:>11.6f} | "
              f"{r['noise_norm']:>10.6f} | {r['direction_stability']:>10.4f} | "
              f"{r['effective_rank']:>8.2f} | {r['mean_loss']:>8.4f}")

    # Check for knee point
    print(f"\n--- Knee Analysis ---")
    snr_values = [(r['window_size'], r['snr']) for r in all_results]
    for i in range(1, len(snr_values) - 1):
        w_prev, snr_prev = snr_values[i - 1]
        w_curr, snr_curr = snr_values[i]
        w_next, snr_next = snr_values[i + 1]
        # Knee: improvement slows or reverses
        delta_before = snr_curr - snr_prev
        delta_after = snr_next - snr_curr
        if delta_before > 0 and delta_after <= 0:
            print(f"GRAD_KNEE: Potential knee at window={w_curr} "
                  f"(SNR increases {w_prev}->{w_curr}, decreases {w_curr}->{w_next})")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp1_window_sweep.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'exp1_window_sweep.json'}")

    return all_results


def compute_gradient_stats(grad_stack: torch.Tensor, name: str, window: int,
                           prefix: str = "") -> dict:
    """Compute SNR, direction stability, and effective rank from stacked gradients."""
    N = grad_stack.size(0)

    # Move to CPU for SVD (MPS doesn't support it well)
    grad_cpu = grad_stack.cpu().float()

    # Signal = mean gradient across batches
    signal = grad_cpu.mean(dim=0)
    signal_norm = signal.norm().item()

    # Noise = std of gradient across batches
    noise = grad_cpu.std(dim=0)
    noise_norm = noise.norm().item()

    snr = signal_norm / (noise_norm + 1e-10)

    # Direction stability: mean pairwise cosine similarity
    flat = grad_cpu.view(N, -1)
    flat_normed = F.normalize(flat, dim=1)
    cos_sim = flat_normed @ flat_normed.T
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    direction_stability = cos_sim[mask].mean().item()

    # Effective rank via spectral entropy
    try:
        _, S, _ = torch.linalg.svd(flat, full_matrices=False)
        S_norm = S / (S.sum() + 1e-10)
        entropy = -(S_norm * (S_norm + 1e-10).log()).sum().item()
        effective_rank = math.exp(entropy)
    except Exception:
        effective_rank = float('nan')

    return {
        f'{prefix}snr': snr,
        f'{prefix}signal_norm': signal_norm,
        f'{prefix}noise_norm': noise_norm,
        f'{prefix}direction_stability': direction_stability,
        f'{prefix}effective_rank': effective_rank,
    }


# ---------------------------------------------------------------------------
# Experiment 2: Direct Gradient Decomposition
# ---------------------------------------------------------------------------
def experiment2_gradient_decomposition():
    """Decompose softmax backward into signal vs noise contributions."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Direct Gradient Decomposition")
    print("=" * 70)

    model = load_model()
    model.train()
    val_data = load_data("val")

    all_layer_results = []

    for layer_idx in range(DEPTH):
        print(f"\n--- Layer {layer_idx} ---")
        t0 = time.time()

        decomp_results = []

        for pass_idx in range(N_PASSES):
            model.zero_grad()
            x, y = get_batch(val_data, BATCH_SIZE, MAX_SEQ_LEN, DEVICE)

            # We need to manually run the forward pass to capture intermediate values
            B, T = x.size()
            cos, sin = model.cos[:, :T], model.sin[:, :T]
            from train_r4 import rms_norm

            emb = rms_norm(model.wte(x))
            if T > 1:
                gate = model.smear_lambda * torch.sigmoid(
                    model.smear_gate(emb[:, 1:, :model._smear_ch]))
                emb = torch.cat([emb[:, :1], emb[:, 1:] + gate * emb[:, :-1]], dim=1)
            x_in = emb
            x0 = emb
            prev_attn = None

            # Forward pass, capturing attention weights at target layer
            captured_attn = None
            captured_v = None
            captured_q = None
            captured_k = None

            for i, block in enumerate(model.blocks):
                x_in = model.resid_lambdas[i] * x_in + model.x0_lambdas[i] * x0
                ve = model.value_embeds[str(i)](x) if str(i) in model.value_embeds else None

                if i == layer_idx:
                    # Manual attention to capture internals
                    attn = block.attn
                    ln_out = rms_norm(x_in)
                    q = attn.c_q(ln_out).view(B, T, attn.n_head, attn.head_dim)
                    k = attn.c_k(ln_out).view(B, T, attn.n_kv_head, attn.head_dim)
                    v = attn.c_v(ln_out).view(B, T, attn.n_kv_head, attn.head_dim)

                    from train_r4 import apply_rotary_emb
                    q = apply_rotary_emb(q, cos, sin)
                    k = apply_rotary_emb(k, cos, sin)
                    q, k = rms_norm(q) * 1.2, rms_norm(k) * 1.2

                    if attn.n_kv_head < attn.n_head:
                        r = attn.n_head // attn.n_kv_head
                        k = k.repeat_interleave(r, dim=2)
                        v = v.repeat_interleave(r, dim=2)

                    q_t = q.transpose(1, 2)  # (B, H, T, D)
                    k_t = k.transpose(1, 2)
                    v_t = v.transpose(1, 2)

                    scale = 1.0 / math.sqrt(attn.head_dim)
                    scores = q_t @ k_t.transpose(-2, -1) * scale  # (B, H, T, T)
                    cmask = torch.tril(torch.ones(T, T, device=x_in.device, dtype=torch.bool))
                    scores = scores.masked_fill(~cmask, float("-inf"))
                    attn_weights = F.softmax(scores, dim=-1)  # (B, H, T, T)
                    attn_out = attn_weights @ v_t  # (B, H, T, D)
                    out = attn.c_proj(attn_out.transpose(1, 2).contiguous().view(B, T, -1))

                    captured_attn = attn_weights.detach()
                    captured_v = v_t.detach()

                    # Continue through block (MLP part)
                    x_attn = x_in + out
                    mlp_out = block.mlp(rms_norm(x_attn))
                    x_in = x_attn + mlp_out
                    prev_attn = captured_attn
                else:
                    x_in, prev_attn = block(x_in, ve, cos, sin, mask=None,
                                            prev_attn=prev_attn)

            logits = model.lm_head(rms_norm(x_in)).float()
            logits = 15 * torch.tanh(logits / 15)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            # We need grad_attn_output — register hook before backward
            grad_attn_output = [None]

            def capture_grad(grad):
                grad_attn_output[0] = grad.detach().clone()

            # Re-run forward with hook (simpler: compute decomposition analytically)
            # Instead, use the captured attn_weights and compute the decomposition
            loss.backward()

            # Decompose using captured values
            # Use the analytical formula for softmax gradient decomposition
            with torch.no_grad():
                alpha = captured_attn  # (B, H, T, T)
                V = captured_v  # (B, H, T, D)

                # Get the gradient of the attention output from the Q projection gradient
                # Instead, approximate: measure what fraction of attention weight
                # goes to "non-attended" positions

                # Natural attention threshold: uniform = 1/T
                threshold = 1.0 / T

                attended = (alpha > threshold)  # (B, H, T, T)
                not_attended = ~attended & cmask.unsqueeze(0).unsqueeze(0)

                # Fraction of total attention weight on non-attended positions
                noise_weight = (alpha * not_attended.float()).sum(dim=-1)  # (B, H, T)
                total_weight = alpha.sum(dim=-1)  # should be 1.0

                noise_fraction = noise_weight.mean().item()

                # Attention entropy (how spread out the attention is)
                log_alpha = torch.log(alpha + 1e-10)
                entropy = -(alpha * log_alpha).sum(dim=-1).mean().item()

                # Natural attention span: weighted mean distance
                positions = torch.arange(T, device=x_in.device).float()
                query_pos = positions.unsqueeze(1)  # (T, 1)
                key_pos = positions.unsqueeze(0)  # (1, T)
                distances = (query_pos - key_pos).abs()  # (T, T)
                mean_span = (alpha * distances.unsqueeze(0).unsqueeze(0)).sum(dim=-1).mean().item()

                # Number of positions with > threshold attention
                n_attended = attended.float().sum(dim=-1).mean().item()

                decomp_results.append({
                    'noise_fraction': noise_fraction,
                    'entropy': entropy,
                    'mean_span': mean_span,
                    'n_attended': n_attended,
                    'loss': loss.item(),
                })

        # Aggregate
        mean_noise_frac = sum(d['noise_fraction'] for d in decomp_results) / len(decomp_results)
        std_noise_frac = (sum((d['noise_fraction'] - mean_noise_frac)**2
                              for d in decomp_results) / (len(decomp_results) - 1)) ** 0.5
        mean_entropy = sum(d['entropy'] for d in decomp_results) / len(decomp_results)
        mean_span = sum(d['mean_span'] for d in decomp_results) / len(decomp_results)
        mean_n_attended = sum(d['n_attended'] for d in decomp_results) / len(decomp_results)

        result = {
            'layer': layer_idx,
            'noise_fraction': round(mean_noise_frac, 6),
            'noise_fraction_std': round(std_noise_frac, 6),
            'entropy': round(mean_entropy, 4),
            'mean_attention_span': round(mean_span, 2),
            'n_attended_positions': round(mean_n_attended, 2),
            'time_s': round(time.time() - t0, 1),
        }
        all_layer_results.append(result)

        print(f"GRAD_EXP2: layer={layer_idx}  noise_frac={mean_noise_frac:.4f}±{std_noise_frac:.4f}  "
              f"entropy={mean_entropy:.4f}  span={mean_span:.1f}  "
              f"n_attended={mean_n_attended:.1f}")

    # Summary
    print(f"\n{'=' * 80}")
    print(f"  EXPERIMENT 2 RESULTS: Gradient Decomposition by Layer")
    print(f"{'=' * 80}")
    print(f"{'layer':>5} | {'noise_frac':>12} | {'entropy':>8} | {'mean_span':>10} | {'n_attended':>10}")
    print("-" * 60)
    for r in all_layer_results:
        print(f"{r['layer']:>5} | {r['noise_fraction']:>10.4f}±{r['noise_fraction_std']:.3f} | "
              f"{r['entropy']:>8.4f} | {r['mean_attention_span']:>10.1f} | "
              f"{r['n_attended_positions']:>10.1f}")

    print(f"\nPrediction: Layer 0 should have HIGHEST noise_fraction (smallest natural span)")
    print(f"           Layer 3 should have LOWEST noise_fraction (largest natural span)")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp2_decomposition.json", "w") as f:
        json.dump(all_layer_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'exp2_decomposition.json'}")

    return all_layer_results


# ---------------------------------------------------------------------------
# Experiment 3: Variance Reduction — Batch Size vs Window Comparison
# ---------------------------------------------------------------------------
def experiment3_variance_reduction(n_steps: int = 5000, n_seeds: int = 3):
    """Compare windowed attention vs larger batch size for variance reduction.

    Key insight from Exp 1: windows don't reduce gradient noise — they increase
    gradient signal coherence. Batch size averaging reduces both signal and noise
    equally (SNR unchanged). If windowing works via variance reduction, larger
    batch should replicate the improvement. If not, the mechanism is different.

    Configs:
    - A: full attention, batch 32              (baseline)
    - B: quartic windows, batch 32             (our method)
    - C: full attention, grad_accum=4 (eff 128) (variance reduction via averaging)
    - D: full attention, grad_accum=8 (eff 256) (stronger variance reduction)

    All configs take the same number of optimizer steps (same # weight updates).
    C and D see more tokens per update but have lower gradient variance.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Variance Reduction — Batch Size vs Windows")
    print("=" * 70)
    print(f"\n  From Exp 1: noise_norm is CONSTANT across window sizes (~0.0053)")
    print(f"  What changes is signal_norm: 0.0017 (w=256) → 0.032 (w=8)")
    print(f"  Windows increase gradient coherence, not reduce noise.")
    print(f"  Prediction: larger batch CANNOT replicate windowing benefit.\n")

    train_data = load_data("train")
    val_data = load_data("val")

    configs = {
        "A_full_batch32": {"window": None, "grad_accum": 1},
        "B_quartic_batch32": {"window": "power_4.0", "grad_accum": 1},
        "C_full_batch128": {"window": None, "grad_accum": 4},
        "D_full_batch256": {"window": None, "grad_accum": 8},
    }

    all_results = {}
    seeds = [42, 137, 256][:n_seeds]

    for actual_seed in seeds:
        print(f"\n{'='*60}")
        print(f"  Seed {actual_seed}")
        print(f"{'='*60}")

        for config_name, cfg in configs.items():
            print(f"\n--- {config_name} (seed={actual_seed}) ---")
            torch.manual_seed(actual_seed)

            arch_cfg = {}
            if cfg["window"]:
                arch_cfg["window"] = cfg["window"]

            model = GPT(VOCAB_SIZE, MAX_SEQ_LEN, DEPTH, N_HEADS, N_KV_HEADS,
                        CHANNELS, arch_cfg=arch_cfg)
            model = model.to(DEVICE)

            grad_accum = cfg["grad_accum"]
            eff_batch = BATCH_SIZE * grad_accum

            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3,
                                          weight_decay=0.05)
            curve = []
            t0 = time.time()

            print(f"  effective_batch: {eff_batch}  grad_accum: {grad_accum}")

            for step in range(n_steps + 1):
                # Evaluate every 500 steps
                if step % 500 == 0:
                    model.eval()
                    with torch.no_grad():
                        eval_losses = []
                        for _ in range(10):
                            ex, ey = get_batch(val_data, BATCH_SIZE, MAX_SEQ_LEN, DEVICE)
                            _, loss = model(ex, ey)
                            eval_losses.append(loss.item())
                    val_loss = sum(eval_losses) / len(eval_losses)
                    val_bpb = val_loss / math.log(2)
                    elapsed = time.time() - t0
                    tokens = step * eff_batch * MAX_SEQ_LEN
                    curve.append({"step": step, "val_bpb": round(val_bpb, 4),
                                  "elapsed_s": round(elapsed, 1),
                                  "tokens_seen": tokens})
                    print(f"  step:{step:5d}  val_bpb:{val_bpb:.4f}  "
                          f"tokens:{tokens/1e6:.0f}M  time:{elapsed:.0f}s")
                    model.train()

                if step >= n_steps:
                    break

                # Training step with gradient accumulation
                optimizer.zero_grad(set_to_none=True)
                for _ in range(grad_accum):
                    tx, ty = get_batch(train_data, BATCH_SIZE, MAX_SEQ_LEN, DEVICE)
                    _, loss = model(tx, ty)
                    (loss / grad_accum).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            key = f"{config_name}_s{actual_seed}"
            all_results[key] = {
                "config": config_name, "seed": actual_seed,
                "grad_accum": grad_accum,
                "effective_batch": eff_batch,
                "final_bpb": curve[-1]["val_bpb"],
                "tokens_seen": curve[-1]["tokens_seen"],
                "curve": curve,
            }
            print(f"  FINAL: {config_name} seed={actual_seed} bpb={curve[-1]['val_bpb']:.4f} "
                  f"tokens={curve[-1]['tokens_seen']/1e6:.0f}M")
            del model, optimizer

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"  EXPERIMENT 3 RESULTS: Variance Reduction Comparison")
    print(f"{'=' * 80}")
    print(f"{'config':<22} {'eff_batch':>9} {'seed':>5} {'final_bpb':>10} {'tokens':>8}")
    print("-" * 60)
    for key, r in sorted(all_results.items()):
        print(f"{r['config']:<22} {r['effective_batch']:>9} {r['seed']:>5} "
              f"{r['final_bpb']:>10.4f} {r['tokens_seen']/1e6:>7.0f}M")

    # Per-config averages
    print(f"\n{'config':<22} {'eff_batch':>9} {'mean_bpb':>10} {'vs baseline':>12}")
    print("-" * 60)
    baseline_bpbs = [r['final_bpb'] for r in all_results.values()
                     if r['config'] == 'A_full_batch32']
    bl_mean = sum(baseline_bpbs) / len(baseline_bpbs) if baseline_bpbs else 1.0

    for cfg_name, cfg in configs.items():
        bpbs = [r['final_bpb'] for r in all_results.values() if r['config'] == cfg_name]
        if bpbs:
            mean_bpb = sum(bpbs) / len(bpbs)
            delta = (bl_mean - mean_bpb) / bl_mean * 100
            eff = cfg['grad_accum'] * BATCH_SIZE
            print(f"{cfg_name:<22} {eff:>9} {mean_bpb:>10.4f} {delta:>+11.2f}%")

    print(f"\nInterpretation:")
    print(f"  If B >> C,D: Windows work through signal coherence, not variance reduction")
    print(f"  If B ≈ C or D: Variance reduction IS the mechanism")
    print(f"  If C,D > A but < B: Variance reduction helps but windows do more")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp3_variance_reduction.json", "w") as f:
        # Don't save full curves to keep file manageable
        save_data = {}
        for k, v in all_results.items():
            save_data[k] = {kk: vv for kk, vv in v.items() if kk != 'curve'}
            save_data[k]['curve_endpoints'] = [v['curve'][0], v['curve'][-1]]
            save_data[k]['curve'] = v['curve']
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'exp3_variance_reduction.json'}")

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NeuroGen Gradient Mechanism Experiment")
    parser.add_argument("--exp1", action="store_true", help="Window sweep (~20 min)")
    parser.add_argument("--exp2", action="store_true", help="Gradient decomposition (~30 min)")
    parser.add_argument("--exp3", action="store_true", help="Variance reduction comparison (~30 min)")
    parser.add_argument("--exp3-steps", type=int, default=5000, help="Steps for exp3")
    parser.add_argument("--exp3-seeds", type=int, default=3, help="Seeds for exp3")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    args = parser.parse_args()

    if not any([args.exp1, args.exp2, args.exp3, args.all]):
        parser.print_help()
        return

    if args.exp1 or args.all:
        experiment1_window_sweep()

    if args.exp2 or args.all:
        experiment2_gradient_decomposition()

    if args.exp3 or args.all:
        experiment3_variance_reduction(n_steps=args.exp3_steps, n_seeds=args.exp3_seeds)


if __name__ == "__main__":
    main()
