"""
NeuroGen: Attention Entropy Analysis

Compares per-layer attention entropy between baseline and quartic models.
Forces manual attention computation to capture full attention weight matrices.

Usage:
    python analyze_attention_entropy.py
"""

import math, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from prepare import load_data, get_batch, VOCAB_SIZE, MAX_SEQ_LEN
from train_r4 import (GPT, ARCHS, DEPTH, CHANNELS, N_HEADS, N_KV_HEADS,
                       rms_norm, apply_rotary_emb, compute_window_mask,
                       get_device)

DEVICE = get_device()
N_BATCHES = 100
BATCH_SIZE = 8  # small batches for memory on MPS


def load_checkpoint(path: str):
    """Load model from checkpoint with correct arch config."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    arch_name = ckpt.get("arch", "baseline")
    arch_cfg = ARCHS.get(arch_name, {})
    model = GPT(VOCAB_SIZE, MAX_SEQ_LEN, DEPTH, N_HEADS, N_KV_HEADS,
                CHANNELS, arch_cfg=arch_cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, arch_name


def compute_attention_entropy(model, val_data, n_batches=N_BATCHES):
    """Compute mean attention entropy per layer by manually forwarding through the model.

    Returns dict: {layer_idx: {"mean_entropy": float, "per_head": [float, ...], "mean_max_attn": float}}
    """
    n_layer = model.n_layer
    # Accumulators: [n_layer][n_head] lists of entropy values
    entropy_accum = [[[] for _ in range(N_HEADS)] for _ in range(n_layer)]
    max_attn_accum = [[[] for _ in range(N_HEADS)] for _ in range(n_layer)]

    arch_cfg = model.arch_cfg
    win_mode = arch_cfg.get("window")

    with torch.no_grad():
        for batch_idx in range(n_batches):
            x, _ = get_batch(val_data, BATCH_SIZE, MAX_SEQ_LEN, DEVICE)
            B, T = x.size()
            cos, sin = model.cos[:, :T], model.sin[:, :T]

            # Embedding + smear (replicate model.forward)
            h = rms_norm(model.wte(x))
            if T > 1:
                gate = model.smear_lambda * torch.sigmoid(
                    model.smear_gate(h[:, 1:, :model._smear_ch]))
                h = torch.cat([h[:, :1], h[:, 1:] + gate * h[:, :-1]], dim=1)
            x0 = h
            prev_attn = None

            for li, block in enumerate(model.blocks):
                h = model.resid_lambdas[li] * h + model.x0_lambdas[li] * x0
                ve = model.value_embeds[str(li)](x) if str(li) in model.value_embeds else None

                normed = rms_norm(h)
                attn = block.attn

                # Manually compute attention weights (always, even for baseline)
                q = attn.c_q(normed).view(B, T, attn.n_head, attn.head_dim)
                k = attn.c_k(normed).view(B, T, attn.n_kv_head, attn.head_dim)
                v = attn.c_v(normed).view(B, T, attn.n_kv_head, attn.head_dim)

                if ve is not None and attn.ve_gate is not None:
                    ve_reshaped = ve.view(B, T, attn.n_kv_head, attn.head_dim)
                    gate_val = 3 * torch.sigmoid(attn.ve_gate(normed[..., :attn._ve_ch]))
                    v = v + gate_val.unsqueeze(-1) * ve_reshaped

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
                att = q_t @ k_t.transpose(-2, -1) * scale

                # Apply causal mask (always)
                cmask = torch.tril(torch.ones(T, T, device=DEVICE, dtype=torch.bool))

                # Apply window mask if this arch uses windows
                if win_mode:
                    wmask = compute_window_mask(T, li, n_layer, win_mode, DEVICE)
                    att = att.masked_fill(wmask[:T, :T] == 0, float("-inf"))
                else:
                    att = att.masked_fill(~cmask, float("-inf"))

                attn_weights = F.softmax(att, dim=-1)  # (B, H, T, T)

                # Compute entropy per head: -sum(p * log(p))
                # Clamp to avoid log(0)
                log_weights = torch.log(attn_weights.clamp(min=1e-10))
                entropy = -(attn_weights * log_weights).sum(dim=-1)  # (B, H, T)
                # Mean over positions and batch
                for head_idx in range(N_HEADS):
                    # Only use positions > 0 to avoid trivial position-0 entropy
                    ent_vals = entropy[:, head_idx, 1:].mean().item()
                    entropy_accum[li][head_idx].append(ent_vals)

                    max_attn_vals = attn_weights[:, head_idx, 1:, :].max(dim=-1).values.mean().item()
                    max_attn_accum[li][head_idx].append(max_attn_vals)

                # Continue forward pass through the block properly
                y = attn_weights @ v_t
                out = attn.c_proj(y.transpose(1, 2).contiguous().view(B, T, -1))

                # CA modulation (replicate block.forward logic)
                if block.ca_ch is not None and block.ca_mode in ("attn", "both", "multiscale"):
                    if block.ca_mode == "multiscale":
                        ca_signal = block.ca_ch(h, layer_idx=block.layer_idx)
                    else:
                        ca_signal = block.ca_ch(h)
                    out = out * (1 + torch.tanh(ca_signal) * 0.1)
                elif block.ca_ch is not None and block.ca_mode == "additive":
                    out = out + block.ca_ch(h) * 0.1

                h = h + out
                mlp_out = block.mlp(rms_norm(h))
                if block.ca_ch is not None and block.ca_mode == "both":
                    ca_signal = block.ca_ch(h)
                    mlp_out = mlp_out * (1 + torch.tanh(ca_signal) * 0.1)
                h = h + mlp_out

            if (batch_idx + 1) % 25 == 0:
                print(f"  batch {batch_idx + 1}/{n_batches}")

    # Aggregate
    results = {}
    for li in range(n_layer):
        per_head = [np.mean(entropy_accum[li][hi]) for hi in range(N_HEADS)]
        per_head_max = [np.mean(max_attn_accum[li][hi]) for hi in range(N_HEADS)]
        results[li] = {
            "mean_entropy": float(np.mean(per_head)),
            "per_head": [round(v, 4) for v in per_head],
            "mean_max_attn": float(np.mean(per_head_max)),
            "per_head_max_attn": [round(v, 4) for v in per_head_max],
        }
    return results


def plot_results(baseline_results, quartic_results, output_dir="charts"):
    """Create publication-quality entropy comparison plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    Path(output_dir).mkdir(exist_ok=True)

    layers = list(range(DEPTH))
    bl_entropy = [baseline_results[l]["mean_entropy"] for l in layers]
    q4_entropy = [quartic_results[l]["mean_entropy"] for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel 1: Mean entropy per layer ---
    ax = axes[0]
    x = np.arange(DEPTH)
    width = 0.35
    bars1 = ax.bar(x - width/2, bl_entropy, width, label="Baseline",
                   color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, q4_entropy, width, label="Quartic (γ=4)",
                   color="#DD8452", alpha=0.85, edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean Attention Entropy (nats)", fontsize=11)
    ax.set_title("Attention Entropy per Layer", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {i}" for i in layers])
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(max(bl_entropy), max(q4_entropy)) * 1.2)

    # --- Panel 2: Per-head entropy heatmap style ---
    ax = axes[1]
    bl_heads = np.array([baseline_results[l]["per_head"] for l in layers])  # (n_layer, n_head)
    q4_heads = np.array([quartic_results[l]["per_head"] for l in layers])
    diff = q4_heads - bl_heads  # negative = quartic has lower entropy (more focused)

    im = ax.imshow(diff, cmap="RdBu_r", aspect="auto", vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    ax.set_xlabel("Head", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title("Entropy Difference (Quartic − Baseline)", fontsize=12, fontweight="bold")
    ax.set_xticks(range(N_HEADS))
    ax.set_xticklabels([f"H{i}" for i in range(N_HEADS)])
    ax.set_yticks(range(DEPTH))
    ax.set_yticklabels([f"L{i}" for i in range(DEPTH)])

    # Annotate cells
    for i in range(DEPTH):
        for j in range(N_HEADS):
            ax.text(j, i, f"{diff[i,j]:+.2f}", ha="center", va="center",
                    fontsize=9, color="black" if abs(diff[i,j]) < np.abs(diff).max()*0.5 else "white")

    plt.colorbar(im, ax=ax, label="Δ Entropy (nats)", shrink=0.8)

    plt.tight_layout()
    for fmt in ["svg", "png"]:
        path = f"{output_dir}/attention_entropy_per_layer.{fmt}"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close()


def main():
    val_data = load_data("val")

    configs = [
        ("Baseline", "checkpoints/model_baseline_42.pt"),
        ("Quartic", "checkpoints/model_window_power_4.0_42.pt"),
    ]

    all_results = {}
    for label, ckpt_path in configs:
        print(f"\n{'='*60}")
        print(f"  Computing attention entropy: {label}")
        print(f"{'='*60}")
        model, arch_name = load_checkpoint(ckpt_path)
        print(f"  arch={arch_name}, params={model.count_parameters():,}")
        results = compute_attention_entropy(model, val_data, n_batches=N_BATCHES)
        all_results[label] = results

        print(f"\n  {'Layer':<8} {'Mean Entropy':<14} {'Per-Head Entropy':<40} {'Mean Max Attn':<14}")
        print(f"  {'-'*76}")
        for li in range(DEPTH):
            r = results[li]
            heads_str = "  ".join(f"{v:.3f}" for v in r["per_head"])
            print(f"  L{li:<6} {r['mean_entropy']:<14.4f} [{heads_str}]  {r['mean_max_attn']:<14.4f}")
        del model

    # Print comparison table
    bl = all_results["Baseline"]
    q4 = all_results["Quartic"]
    print(f"\n{'='*60}")
    print(f"  COMPARISON: Baseline vs Quartic")
    print(f"{'='*60}")
    print(f"  {'Layer':<8} {'Baseline':<12} {'Quartic':<12} {'Δ Entropy':<12} {'% Change':<10}")
    print(f"  {'-'*54}")
    for li in range(DEPTH):
        bl_e = bl[li]["mean_entropy"]
        q4_e = q4[li]["mean_entropy"]
        delta = q4_e - bl_e
        pct = delta / bl_e * 100 if bl_e > 0 else 0
        marker = "◀ more focused" if delta < 0 else ""
        print(f"  L{li:<6} {bl_e:<12.4f} {q4_e:<12.4f} {delta:<+12.4f} {pct:<+10.1f}% {marker}")

    # Save results
    Path("gradient_results").mkdir(exist_ok=True)
    save_data = {}
    for label, results in all_results.items():
        save_data[label] = {str(k): v for k, v in results.items()}
    with open("gradient_results/attention_entropy.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to gradient_results/attention_entropy.json")

    # Plot
    plot_results(bl, q4)

    # Summary
    bl_mean = np.mean([bl[l]["mean_entropy"] for l in range(DEPTH)])
    q4_mean = np.mean([q4[l]["mean_entropy"] for l in range(DEPTH)])
    print(f"\n  Overall mean entropy: Baseline={bl_mean:.4f}, Quartic={q4_mean:.4f}")
    print(f"  Δ = {q4_mean - bl_mean:+.4f} ({(q4_mean - bl_mean)/bl_mean*100:+.1f}%)")

    # Check prediction: quartic should have LOWER entropy at early layers
    early_bl = np.mean([bl[l]["mean_entropy"] for l in range(DEPTH//2)])
    early_q4 = np.mean([q4[l]["mean_entropy"] for l in range(DEPTH//2)])
    print(f"\n  Early layers (L0-L1): Baseline={early_bl:.4f}, Quartic={early_q4:.4f}")
    if early_q4 < early_bl:
        print(f"  ✓ CONFIRMED: Quartic has lower entropy at early layers (more focused attention)")
    else:
        print(f"  ✗ SURPRISE: Quartic has higher entropy at early layers")


if __name__ == "__main__":
    main()
