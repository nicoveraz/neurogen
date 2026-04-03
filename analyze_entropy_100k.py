"""
NeuroGen: Attention Entropy at 100K steps + combined 20K vs 100K plot.

Reuses the entropy computation from analyze_attention_entropy.py,
loads existing 20K data from gradient_results/attention_entropy.json,
and creates a publication-quality combined figure.

Usage:
    python analyze_entropy_100k.py
"""

import json, math, sys
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
BATCH_SIZE = 8


def load_checkpoint(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    arch_name = ckpt.get("arch", "baseline")
    arch_cfg = ARCHS.get(arch_name, {})
    model = GPT(VOCAB_SIZE, MAX_SEQ_LEN, DEPTH, N_HEADS, N_KV_HEADS,
                CHANNELS, arch_cfg=arch_cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    steps = ckpt.get("max_steps", ckpt.get("total_steps", "?"))
    return model, arch_name, steps


def compute_attention_entropy(model, val_data, n_batches=N_BATCHES):
    n_layer = model.n_layer
    entropy_accum = [[[] for _ in range(N_HEADS)] for _ in range(n_layer)]
    max_attn_accum = [[[] for _ in range(N_HEADS)] for _ in range(n_layer)]
    arch_cfg = model.arch_cfg
    win_mode = arch_cfg.get("window")

    with torch.no_grad():
        for batch_idx in range(n_batches):
            x, _ = get_batch(val_data, BATCH_SIZE, MAX_SEQ_LEN, DEVICE)
            B, T = x.size()
            cos, sin = model.cos[:, :T], model.sin[:, :T]
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

                q_t, k_t, v_t = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                scale = 1.0 / math.sqrt(attn.head_dim)
                att = q_t @ k_t.transpose(-2, -1) * scale

                if win_mode:
                    wmask = compute_window_mask(T, li, n_layer, win_mode, DEVICE)
                    att = att.masked_fill(wmask[:T, :T] == 0, float("-inf"))
                else:
                    cmask = torch.tril(torch.ones(T, T, device=DEVICE, dtype=torch.bool))
                    att = att.masked_fill(~cmask, float("-inf"))

                attn_weights = F.softmax(att, dim=-1)

                log_w = torch.log(attn_weights.clamp(min=1e-10))
                entropy = -(attn_weights * log_w).sum(dim=-1)
                for hi in range(N_HEADS):
                    entropy_accum[li][hi].append(entropy[:, hi, 1:].mean().item())
                    max_attn_accum[li][hi].append(attn_weights[:, hi, 1:, :].max(dim=-1).values.mean().item())

                # Continue forward
                y = attn_weights @ v_t
                out = attn.c_proj(y.transpose(1, 2).contiguous().view(B, T, -1))
                if block.ca_ch is not None and block.ca_mode in ("attn", "both", "multiscale"):
                    ca_signal = block.ca_ch(h, layer_idx=block.layer_idx) if block.ca_mode == "multiscale" else block.ca_ch(h)
                    out = out * (1 + torch.tanh(ca_signal) * 0.1)
                elif block.ca_ch is not None and block.ca_mode == "additive":
                    out = out + block.ca_ch(h) * 0.1
                h = h + out
                mlp_out = block.mlp(rms_norm(h))
                if block.ca_ch is not None and block.ca_mode == "both":
                    mlp_out = mlp_out * (1 + torch.tanh(block.ca_ch(h)) * 0.1)
                h = h + mlp_out

            if (batch_idx + 1) % 25 == 0:
                print(f"  batch {batch_idx + 1}/{n_batches}")

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


def plot_combined(data_20k, data_100k, output_dir="charts"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(exist_ok=True)
    layers = list(range(DEPTH))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: 20K entropy ---
    ax = axes[0]
    bl_20k = [data_20k["Baseline"][str(l)]["mean_entropy"] for l in layers]
    q4_20k = [data_20k["Quartic"][str(l)]["mean_entropy"] for l in layers]
    x = np.arange(DEPTH)
    w = 0.35
    bars1 = ax.bar(x - w/2, bl_20k, w, label="Baseline", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + w/2, q4_20k, w, label="Quartic", color="#DD8452", alpha=0.85)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean Attention Entropy (nats)", fontsize=11)
    ax.set_title("20K Steps", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in layers])
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = max(max(bl_20k), max(q4_20k), 3.0) * 1.2
    ax.set_ylim(0, ymax)

    # --- Panel 2: 100K entropy ---
    ax = axes[1]
    bl_100k = [data_100k["Baseline"][str(l)]["mean_entropy"] for l in layers]
    q4_100k = [data_100k["Quartic"][str(l)]["mean_entropy"] for l in layers]
    bars1 = ax.bar(x - w/2, bl_100k, w, label="Baseline", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + w/2, q4_100k, w, label="Quartic", color="#DD8452", alpha=0.85)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_title("100K Steps", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in layers])
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, ymax)

    # --- Panel 3: Per-head heatmap diff at 100K ---
    ax = axes[2]
    bl_heads = np.array([data_100k["Baseline"][str(l)]["per_head"] for l in layers])
    q4_heads = np.array([data_100k["Quartic"][str(l)]["per_head"] for l in layers])
    diff = q4_heads - bl_heads
    vmax = max(np.abs(diff).max(), 0.5)
    im = ax.imshow(diff, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Head", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title("Entropy Diff at 100K (Q-B)", fontsize=12, fontweight="bold")
    ax.set_xticks(range(N_HEADS))
    ax.set_xticklabels([f"H{i}" for i in range(N_HEADS)])
    ax.set_yticks(range(DEPTH))
    ax.set_yticklabels([f"L{i}" for i in range(DEPTH)])
    for i in range(DEPTH):
        for j in range(N_HEADS):
            color = "white" if abs(diff[i, j]) > vmax * 0.5 else "black"
            ax.text(j, i, f"{diff[i,j]:+.2f}", ha="center", va="center", fontsize=9, color=color)
    plt.colorbar(im, ax=ax, label="Entropy (nats)", shrink=0.8)

    plt.tight_layout()
    for fmt in ["svg", "png"]:
        path = f"{output_dir}/attention_entropy_20k_vs_100k.{fmt}"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close()

    # --- Standalone clean bar chart for README (just 100K) ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - w/2, bl_100k, w, label="Baseline (100K)", color="#4C72B0", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + w/2, q4_100k, w, label="Quartic (100K)", color="#DD8452", alpha=0.85, edgecolor="white")
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10)
    # Add % change annotations
    for i in range(DEPTH):
        pct = (q4_100k[i] - bl_100k[i]) / bl_100k[i] * 100
        y_pos = min(bl_100k[i], q4_100k[i]) - 0.08
        ax.text(i, y_pos, f"{pct:+.0f}%", ha="center", va="top", fontsize=9,
                fontweight="bold", color="#C44E52" if pct > 0 else "#4C72B0")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean Attention Entropy (nats)", fontsize=12)
    ax.set_title("Attention Entropy per Layer (100K steps)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {i}" for i in layers], fontsize=11)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(max(bl_100k), max(q4_100k)) * 1.25)
    plt.tight_layout()
    for fmt in ["svg", "png"]:
        path = f"{output_dir}/attention_entropy_per_layer.{fmt}"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close()


def main():
    val_data = load_data("val")

    # Load existing 20K data
    data_20k = json.load(open("gradient_results/attention_entropy.json"))
    print("Loaded 20K entropy data from gradient_results/attention_entropy.json")

    # Compute 100K entropy
    configs = [
        ("Baseline", "checkpoints/model_baseline_42.pt"),
        ("Quartic", "checkpoints/model_window_power_4.0_42.pt"),
    ]
    data_100k = {}
    for label, ckpt_path in configs:
        print(f"\n{'='*60}")
        print(f"  Computing attention entropy (100K): {label}")
        print(f"{'='*60}")
        model, arch_name, steps = load_checkpoint(ckpt_path)
        print(f"  arch={arch_name}, steps={steps}, params={model.count_parameters():,}")
        results = compute_attention_entropy(model, val_data)
        data_100k[label] = {str(k): v for k, v in results.items()}

        print(f"\n  {'Layer':<8} {'Entropy':<12} {'Per-Head':<40}")
        print(f"  {'-'*60}")
        for li in range(DEPTH):
            r = results[li]
            heads = "  ".join(f"{v:.3f}" for v in r["per_head"])
            print(f"  L{li:<6} {r['mean_entropy']:<12.4f} [{heads}]")
        del model

    # Save 100K results
    with open("gradient_results/attention_entropy_100k.json", "w") as f:
        json.dump(data_100k, f, indent=2)
    print(f"\nSaved gradient_results/attention_entropy_100k.json")

    # Comparison tables
    print(f"\n{'='*70}")
    print(f"  ENTROPY COMPARISON: 20K vs 100K")
    print(f"{'='*70}")
    print(f"  {'':8} {'--- 20K Steps ---':^28} {'--- 100K Steps ---':^28}")
    print(f"  {'Layer':<8} {'Baseline':<10} {'Quartic':<10} {'Change':<10} {'Baseline':<10} {'Quartic':<10} {'Change':<10}")
    print(f"  {'-'*68}")
    for li in range(DEPTH):
        bl_20 = data_20k["Baseline"][str(li)]["mean_entropy"]
        q4_20 = data_20k["Quartic"][str(li)]["mean_entropy"]
        pct_20 = (q4_20 - bl_20) / bl_20 * 100
        bl_100 = data_100k["Baseline"][str(li)]["mean_entropy"]
        q4_100 = data_100k["Quartic"][str(li)]["mean_entropy"]
        pct_100 = (q4_100 - bl_100) / bl_100 * 100
        print(f"  L{li:<6} {bl_20:<10.3f} {q4_20:<10.3f} {pct_20:<+9.1f}% {bl_100:<10.3f} {q4_100:<10.3f} {pct_100:<+9.1f}%")

    # Plot
    plot_combined(data_20k, data_100k)


if __name__ == "__main__":
    main()
