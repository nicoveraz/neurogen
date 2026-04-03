"""
NeuroGen Validation: Step-budget convergence runs with full diagnostics.

Uses STEP budget (not time budget) to eliminate throughput confounds.
Logs val_bpb at every eval interval for learning curve analysis.

Usage:
    uv run validate.py --arch baseline --steps 20000 --seed 42
    uv run validate.py --throughput  # Phase V1: step count audit
    uv run validate.py --tier1       # Phase V2: all Tier 1 runs
"""

import argparse, time, math, json, sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    load_data, get_batch, evaluate_val_bpb, get_device, get_peak_memory_mb,
    VOCAB_SIZE, MAX_SEQ_LEN,
)
from train_r4 import (
    GPT, ARCHS, DEPTH, CHANNELS, N_HEADS, N_KV_HEADS, BATCH_SIZE, LR,
    WEIGHT_DECAY, WARMUP, rms_norm, get_lr, apply_universal_init,
    compute_window_mask, _is_embryo_target, embryo_ca_step,
)
from ca_rules import block_diagonal_init

DEVICE = get_device()
EVAL_INTERVAL = 500

# ---------------------------------------------------------------------------
# Attention span measurement
# ---------------------------------------------------------------------------
def measure_attention_spans(model, val_data, device):
    """Measure effective attention span per layer.
    Returns list of (effective_span, window_size, utilization) per layer."""
    model.eval()
    block_size = MAX_SEQ_LEN
    x, _ = get_batch(val_data, 8, block_size, device)  # small batch
    B, T = x.size()
    spans = []

    # Get window sizes
    arch_cfg = model.arch_cfg
    win_mode = arch_cfg.get("window")

    with torch.no_grad():
        cos, sin = model.cos[:, :T], model.sin[:, :T]
        h = rms_norm(model.wte(x))
        if T > 1:
            gate = model.smear_lambda * torch.sigmoid(
                model.smear_gate(h[:, 1:, :model._smear_ch]))
            h = torch.cat([h[:, :1], h[:, 1:] + gate * h[:, :-1]], dim=1)
        x0 = h

        for i, block in enumerate(model.blocks):
            h = model.resid_lambdas[i] * h + model.x0_lambdas[i] * x0
            ve = model.value_embeds[str(i)](x) if str(i) in model.value_embeds else None
            normed = rms_norm(h)

            # Compute attention weights manually
            attn = block.attn
            B2, T2, C = normed.size()
            q = attn.c_q(normed).view(B2, T2, attn.n_head, attn.head_dim)
            k = attn.c_k(normed).view(B2, T2, attn.n_kv_head, attn.head_dim)
            from train_r4 import apply_rotary_emb as rope
            q = rope(q, cos, sin)
            k = rope(k, cos, sin)
            q, k = rms_norm(q) * 1.2, rms_norm(k) * 1.2
            if attn.n_kv_head < attn.n_head:
                r = attn.n_head // attn.n_kv_head
                k = k.repeat_interleave(r, dim=2)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            scale = 1.0 / math.sqrt(attn.head_dim)
            att = q @ k.transpose(-2, -1) * scale

            # Apply causal mask
            cmask = torch.tril(torch.ones(T2, T2, device=device, dtype=torch.bool))
            att = att.masked_fill(~cmask, float("-inf"))

            # Apply window mask if present
            if win_mode:
                wmask = compute_window_mask(T2, i, model.n_layer, win_mode, device)
                if wmask is not None:
                    att = att.masked_fill(wmask == 0, float("-inf"))

            attn_weights = F.softmax(att, dim=-1)  # (B, heads, T, T)

            # Compute effective span: mean distance of attention
            positions = torch.arange(T2, device=device).float()
            dist_matrix = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
            expected_dist = (attn_weights * dist_matrix.unsqueeze(0).unsqueeze(0)).sum(-1)
            eff_span = expected_dist.mean().item()

            # Get window size for this layer
            if win_mode:
                wmask_test = compute_window_mask(T2, i, model.n_layer, win_mode, device)
                window_size = wmask_test.sum(-1).max().item() if wmask_test is not None else T2
            else:
                window_size = T2
            util = eff_span / max(window_size, 1)

            spans.append((round(eff_span, 1), int(window_size), round(util, 3)))

            # Forward through block for next layer
            h, _ = block(h, ve, cos, sin)

    model.train()
    return spans


# ---------------------------------------------------------------------------
# Step-budget training
# ---------------------------------------------------------------------------
def train_steps(arch: str, max_steps: int = 20000, seed: int = 42,
                eval_interval: int = 500, quiet: bool = False):
    """Train for exactly max_steps steps (not time-based)."""
    arch_cfg = ARCHS.get(arch, {})
    torch.manual_seed(seed)

    train_data = load_data("train")
    val_data = load_data("val")
    block_size = MAX_SEQ_LEN

    model = GPT(VOCAB_SIZE, block_size, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS,
                arch_cfg=arch_cfg).to(DEVICE)
    params = model.count_parameters()

    # Standard init
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2 and not any(s in name for s in ("wte", "lm_head", "ve_gate")):
                if min(p.shape) >= 8:
                    nn.init.xavier_uniform_(p)
                    ca = block_diagonal_init(p.shape, n_blocks=min(4, min(p.shape)),
                                             target_std=p.std().item() * 0.05)
                    p.data.add_(ca.to(p.device))

    universal_mode = arch_cfg.get("universal")
    if universal_mode:
        apply_universal_init(model, universal_mode)

    if not quiet:
        print(f"arch: {arch}")
        print(f"seed: {seed}")
        print(f"max_steps: {max_steps}")
        print(f"params: {params:,}")
        print(f"device: {DEVICE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()

    lr = LR
    min_lr = lr / 10
    t0 = time.time()
    prev_vbpb = 8.0
    ema_vbpb = 8.0
    results = []

    for step in range(max_steps + 1):
        # Eval checkpoint
        if step % eval_interval == 0:
            vbpb = evaluate_val_bpb(model, val_data, BATCH_SIZE, block_size, DEVICE)
            elapsed = time.time() - t0
            delta = vbpb - prev_vbpb
            ema_vbpb = 0.1 * vbpb + 0.9 * ema_vbpb
            grad_norm = 0.0

            # Attention spans (every 5000 steps to limit overhead)
            spans = []
            if step % 5000 == 0:
                spans = measure_attention_spans(model, val_data, DEVICE)

            row = {
                "arch": arch, "seed": seed, "step": step,
                "val_bpb": round(vbpb, 6), "val_bpb_delta": round(delta, 6),
                "val_bpb_ema": round(ema_vbpb, 6),
                "elapsed_s": round(elapsed, 1),
            }
            if spans:
                for li, (sp, ws, ut) in enumerate(spans):
                    row[f"attn_span_l{li}"] = sp
                    row[f"attn_window_l{li}"] = ws
                    row[f"attn_util_l{li}"] = ut

            results.append(row)
            prev_vbpb = vbpb

            if not quiet:
                span_str = ""
                if spans:
                    span_str = " spans=[" + ",".join(f"{s[0]:.0f}/{s[1]}" for s in spans) + "]"
                print(f"step:{step:6d} vbpb:{vbpb:.4f} d:{delta:+.4f} "
                      f"ema:{ema_vbpb:.4f} {elapsed:.0f}s{span_str}")

        if step >= max_steps:
            break

        # Training step
        x, y = get_batch(train_data, BATCH_SIZE, block_size, DEVICE)
        cur_lr = get_lr(step, WARMUP, max_steps, lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        _, loss = model(x, y, step=step, total_steps=max_steps)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()

    total_time = time.time() - t0
    final_vbpb = results[-1]["val_bpb"]

    summary = {
        "arch": arch, "seed": seed, "max_steps": max_steps,
        "final_vbpb": final_vbpb, "total_time_s": round(total_time, 1),
        "steps_per_sec": round(max_steps / total_time, 2),
        "params": params,
    }
    print(f"VALIDATION_RESULT: {json.dumps(summary)}")

    # Save detailed curve
    out_dir = Path("validation_results")
    out_dir.mkdir(exist_ok=True)
    curve_path = out_dir / f"{arch}_s{seed}.json"
    with open(curve_path, "w") as f:
        json.dump({"summary": summary, "curve": results}, f, indent=2)
    print(f"Saved: {curve_path}")

    # Save model checkpoint
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"model_{arch}_{seed}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "arch": arch,
        "seed": seed,
        "max_steps": max_steps,
        "val_bpb": final_vbpb,
        "total_time_s": round(total_time, 1),
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")

    return summary, results


# ---------------------------------------------------------------------------
# Phase V1: Throughput audit
# ---------------------------------------------------------------------------
def throughput_audit():
    """Run 500 steps of each config, measure steps/sec."""
    configs = ["baseline", "window_linear", "window_quadratic", "window_power_3.0",
               "window_power_4.0", "window_power_5.0", "window_logarithmic",
               "window_quad_induction"]

    print("=== Phase V1: Throughput Audit (500 steps each) ===")
    print(f"{'arch':<28} {'steps/sec':<12} {'elapsed':<10} {'note'}")
    print("-" * 65)

    baseline_sps = None
    for arch in configs:
        arch_cfg = ARCHS.get(arch, {})
        torch.manual_seed(42)
        train_data = load_data("train")
        block_size = MAX_SEQ_LEN

        model = GPT(VOCAB_SIZE, block_size, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS,
                     arch_cfg=arch_cfg).to(DEVICE)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.dim() >= 2 and not any(s in name for s in ("wte", "lm_head", "ve_gate")):
                    if min(p.shape) >= 8:
                        nn.init.xavier_uniform_(p)
        universal_mode = arch_cfg.get("universal")
        if universal_mode:
            apply_universal_init(model, universal_mode)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        model.train()

        # Warmup
        for _ in range(10):
            x, y = get_batch(train_data, BATCH_SIZE, block_size, DEVICE)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Timed run
        t0 = time.time()
        for s in range(500):
            x, y = get_batch(train_data, BATCH_SIZE, block_size, DEVICE)
            _, loss = model(x, y, step=s, total_steps=500)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        elapsed = time.time() - t0
        sps = 500 / elapsed

        if baseline_sps is None:
            baseline_sps = sps
        diff = (sps - baseline_sps) / baseline_sps * 100
        note = "" if abs(diff) < 3 else f" ({diff:+.1f}% vs baseline)"
        print(f"{arch:<28} {sps:.2f}       {elapsed:.1f}s{note}")

        del model, optimizer


# ---------------------------------------------------------------------------
# Phase V2: Tier 1 convergence runs
# ---------------------------------------------------------------------------
def run_tier1(max_steps=20000):
    """Run all Tier 1 experiments: 4 configs x 5 seeds."""
    configs = ["baseline", "window_power_4.0", "window_quadratic", "window_quad_induction"]
    seeds = [42, 137, 256, 789, 1337]

    print(f"=== Phase V2 Tier 1: Convergence Runs ({max_steps} steps) ===")
    print(f"Configs: {configs}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {len(configs) * len(seeds)}")

    for arch in configs:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  {arch} seed={seed}")
            print(f"{'='*60}")
            train_steps(arch=arch, max_steps=max_steps, seed=seed, quiet=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NeuroGen Validation")
    parser.add_argument("--arch", type=str, default="baseline")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--throughput", action="store_true", help="Phase V1: throughput audit")
    parser.add_argument("--tier1", action="store_true", help="Phase V2 Tier 1: all convergence runs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.throughput:
        throughput_audit()
    elif args.tier1:
        run_tier1(max_steps=args.steps)
    else:
        train_steps(arch=args.arch, max_steps=args.steps, seed=args.seed,
                    eval_interval=args.eval_interval, quiet=args.quiet)


if __name__ == "__main__":
    main()
