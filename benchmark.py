"""
Benchmark script for NeuroGen autoresearch.

Runs controlled multi-seed comparisons of initialization methods
and produces statistical reports. Separate from the autoresearch loop —
the human runs this when they want rigorous evidence.

Usage:
    uv run benchmark.py --compare "default,xavier,block_diagonal,grid_ca" --seeds 5 --minutes 2
    uv run benchmark.py --baseline --minutes 2 --seeds 3   # establish xavier reference
"""

import argparse
import csv
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    load_data, get_batch, evaluate_val_bpb, get_device, get_peak_memory_mb,
    VOCAB_SIZE, MAX_SEQ_LEN,
)

# ---------------------------------------------------------------------------
# Import model and training components from train.py
# ---------------------------------------------------------------------------

from train import GPT, get_lr, DEPTH, CHANNELS, N_HEADS, N_KV_HEADS, LR, WEIGHT_DECAY

DEVICE = get_device()
BATCH_SIZE = 64
OUTPUTS_DIR = Path("outputs")
REFERENCE_FILE = OUTPUTS_DIR / "reference_bpb.json"

# ---------------------------------------------------------------------------
# Reference baselines (filled in by --baseline runs)
# ---------------------------------------------------------------------------

def load_references() -> dict:
    """Load reference val_bpb values from disk."""
    if REFERENCE_FILE.exists():
        return json.loads(REFERENCE_FILE.read_text())
    return {}


def save_references(refs: dict) -> None:
    """Save reference val_bpb values to disk."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    REFERENCE_FILE.write_text(json.dumps(refs, indent=2) + "\n")


def get_reference_key(minutes: float) -> str:
    """Config key for reference lookup."""
    return f"xavier_d{DEPTH}_{minutes:.0f}min"


def vs_baseline_pct(val_bpb: float, baseline_bpb: float) -> float:
    """Percentage improvement over baseline. Positive = better (lower bpb)."""
    if baseline_bpb < 1e-8:
        return 0.0
    return (baseline_bpb - val_bpb) / baseline_bpb * 100


# ---------------------------------------------------------------------------
# FLOPs estimation
# ---------------------------------------------------------------------------

def estimate_flops_per_step(n_params: int, block_size: int, batch_size: int) -> float:
    """Rough FLOPs per training step: ~6 * N * B * T (forward + backward)."""
    return 6.0 * n_params * batch_size * block_size


def estimate_ca_flops(method: str, n_params: int) -> float:
    """Estimate CA development FLOPs. Wall time is the real proxy on M1 Pro,
    but this gives order-of-magnitude accounting."""
    # CA methods do O(n_steps * n_params) work
    ca_steps = {
        "grid_ca": 64, "modular_ca": 48, "reaction_diffusion": 200,
        "hierarchical_ca": 72,
    }
    steps = ca_steps.get(method, 0)
    # ~10 FLOPs per CA step per param (neighborhood ops + update)
    return steps * n_params * 10.0


# ---------------------------------------------------------------------------
# Initialization methods registry
# ---------------------------------------------------------------------------

def _is_ca_target(name: str, p: torch.Tensor) -> bool:
    """Check if a parameter should receive CA initialization.
    Skips embeddings, lm_head, and small params where CA would fail."""
    if p.dim() != 2:
        return False
    if "wte" in name or "lm_head" in name:
        return False
    if min(p.shape) < 8:  # CA needs reasonable grid size
        return False
    return True


def apply_init(model: GPT, method: str) -> float:
    """Apply an initialization method. Returns wall time spent on init (seconds)."""
    t0 = time.time()

    if method == "default":
        return 0.0  # nanochat default already applied

    if method == "xavier":
        for name, p in model.named_parameters():
            if p.dim() >= 2 and "wte" not in name and "lm_head" not in name:
                nn.init.xavier_uniform_(p)
        return time.time() - t0

    if method == "orthogonal":
        from ca_rules import orthogonal_init
        for name, p in model.named_parameters():
            if p.dim() == 2 and "wte" not in name and "lm_head" not in name:
                w = orthogonal_init(p.shape)
                w = w * 0.02 / max(w.std().item(), 1e-8)
                p.data.copy_(w)
        return time.time() - t0

    if method == "block_diagonal":
        from ca_rules import block_diagonal_init
        for name, p in model.named_parameters():
            if p.dim() == 2 and "wte" not in name and "lm_head" not in name:
                p.data.copy_(block_diagonal_init(p.shape, n_blocks=4))
        return time.time() - t0

    if method == "grid_ca":
        from ca_rules import grid_ca_develop
        for name, p in model.named_parameters():
            if _is_ca_target(name, p):
                p.data.copy_(grid_ca_develop(p.shape, n_steps=64, seed="center"))
        return time.time() - t0

    if method == "modular_ca":
        from ca_rules import modular_init
        for name, p in model.named_parameters():
            if _is_ca_target(name, p):
                p.data.copy_(modular_init(p.shape, n_modules=4))
        return time.time() - t0

    if method == "reaction_diffusion":
        from ca_rules import reaction_diffusion_init
        for name, p in model.named_parameters():
            if _is_ca_target(name, p):
                p.data.copy_(reaction_diffusion_init(p.shape, n_steps=200))
        return time.time() - t0

    if method == "hierarchical_ca":
        from ca_rules import hierarchical_init_for_layer
        layer_idx = 0
        for name, p in model.named_parameters():
            if _is_ca_target(name, p):
                w = hierarchical_init_for_layer(p.shape, layer_idx % DEPTH, DEPTH)
                p.data.copy_(w)
                layer_idx += 1
        return time.time() - t0

    raise ValueError(f"Unknown init method: {method}")


# ---------------------------------------------------------------------------
# Quality metrics (lightweight, no external deps)
# ---------------------------------------------------------------------------

_QUALITY_PROMPTS = [
    "Once upon a time, there was a little",
    "The cat sat on the mat and looked at",
    '"Can you help me?" asked the',
]


def _generate_text(model: GPT, prompt: str, max_tokens: int = 100) -> str:
    """Generate text from byte-level model."""
    model.eval()
    bs = model.block_size
    ids = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for _ in range(max_tokens):
            logits, _ = model(ids[:, -bs:])
            logits = logits[:, -1, :] / 0.8
            v, _ = torch.topk(logits, min(50, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            try:
                nxt = torch.multinomial(probs, 1)
            except RuntimeError:
                nxt = torch.multinomial(probs.cpu(), 1).to(DEVICE)
            ids = torch.cat([ids, nxt], 1)
    return bytes(ids[0].tolist()).decode("utf-8", errors="replace")


def _repetition_rate(text: str, n: int = 3) -> float:
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def _unique_token_ratio(text: str) -> float:
    words = text.split()
    return len(set(words)) / len(words) if words else 0.0


def _self_perplexity(model: GPT, text: str) -> float:
    text_bytes = text.encode("utf-8")
    if len(text_bytes) < 4:
        return float("inf")
    ids = torch.tensor([list(text_bytes)], dtype=torch.long, device=DEVICE)
    ids = ids[:, :model.block_size + 1]
    if ids.size(1) < 2:
        return float("inf")
    model.eval()
    with torch.no_grad():
        _, loss = model(ids[:, :-1], ids[:, 1:])
    return loss.exp().item()


def compute_quality(model: GPT) -> dict:
    """Quick quality eval: 3 prompts, 1 sample each."""
    reps, uniqs, ppls = [], [], []
    for prompt in _QUALITY_PROMPTS:
        text = _generate_text(model, prompt)
        generated = text[len(prompt):]
        reps.append(_repetition_rate(generated))
        uniqs.append(_unique_token_ratio(generated))
        ppls.append(_self_perplexity(model, text))
    return {
        "repetition_3gram": statistics.mean(reps),
        "unique_token_ratio": statistics.mean(uniqs),
        "self_perplexity": statistics.mean(ppls),
    }


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def run_single(method: str, seed: int, minutes: float, eval_quality: bool = False) -> dict:
    """Run one training experiment. Returns metrics dict."""
    torch.manual_seed(seed)

    train_data = load_data("train")
    val_data = load_data("val")
    block_size = MAX_SEQ_LEN
    time_budget = minutes * 60

    model = GPT(VOCAB_SIZE, block_size, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS).to(DEVICE)
    n_params = model.count_parameters()
    init_wall_time = apply_init(model, method)

    # Init metrics
    init_bpb = evaluate_val_bpb(model, val_data, BATCH_SIZE, block_size, DEVICE)
    x0, y0 = get_batch(train_data, BATCH_SIZE, block_size, DEVICE)
    _, init_loss_val = model(x0, y0)
    init_loss = init_loss_val.item()

    # Init diagnostics (computed on CPU for MPS safety)
    init_weight_std = compute_weight_std(model)
    init_head_div = compute_head_diversity(model)
    init_block_ratio = compute_block_diag_ratio(model)
    init_layer_sim = compute_layer_similarity(model)

    # Train — baseline gets the CA init time back as extra training time
    train_budget = time_budget - init_wall_time
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()
    step = 0
    warmup = 100
    max_steps = 100_000
    min_lr = LR / 10
    t0 = time.time()

    while time.time() - t0 < train_budget:
        x, y = get_batch(train_data, BATCH_SIZE, block_size, DEVICE)
        lr = get_lr(step, warmup, max_steps, LR, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

    train_wall_time = time.time() - t0
    val_bpb = evaluate_val_bpb(model, val_data, BATCH_SIZE, block_size, DEVICE)
    peak_mem = get_peak_memory_mb()

    # FLOPs accounting
    flops_per_step = estimate_flops_per_step(n_params, block_size, BATCH_SIZE)
    train_flops = flops_per_step * step
    ca_flops = estimate_ca_flops(method, n_params)
    total_flops = train_flops + ca_flops

    result = {
        "method": method,
        "seed": seed,
        "val_bpb": val_bpb,
        "init_loss": init_loss,
        "init_bpb": init_bpb,
        "final_train_loss": loss.item(),
        "total_steps": step,
        "wall_time_s": init_wall_time + train_wall_time,
        "init_wall_time_s": init_wall_time,
        "train_wall_time_s": train_wall_time,
        "total_flops": total_flops,
        "ca_flops": ca_flops,
        "peak_memory_mb": peak_mem,
        "init_weight_std": init_weight_std,
        "init_head_diversity": init_head_div,
        "init_block_diag_ratio": init_block_ratio,
        "init_layer_similarity": init_layer_sim,
        "vs_baseline_pct": 0.0,  # filled in after all runs
        "repetition_3gram": 0.0,
        "unique_token_ratio": 0.0,
        "self_perplexity": 0.0,
    }

    if eval_quality:
        quality = compute_quality(model)
        result.update(quality)

    return result


# ---------------------------------------------------------------------------
# Diagnostic metric helpers
# ---------------------------------------------------------------------------

def compute_weight_std(model: GPT) -> float:
    """Mean std of all weight parameters (excludes embeddings)."""
    stds = []
    for name, p in model.named_parameters():
        if p.dim() >= 2 and "wte" not in name and "lm_head" not in name:
            stds.append(p.data.cpu().float().std().item())
    return statistics.mean(stds) if stds else 0.0


def compute_head_diversity(model: GPT) -> float:
    """Mean pairwise cosine distance between Q-projection weight vectors."""
    q_weights = []
    for block in model.blocks:
        w = block.attn.c_q.weight.data.cpu().float()
        head_dim = w.shape[0] // model.blocks[0].attn.n_head
        for h in range(model.blocks[0].attn.n_head):
            q_weights.append(w[h * head_dim : (h + 1) * head_dim].flatten())
    if len(q_weights) < 2:
        return 0.0
    dists = []
    for i in range(len(q_weights)):
        for j in range(i + 1, len(q_weights)):
            cos_sim = torch.cosine_similarity(
                q_weights[i].unsqueeze(0), q_weights[j].unsqueeze(0)
            ).item()
            dists.append(1.0 - cos_sim)
    return statistics.mean(dists)


def compute_block_diag_ratio(model: GPT) -> float:
    """Ratio of energy in block-diagonal vs off-diagonal for attention weights."""
    ratios = []
    for block in model.blocks:
        for name in ["c_q", "c_k", "c_v"]:
            w = getattr(block.attn, name).weight.data.cpu().float()
            h, ww = w.shape
            n_blocks = min(4, min(h, ww))
            bh, bw = h // n_blocks, ww // n_blocks
            diag_energy = 0.0
            for b in range(n_blocks):
                r0, r1 = b * bh, min((b + 1) * bh, h)
                c0, c1 = b * bw, min((b + 1) * bw, ww)
                diag_energy += w[r0:r1, c0:c1].pow(2).sum().item()
            total_energy = w.pow(2).sum().item()
            if total_energy > 1e-12:
                ratios.append(diag_energy / total_energy)
    return statistics.mean(ratios) if ratios else 0.0


def compute_layer_similarity(model: GPT) -> float:
    """Mean cosine similarity between weight matrices of adjacent layers."""
    layer_vecs = []
    for block in model.blocks:
        params = []
        for name, p in block.named_parameters():
            if "ve_gate" not in name:  # skip variable-size params
                params.append(p.data.cpu().float().flatten())
        layer_vecs.append(torch.cat(params))
    if len(layer_vecs) < 2:
        return 0.0
    min_len = min(v.shape[0] for v in layer_vecs)
    sims = []
    for i in range(len(layer_vecs) - 1):
        sim = torch.cosine_similarity(
            layer_vecs[i][:min_len].unsqueeze(0),
            layer_vecs[i + 1][:min_len].unsqueeze(0),
        ).item()
        sims.append(sim)
    return statistics.mean(sims)


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def paired_ttest(a: list[float], b: list[float]) -> float:
    """Paired t-test p-value (two-tailed). Returns 1.0 if n < 2."""
    n = len(a)
    if n < 2 or len(b) != n:
        return 1.0
    diffs = [ai - bi for ai, bi in zip(a, b)]
    mean_d = statistics.mean(diffs)
    std_d = statistics.stdev(diffs)
    if std_d < 1e-12:
        return 0.0 if abs(mean_d) > 1e-12 else 1.0
    t_stat = mean_d / (std_d / math.sqrt(n))
    p = 2 * (1 - _normal_cdf(abs(t_stat)))
    return p


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---------------------------------------------------------------------------
# FLOPs-matched comparison
# ---------------------------------------------------------------------------

def flops_matched_comparison(
    all_results: list[dict], baseline_method: str = "xavier"
) -> list[str]:
    """Compare methods at matched total FLOPs. Returns report lines."""
    baseline_runs = [r for r in all_results if r["method"] == baseline_method]
    if not baseline_runs:
        return ["(no baseline runs for FLOPs comparison)"]

    # Build FLOPs → val_bpb mapping for baseline
    baseline_by_flops = sorted(baseline_runs, key=lambda r: r["total_flops"])

    lines = ["", "## FLOPs-Matched Comparison", ""]
    lines.append(f"| Method | total_flops | val_bpb | matched_baseline | improvement |")
    lines.append(f"|--------|-------------|---------|------------------|-------------|")

    other_methods = set(r["method"] for r in all_results) - {baseline_method}
    for method in sorted(other_methods):
        runs = [r for r in all_results if r["method"] == method]
        for r in runs:
            # Find baseline with closest FLOPs
            closest = min(baseline_by_flops, key=lambda b: abs(b["total_flops"] - r["total_flops"]))
            pct = vs_baseline_pct(r["val_bpb"], closest["val_bpb"])
            lines.append(
                f"| {method} s{r['seed']} | {r['total_flops']:.2e} | {r['val_bpb']:.4f} "
                f"| {closest['val_bpb']:.4f} | {pct:+.1f}% |"
            )

    return lines


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    all_results: list[dict], methods: list[str], n_seeds: int, minutes: float,
    baseline_bpb: float | None = None,
) -> str:
    """Generate markdown report from benchmark results."""
    lines = [
        f"# NeuroGen Benchmark Report",
        f"",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Seeds:** {n_seeds} | **Time budget:** {minutes} min/run | **Device:** {DEVICE}",
        f"**Model:** depth={DEPTH}, channels={CHANNELS}, heads={N_HEADS}",
    ]
    if baseline_bpb is not None:
        lines.append(f"**Xavier baseline:** {baseline_bpb:.4f} val_bpb")
    lines += ["", "## Results", ""]

    has_baseline = baseline_bpb is not None
    header = "| Method | val_bpb | vs_baseline | init_loss | steps | init_time_s | total_flops |"
    sep = "|--------|---------|-------------|-----------|-------|-------------|-------------|"
    lines += [header, sep]

    method_bpbs: dict[str, list[float]] = {}
    for m in methods:
        runs = [r for r in all_results if r["method"] == m]
        bpbs = [r["val_bpb"] for r in runs]
        method_bpbs[m] = bpbs
        mean_bpb = statistics.mean(bpbs)
        std_bpb = statistics.stdev(bpbs) if len(bpbs) > 1 else 0.0
        mean_init = statistics.mean([r["init_loss"] for r in runs])
        mean_steps = statistics.mean([r["total_steps"] for r in runs])
        mean_init_time = statistics.mean([r["init_wall_time_s"] for r in runs])
        mean_flops = statistics.mean([r["total_flops"] for r in runs])

        if has_baseline:
            pct = vs_baseline_pct(mean_bpb, baseline_bpb)
            vs_str = f"{pct:+.1f}%"
        else:
            vs_str = "—"

        lines.append(
            f"| {m} | {mean_bpb:.4f} +/- {std_bpb:.4f} | {vs_str} | {mean_init:.4f} "
            f"| {mean_steps:.0f} | {mean_init_time:.1f} | {mean_flops:.2e} |"
        )

    # Statistical comparison vs best baseline
    best_method = min(method_bpbs, key=lambda m: statistics.mean(method_bpbs[m]))
    best_bpbs = method_bpbs[best_method]
    lines += ["", "## Statistical Comparison", "", f"**Best method:** {best_method}", ""]
    lines.append("| Method | mean diff | p-value | significant? |")
    lines.append("|--------|-----------|---------|--------------|")
    for m in methods:
        if m == best_method:
            continue
        p = paired_ttest(method_bpbs[m], best_bpbs)
        diff = statistics.mean(method_bpbs[m]) - statistics.mean(best_bpbs)
        sig = "YES" if p < 0.05 else "no"
        lines.append(f"| {m} | {diff:+.4f} | {p:.4f} | {sig} |")

    # Init diagnostics
    lines += ["", "## Init Diagnostics", ""]
    lines.append("| Method | weight_std | head_diversity | block_diag_ratio | layer_similarity |")
    lines.append("|--------|------------|----------------|------------------|------------------|")
    for m in methods:
        runs = [r for r in all_results if r["method"] == m]
        ws = statistics.mean([r["init_weight_std"] for r in runs])
        hd = statistics.mean([r["init_head_diversity"] for r in runs])
        br = statistics.mean([r["init_block_diag_ratio"] for r in runs])
        ls = statistics.mean([r["init_layer_similarity"] for r in runs])
        lines.append(f"| {m} | {ws:.4f} | {hd:.4f} | {br:.4f} | {ls:.4f} |")

    # FLOPs-matched comparison (if baseline method present)
    baseline_method = "xavier" if "xavier" in methods else "default"
    if len(methods) > 1:
        lines += flops_matched_comparison(all_results, baseline_method)

    # Quality comparison (if quality data present)
    has_quality = any(r.get("repetition_3gram", 0) > 0 or r.get("unique_token_ratio", 0) > 0
                      for r in all_results)
    if has_quality:
        lines += ["", "## Quality Comparison", ""]
        lines.append("| method | val_bpb | repetition | unique_ratio | self_perplexity |")
        lines.append("|--------|---------|------------|--------------|-----------------|")
        for m in methods:
            runs = [r for r in all_results if r["method"] == m]
            vb = statistics.mean([r["val_bpb"] for r in runs])
            rep = statistics.mean([r["repetition_3gram"] for r in runs])
            uniq = statistics.mean([r["unique_token_ratio"] for r in runs])
            ppl = statistics.mean([r["self_perplexity"] for r in runs])
            rep_std = statistics.stdev([r["repetition_3gram"] for r in runs]) if len(runs) > 1 else 0.0
            uniq_std = statistics.stdev([r["unique_token_ratio"] for r in runs]) if len(runs) > 1 else 0.0
            ppl_std = statistics.stdev([r["self_perplexity"] for r in runs]) if len(runs) > 1 else 0.0
            lines.append(
                f"| {m} | {vb:.4f} | {rep:.3f}+/-{rep_std:.3f} "
                f"| {uniq:.3f}+/-{uniq_std:.3f} | {ppl:.1f}+/-{ppl_std:.1f} |"
            )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NeuroGen multi-seed benchmark")
    parser.add_argument(
        "--compare", type=str, default=None,
        help="Comma-separated init methods (e.g. 'default,xavier,grid_ca')",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds per method")
    parser.add_argument("--minutes", type=float, default=2.0, help="Minutes per run")
    parser.add_argument(
        "--baseline", action="store_true",
        help="Run xavier baseline and save as reference for this config",
    )
    parser.add_argument(
        "--quality", action="store_true",
        help="Also evaluate output quality (repetition, diversity, self-perplexity)",
    )
    args = parser.parse_args()

    if not args.compare and not args.baseline:
        parser.error("Provide --compare or --baseline")

    # Load existing references
    refs = load_references()
    ref_key = get_reference_key(args.minutes)

    # --baseline mode: run xavier and save reference
    if args.baseline:
        methods = ["xavier"]
        n_seeds = args.seeds
        print(f"Establishing xavier baseline: {n_seeds} seeds x {args.minutes} min")
        print()
        OUTPUTS_DIR.mkdir(exist_ok=True)
        results = []
        for si in range(n_seeds):
            seed = 42 + si
            print(f"[{si+1}/{n_seeds}] xavier seed={seed} ...", end=" ", flush=True)
            result = run_single("xavier", seed, args.minutes)
            results.append(result)
            print(f"val_bpb={result['val_bpb']:.4f} ({result['total_steps']} steps)")
        mean_bpb = statistics.mean([r["val_bpb"] for r in results])
        std_bpb = statistics.stdev([r["val_bpb"] for r in results]) if len(results) > 1 else 0.0
        refs[ref_key] = {"mean": mean_bpb, "std": std_bpb, "seeds": n_seeds}
        save_references(refs)
        print(f"\nBaseline saved: {ref_key} = {mean_bpb:.4f} +/- {std_bpb:.4f}")
        if not args.compare:
            return

    # --compare mode
    methods = [m.strip() for m in args.compare.split(",")]
    n_seeds = args.seeds
    total_runs = len(methods) * n_seeds

    # Look up baseline reference
    baseline_bpb = None
    if ref_key in refs:
        baseline_bpb = refs[ref_key]["mean"]
        print(f"Reference baseline ({ref_key}): {baseline_bpb:.4f} val_bpb")
    else:
        print(f"No reference baseline for {ref_key}. Run --baseline first for vs_baseline_pct.")
    print(f"Benchmark: {len(methods)} methods x {n_seeds} seeds = {total_runs} runs")
    print(f"Estimated time: {total_runs * args.minutes:.0f} minutes")
    print(f"Methods: {methods}")
    print()

    OUTPUTS_DIR.mkdir(exist_ok=True)
    all_results = []

    for mi, method in enumerate(methods):
        for si in range(n_seeds):
            run_id = mi * n_seeds + si + 1
            seed = 42 + si
            print(f"[{run_id}/{total_runs}] {method} seed={seed} ...", end=" ", flush=True)
            result = run_single(method, seed, args.minutes, eval_quality=args.quality)

            # Fill in vs_baseline_pct
            if baseline_bpb is not None:
                result["vs_baseline_pct"] = vs_baseline_pct(result["val_bpb"], baseline_bpb)
                vs_str = f" (vs baseline: {result['vs_baseline_pct']:+.1f}%)"
            else:
                vs_str = ""

            all_results.append(result)
            print(f"val_bpb={result['val_bpb']:.4f}{vs_str} ({result['total_steps']} steps)")

    # Save raw CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUTS_DIR / f"benchmark_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nRaw data saved to {csv_path}")

    # Generate and save report
    report = generate_report(all_results, methods, n_seeds, args.minutes, baseline_bpb)
    md_path = OUTPUTS_DIR / f"benchmark_{ts}.md"
    md_path.write_text(report)
    print(f"Report saved to {md_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
