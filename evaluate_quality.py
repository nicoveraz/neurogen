"""
NeuroGen Output Quality Evaluation.

Standalone script — evaluates trained models on text generation quality.
Does NOT modify train_r4.py or interrupt running experiments.

Usage:
    uv run evaluate_quality.py --checkpoint checkpoints/model_baseline_42.pt
    uv run evaluate_quality.py --compare ckpt1.pt ckpt2.pt
    uv run evaluate_quality.py --glob "checkpoints/model_*.pt" --report quality_report.md
    uv run evaluate_quality.py --live --arch baseline window_quad_induction --seed 42 --minutes 40
"""

import argparse, glob, json, math, re, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

from prepare import (
    load_data, get_batch, evaluate_val_bpb, get_device,
    VOCAB_SIZE, MAX_SEQ_LEN,
)

DEVICE = get_device()

EVAL_PROMPTS = [
    "Once upon a time, there was a little",
    "The cat sat on the mat and looked at",
    '"Can you help me?" asked the',
    "It was a beautiful sunny day. The children",
    "The most important thing about being kind is",
    "There was a little bear who lived in the forest. Every morning, he would",
    "Sarah was sad because she lost her",
    "The dog ran fast because",
    "One day, a bird flew into the",
    'Mom said, "It is time to',
]

# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def repetition_rate(text: str, n: int = 3) -> float:
    """Fraction of n-grams that are repeated. Healthy: <0.3. Degenerate: >0.7."""
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def unique_token_ratio(text: str) -> float:
    """Unique words / total words. Healthy: 0.4-0.7. Degenerate: <0.2."""
    words = text.split()
    return len(set(words)) / len(words) if words else 0.0


def sentence_completion_rate(text: str) -> float:
    """Fraction of sentences ending with punctuation. Healthy: >0.5."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1:
        return 0.0
    return (len(sentences) - 1) / len(sentences)


def mean_word_length(text: str) -> float:
    """Average chars per word. Healthy English: 3.5-5.5."""
    words = text.split()
    return sum(len(w) for w in words) / len(words) if words else 0.0


def local_coherence(text: str, window: int = 15) -> float:
    """Vocabulary overlap in sliding windows. Higher = more topically consistent."""
    words = text.lower().split()
    if len(words) < window:
        return 0.0
    ratios = []
    for i in range(len(words) - window):
        chunk = words[i:i + window]
        ratios.append(len(set(chunk)) / len(chunk))
    return 1.0 - (sum(ratios) / len(ratios))


def self_perplexity(model, text: str, device: str) -> float:
    """Feed model's own output back and measure loss. Low = coherent."""
    tokens = list(text.encode("utf-8"))
    if len(tokens) < 3:
        return float("inf")
    tokens = tokens[:MAX_SEQ_LEN]
    x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        _, loss = model(x, y)
    return loss.exp().item()


# ---------------------------------------------------------------------------
# Training efficiency metrics
# ---------------------------------------------------------------------------

def compute_efficiency(meta: dict, val_bpb: float) -> dict:
    """Compute training efficiency metrics from checkpoint metadata.
    Measures how well each approach uses compute budget."""
    total_steps = meta.get("total_steps", 0)
    wall_time = meta.get("wall_time_s", meta.get("time_budget_s", 1))
    params = meta.get("params", meta.get("total_params", 0))
    ca_overhead = meta.get("ca_overhead_pct", 0)

    # Steps per second (throughput)
    steps_per_sec = total_steps / max(wall_time, 1)
    # Effective training (discount for CA overhead)
    effective_steps = total_steps * (1 - ca_overhead / 100)
    # BPB per training step (convergence speed)
    bpb_per_step = (8.0 - val_bpb) / max(total_steps, 1)  # improvement from random
    # BPB per wall-second (real-world efficiency)
    bpb_per_sec = (8.0 - val_bpb) / max(wall_time, 1)
    # BPB per parameter (parameter efficiency)
    bpb_per_param = (8.0 - val_bpb) / max(params, 1) * 1e6

    return {
        "steps_per_sec": round(steps_per_sec, 2),
        "effective_steps": int(effective_steps),
        "bpb_improvement": round(8.0 - val_bpb, 4),
        "bpb_per_kstep": round(bpb_per_step * 1000, 4),
        "bpb_per_min": round(bpb_per_sec * 60, 4),
        "ca_overhead_pct": round(ca_overhead, 1),
    }


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate_text(model, prompt: str, max_tokens: int = 100,
                  temperature: float = 0.8, top_k: int = 50,
                  device: str = "cpu") -> str:
    """Generate text from byte-level prompt."""
    model.eval()
    ids = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=device)
    block_size = min(MAX_SEQ_LEN, 256)
    with torch.no_grad():
        for _ in range(max_tokens):
            logits, _ = model(ids[:, -block_size:])
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            try:
                nxt = torch.multinomial(probs, 1)
            except RuntimeError:
                nxt = torch.multinomial(probs.cpu(), 1).to(device)
            ids = torch.cat([ids, nxt], dim=1)
    return bytes(ids[0].tolist()).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, device: str, prompts: list[str] | None = None,
                   n_samples: int = 3, max_tokens: int = 100,
                   temperature: float = 0.8) -> dict:
    """Run full quality evaluation on a model."""
    if prompts is None:
        prompts = EVAL_PROMPTS

    all_generated = []
    all_samples = []
    for prompt in prompts:
        for _ in range(n_samples):
            full = generate_text(model, prompt, max_tokens=max_tokens,
                                 temperature=temperature, device=device)
            gen = full[len(prompt):]
            all_generated.append(gen)
            all_samples.append({"prompt": prompt, "generated": gen, "full": full})

    combined = " ".join(all_generated)
    metrics = {
        "n_samples": len(all_generated),
        "repetition_3gram": round(repetition_rate(combined, 3), 4),
        "unique_token_ratio": round(unique_token_ratio(combined), 4),
        "sentence_completion": round(sentence_completion_rate(combined), 4),
        "mean_word_length": round(mean_word_length(combined), 2),
        "local_coherence": round(local_coherence(combined), 4),
    }

    ppls = []
    for s in all_samples[:10]:
        ppl = self_perplexity(model, s["full"], device)
        if ppl < 1e6:
            ppls.append(ppl)
    metrics["self_perplexity"] = round(sum(ppls) / max(len(ppls), 1), 1) if ppls else float("inf")

    sample_ppls = [(self_perplexity(model, s["full"], device), s) for s in all_samples]
    sample_ppls.sort(key=lambda x: x[0])

    return {
        "metrics": metrics,
        "samples": all_samples,
        "best_sample": sample_ppls[0] if sample_ppls else None,
        "worst_sample": sample_ppls[-1] if sample_ppls else None,
    }


def print_evaluation(result: dict, label: str = "Model"):
    m = result["metrics"]
    meta = result.get("meta", {})
    print(f"\n=== Output Quality: {label} ===")
    print(f"samples: {m['n_samples']}")
    if meta.get("val_bpb"):
        print(f"val_bpb: {meta['val_bpb']:.4f}")

    # Training efficiency
    if meta.get("total_steps"):
        eff = compute_efficiency(meta, meta.get("val_bpb", 8.0))
        print(f"\n{'--- Training Efficiency ---'}")
        for k, v in eff.items():
            print(f"  {k:<20} {v}")

    print(f"\n{'--- Generation Quality ---'}")
    for k, v in m.items():
        if k != "n_samples":
            print(f"  {k:<24} {v}")
    if result["best_sample"]:
        ppl, s = result["best_sample"]
        print(f"\n--- Best (self_ppl={ppl:.1f}) ---")
        print(f'prompt: "{s["prompt"]}"')
        print(f"> {s['generated'][:200].replace(chr(10), ' ')}")
    if result["worst_sample"]:
        ppl, s = result["worst_sample"]
        print(f"\n--- Worst (self_ppl={ppl:.1f}) ---")
        print(f'prompt: "{s["prompt"]}"')
        print(f"> {s['generated'][:200].replace(chr(10), ' ')}")


def print_comparison(results: list[tuple[str, dict]]):
    labels = [r[0] for r in results]
    print("\n=== Quality Comparison ===")
    header = f"{'metric':<24}" + "".join(f"{l:<16}" for l in labels)
    if len(results) == 2:
        header += "diff"
    print(header)
    print("-" * (24 + 16 * len(labels) + 12))
    better_higher = {"unique_token_ratio", "sentence_completion", "local_coherence"}
    better_lower = {"repetition_3gram", "self_perplexity"}
    for k in results[0][1]["metrics"]:
        if k == "n_samples":
            continue
        vals = [r[1]["metrics"][k] for r in results]
        row = f"{k:<24}" + "".join(f"{v:<16}" for v in vals)
        if len(results) == 2 and vals[0] != 0:
            d = (vals[1] - vals[0]) / abs(vals[0]) * 100
            tag = ""
            if k in better_lower:
                tag = " (better)" if d < 0 else " (worse)"
            elif k in better_higher:
                tag = " (better)" if d > 0 else " (worse)"
            row += f"{d:+.0f}%{tag}"
        print(row)

    print("\n=== Head-to-Head Samples ===")
    for prompt in EVAL_PROMPTS[:3]:
        print(f'\nprompt: "{prompt}"')
        for label, result in results:
            for s in result["samples"]:
                if s["prompt"] == prompt:
                    print(f"  [{label}] {s['generated'][:150].replace(chr(10), ' ')}")
                    break


def write_report(results: list[tuple[str, dict]], path: str):
    lines = ["# Output Quality Report\n"]

    # Training efficiency table
    lines.append("## Training Efficiency\n")
    lines.append("| model | val_bpb | steps | overhead | bpb/kstep | bpb/min |")
    lines.append("|-------|---------|-------|----------|-----------|---------|")
    for label, r in results:
        meta = r.get("meta", {})
        vb = meta.get("val_bpb", 0)
        eff = compute_efficiency(meta, vb) if meta.get("total_steps") else {}
        lines.append(
            f"| {label} | {vb:.4f} | {meta.get('total_steps', '?')} | "
            f"{eff.get('ca_overhead_pct', '?')}% | "
            f"{eff.get('bpb_per_kstep', '?')} | {eff.get('bpb_per_min', '?')} |"
        )

    # Generation quality table
    lines.append("\n## Generation Quality\n")
    lines.append("| model | repetition | diversity | completion | self_ppl | coherence | word_len |")
    lines.append("|-------|------------|-----------|------------|----------|-----------|----------|")
    for label, r in results:
        m = r["metrics"]
        lines.append(
            f"| {label} | {m['repetition_3gram']:.3f} | {m['unique_token_ratio']:.3f} | "
            f"{m['sentence_completion']:.3f} | {m['self_perplexity']:.1f} | "
            f"{m['local_coherence']:.3f} | {m['mean_word_length']:.1f} |"
        )
    lines.append("\n## Sample Outputs\n")
    for label, r in results:
        lines.append(f"### {label}\n")
        for s in r["samples"][:3]:
            lines.append(f"**Prompt:** {s['prompt']}")
            lines.append(f"> {s['generated'][:200].replace(chr(10), ' ')}\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {path}")


# ---------------------------------------------------------------------------
# Load / train
# ---------------------------------------------------------------------------

def load_from_checkpoint(path: str):
    from train_r4 import GPT, ARCHS, CHANNELS, DEPTH, N_HEADS, N_KV_HEADS
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    arch_cfg = ARCHS.get(ckpt.get("arch", "baseline"), {})
    model = GPT(VOCAB_SIZE, MAX_SEQ_LEN, DEPTH, N_HEADS, N_KV_HEADS,
                CHANNELS, arch_cfg=arch_cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, {k: v for k, v in ckpt.items() if k != "model_state_dict"}


def train_and_evaluate(arch: str, seed: int, minutes: float):
    from train_r4 import train
    result = train(time_budget=minutes * 60, seed=seed, arch=arch, quiet=True)
    model = result.get("_model")
    if model is None:
        print("ERROR: train() did not return _model")
        sys.exit(1)
    return model, {"val_bpb": result["val_bpb"], "arch": arch, "seed": seed}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NeuroGen Quality Evaluation")
    parser.add_argument("--checkpoint", type=str, nargs="+")
    parser.add_argument("--compare", type=str, nargs=2)
    parser.add_argument("--glob", type=str)
    parser.add_argument("--report", type=str, default=None)
    parser.add_argument("--live", action="store_true",
                        help="Train and evaluate (no checkpoint needed)")
    parser.add_argument("--arch", type=str, nargs="+", default=["baseline"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minutes", type=float, default=40)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    results = []

    if args.live:
        for arch in args.arch:
            print(f"\n--- Training {arch} seed={args.seed} {args.minutes}min ---")
            model, meta = train_and_evaluate(arch, args.seed, args.minutes)
            label = f"{arch}_s{args.seed}"
            print(f"val_bpb: {meta['val_bpb']:.4f}  Generating samples...")
            r = evaluate_model(model, DEVICE, n_samples=args.samples,
                               max_tokens=args.max_tokens)
            r["meta"] = meta
            results.append((label, r))
            print_evaluation(r, label)
            del model

    elif args.compare:
        for path in args.compare:
            model, meta = load_from_checkpoint(path)
            label = Path(path).stem
            print(f"Evaluating {label}...")
            r = evaluate_model(model, DEVICE, n_samples=args.samples,
                               max_tokens=args.max_tokens)
            r["meta"] = meta
            results.append((label, r))
            del model

    elif args.checkpoint:
        for path in args.checkpoint:
            model, meta = load_from_checkpoint(path)
            label = Path(path).stem
            r = evaluate_model(model, DEVICE, n_samples=args.samples,
                               max_tokens=args.max_tokens)
            r["meta"] = meta
            results.append((label, r))
            print_evaluation(r, label)
            del model

    elif args.glob:
        for path in sorted(glob.glob(args.glob)):
            model, meta = load_from_checkpoint(path)
            label = Path(path).stem
            print(f"Evaluating {label}...")
            r = evaluate_model(model, DEVICE, n_samples=args.samples,
                               max_tokens=args.max_tokens)
            r["meta"] = meta
            results.append((label, r))
            del model

    else:
        parser.print_help()
        return

    if len(results) >= 2:
        print_comparison(results)
    if args.report and results:
        write_report(results, args.report)
    summary = [{"label": l, **r["metrics"], **(r.get("meta", {}))} for l, r in results]
    print(f"\nQUALITY_JSON: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
