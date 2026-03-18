"""
Output quality evaluation for NeuroGen models.

Generates text from a trained model and measures quality automatically.
Complements val_bpb (prediction quality) with generation quality metrics.

Usage:
    uv run evaluate_quality.py                                    # train default, evaluate
    uv run evaluate_quality.py --methods "default,xavier,grid_ca" # compare methods
    uv run evaluate_quality.py --quality-over-time --methods "default,grid_ca"
"""

import argparse
import math
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from prepare import (
    load_data, get_batch, evaluate_val_bpb, get_device,
    VOCAB_SIZE, MAX_SEQ_LEN, TIME_BUDGET,
)
from train import GPT, get_lr, DEPTH, CHANNELS, N_HEADS, N_KV_HEADS, LR, WEIGHT_DECAY
from benchmark import apply_init

DEVICE = get_device()
BATCH_SIZE = 64
OUTPUTS_DIR = Path("outputs")

# ---------------------------------------------------------------------------
# Evaluation prompts (TinyStories domain)
# ---------------------------------------------------------------------------

EVAL_PROMPTS = [
    # Narrative continuation
    "Once upon a time, there was a little",
    "The cat sat on the mat and looked at",
    "Sarah walked into the room and saw",
    # Dialogue
    '"Can you help me?" asked the',
    '"I don\'t understand," said Tom. "Why did you',
    # Description
    "The house was old and",
    "It was a beautiful sunny day. The children",
    # Abstract
    "The most important thing about being kind is",
    "When you make a mistake, you should",
    # Longer context
    "There was a little bear who lived in the forest. Every morning, he would wake up and",
]

GENERATION_CONFIG = {
    "max_new_tokens": 100,
    "temperature": 0.8,
    "top_k": 50,
    "num_samples_per_prompt": 3,
}

# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate_text(
    model: GPT,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
) -> str:
    """Generate text from a byte-level model given a string prompt."""
    model.eval()
    block_size = model.block_size
    prompt_bytes = prompt.encode("utf-8")
    ids = torch.tensor([list(prompt_bytes)], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(ids[:, -block_size:])
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            try:
                nxt = torch.multinomial(probs, 1)
            except RuntimeError:
                nxt = torch.multinomial(probs.cpu(), 1).to(DEVICE)
            ids = torch.cat([ids, nxt], 1)

    output_bytes = bytes(ids[0].tolist())
    return output_bytes.decode("utf-8", errors="replace")


def generate_samples(model: GPT, prompts: list[str], config: dict) -> list[dict]:
    """Generate multiple samples per prompt. Returns list of {prompt, text, generated}."""
    samples = []
    for prompt in prompts:
        for _ in range(config["num_samples_per_prompt"]):
            full_text = generate_text(
                model, prompt,
                max_new_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
                top_k=config["top_k"],
            )
            generated = full_text[len(prompt):]
            samples.append({"prompt": prompt, "text": full_text, "generated": generated})
    return samples


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def repetition_rate(text: str, n: int = 3) -> float:
    """Fraction of n-grams that are repeated. Lower is better."""
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def unique_token_ratio(text: str) -> float:
    """Unique words / total words. Higher is better."""
    words = text.split()
    return len(set(words)) / len(words) if words else 0.0


def mean_word_length(text: str) -> float:
    """Average characters per word. Healthy English: 4-5."""
    words = text.split()
    return sum(len(w) for w in words) / len(words) if words else 0.0


def sentence_completion_rate(text: str) -> float:
    """Fraction of sentences ending with punctuation."""
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1:
        return 0.0
    # Last chunk is likely incomplete, count completed ones
    return (len(sentences) - 1) / len(sentences)


def local_coherence(text: str, window: int = 10) -> float:
    """Within sliding windows, ratio of unique words (topical consistency proxy)."""
    words = text.lower().split()
    if len(words) <= window:
        return len(set(words)) / len(words) if words else 0.0
    scores = []
    for i in range(len(words) - window):
        chunk = words[i:i+window]
        scores.append(len(set(chunk)) / len(chunk))
    return sum(scores) / len(scores) if scores else 0.0


def self_perplexity(model: GPT, text: str) -> float:
    """Feed generated text back through model, measure loss.
    Lower = model is confident in its own output."""
    text_bytes = text.encode("utf-8")
    if len(text_bytes) < 4:
        return float("inf")
    ids = torch.tensor([list(text_bytes)], dtype=torch.long, device=DEVICE)
    block_size = model.block_size
    ids = ids[:, :block_size + 1]  # trim to fit
    if ids.size(1) < 2:
        return float("inf")
    x, y = ids[:, :-1], ids[:, 1:]
    model.eval()
    with torch.no_grad():
        _, loss = model(x, y)
    return loss.exp().item()


def compute_quality_metrics(model: GPT, samples: list[dict]) -> dict:
    """Compute all quality metrics across samples. Returns mean of each metric."""
    metrics = {
        "repetition_3gram": [],
        "unique_token_ratio": [],
        "mean_word_length": [],
        "sentence_completion": [],
        "local_coherence": [],
        "self_perplexity": [],
    }
    for s in samples:
        gen = s["generated"]
        metrics["repetition_3gram"].append(repetition_rate(gen, n=3))
        metrics["unique_token_ratio"].append(unique_token_ratio(gen))
        metrics["mean_word_length"].append(mean_word_length(gen))
        metrics["sentence_completion"].append(sentence_completion_rate(gen))
        metrics["local_coherence"].append(local_coherence(gen))
        metrics["self_perplexity"].append(self_perplexity(model, s["text"]))
    return {k: statistics.mean(v) if v else 0.0 for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Train a model with given init method
# ---------------------------------------------------------------------------

def train_model(method: str, seed: int, minutes: float, stop_at_step: int | None = None):
    """Train and return (model, val_bpb, steps)."""
    torch.manual_seed(seed)
    train_data = load_data("train")
    val_data = load_data("val")
    block_size = MAX_SEQ_LEN
    time_budget = minutes * 60

    model = GPT(VOCAB_SIZE, block_size, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS).to(DEVICE)
    apply_init(model, method)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()
    step = 0
    warmup = 100
    max_steps = 100_000
    min_lr = LR / 10
    t0 = time.time()

    while True:
        if stop_at_step is not None and step >= stop_at_step:
            break
        if stop_at_step is None and time.time() - t0 >= time_budget:
            break
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

    val_bpb = evaluate_val_bpb(model, val_data, BATCH_SIZE, block_size, DEVICE)
    return model, val_bpb, step


# ---------------------------------------------------------------------------
# Quality-over-time evaluation
# ---------------------------------------------------------------------------

def quality_over_training(
    method: str, seed: int, eval_steps: list[int], minutes: float,
) -> list[dict]:
    """Train and evaluate quality at checkpoints."""
    torch.manual_seed(seed)
    train_data = load_data("train")
    val_data = load_data("val")
    block_size = MAX_SEQ_LEN
    time_budget = minutes * 60

    model = GPT(VOCAB_SIZE, block_size, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS).to(DEVICE)
    apply_init(model, method)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()
    step = 0
    warmup = 100
    max_steps = 100_000
    min_lr = LR / 10
    t0 = time.time()

    eval_steps_sorted = sorted(eval_steps)
    next_eval_idx = 0
    curve = []

    while time.time() - t0 < time_budget:
        # Check if we should evaluate at this step
        while next_eval_idx < len(eval_steps_sorted) and step >= eval_steps_sorted[next_eval_idx]:
            val_bpb = evaluate_val_bpb(model, val_data, BATCH_SIZE, block_size, DEVICE)
            samples = generate_samples(model, EVAL_PROMPTS[:3], {
                **GENERATION_CONFIG, "num_samples_per_prompt": 1,
            })
            metrics = compute_quality_metrics(model, samples)
            metrics["step"] = eval_steps_sorted[next_eval_idx]
            metrics["val_bpb"] = val_bpb
            metrics["method"] = method
            curve.append(metrics)
            model.train()
            next_eval_idx += 1

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

    # Final evaluation
    val_bpb = evaluate_val_bpb(model, val_data, BATCH_SIZE, block_size, DEVICE)
    samples = generate_samples(model, EVAL_PROMPTS[:3], {
        **GENERATION_CONFIG, "num_samples_per_prompt": 1,
    })
    metrics = compute_quality_metrics(model, samples)
    metrics["step"] = step
    metrics["val_bpb"] = val_bpb
    metrics["method"] = method
    curve.append(metrics)

    return curve


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_comparison_report(
    results: dict[str, dict], samples: dict[str, list[dict]],
) -> str:
    """Format a quality comparison report."""
    lines = [
        "# Output Quality Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Model:** depth={DEPTH}, channels={CHANNELS} | **Device:** {DEVICE}",
        f"**Samples:** {len(EVAL_PROMPTS)} prompts x {GENERATION_CONFIG['num_samples_per_prompt']} samples",
        "",
        "## Quality Metrics",
        "",
    ]

    # Build comparison table
    methods = list(results.keys())
    header = "| metric | " + " | ".join(methods) + " |"
    sep = "|--------|" + "|".join(["--------"] * len(methods)) + "|"
    lines += [header, sep]

    metric_names = [
        "val_bpb", "repetition_3gram", "unique_token_ratio",
        "mean_word_length", "sentence_completion", "local_coherence", "self_perplexity",
    ]
    for metric in metric_names:
        vals = []
        for m in methods:
            v = results[m].get(metric, 0.0)
            vals.append(f"{v:.4f}")
        lines.append(f"| {metric} | " + " | ".join(vals) + " |")

    # Sample outputs
    lines += ["", "## Sample Outputs (temperature=0.8)", ""]
    for prompt in EVAL_PROMPTS[:3]:
        lines.append(f"**Prompt:** \"{prompt}\"")
        lines.append("")
        for m in methods:
            method_samples = [s for s in samples[m] if s["prompt"] == prompt]
            if method_samples:
                gen = method_samples[0]["generated"][:200]
                lines.append(f"[{m}] {prompt}{gen}")
        lines.append("")

    return "\n".join(lines) + "\n"


def format_quality_over_time(curves: dict[str, list[dict]]) -> str:
    """Format quality-over-time report."""
    lines = [
        "# Quality Over Training Time",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    for method, curve in curves.items():
        lines += [f"## {method}", ""]
        lines.append("| step | val_bpb | repetition | unique_ratio | sent_complete | self_ppl |")
        lines.append("|------|---------|------------|--------------|---------------|----------|")
        for point in curve:
            lines.append(
                f"| {point['step']} | {point['val_bpb']:.4f} "
                f"| {point['repetition_3gram']:.3f} "
                f"| {point['unique_token_ratio']:.3f} "
                f"| {point['sentence_completion']:.3f} "
                f"| {point['self_perplexity']:.1f} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NeuroGen output quality evaluation")
    parser.add_argument(
        "--methods", type=str, default="default",
        help="Comma-separated init methods to evaluate",
    )
    parser.add_argument("--minutes", type=float, default=2.0, help="Training time per method")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--quality-over-time", action="store_true",
        help="Evaluate quality at checkpoints during training",
    )
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    OUTPUTS_DIR.mkdir(exist_ok=True)

    if args.quality_over_time:
        eval_steps = [50, 100, 150, 200, 250]
        print(f"Quality over time: {methods} | eval at steps {eval_steps}")
        print()
        curves = {}
        for method in methods:
            print(f"Training {method}...", flush=True)
            curve = quality_over_training(method, args.seed, eval_steps, args.minutes)
            curves[method] = curve
            for point in curve:
                print(
                    f"  step {point['step']:4d} | val_bpb {point['val_bpb']:.4f} "
                    f"| rep {point['repetition_3gram']:.3f} "
                    f"| uniq {point['unique_token_ratio']:.3f} "
                    f"| sent {point['sentence_completion']:.3f}"
                )
            print()

        report = format_quality_over_time(curves)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = OUTPUTS_DIR / f"quality_over_time_{ts}.md"
        report_path.write_text(report)
        print(f"Report saved to {report_path}")
        print()
        print(report)
        return

    # Standard quality comparison
    all_results = {}
    all_samples = {}

    for method in methods:
        print(f"Training and evaluating {method}...", flush=True)
        model, val_bpb, steps = train_model(method, args.seed, args.minutes)
        print(f"  val_bpb={val_bpb:.4f} ({steps} steps)")

        print(f"  Generating samples...", flush=True)
        samples = generate_samples(model, EVAL_PROMPTS, GENERATION_CONFIG)
        metrics = compute_quality_metrics(model, samples)
        metrics["val_bpb"] = val_bpb
        all_results[method] = metrics
        all_samples[method] = samples

        print(
            f"  repetition={metrics['repetition_3gram']:.3f} "
            f"unique={metrics['unique_token_ratio']:.3f} "
            f"sent_complete={metrics['sentence_completion']:.3f} "
            f"self_ppl={metrics['self_perplexity']:.1f}"
        )
        print()

    # Report
    report = format_comparison_report(all_results, all_samples)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUTS_DIR / f"quality_{ts}.md"
    report_path.write_text(report)
    print(f"Report saved to {report_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
