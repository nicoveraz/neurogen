"""
NeuroGen: Side-by-side story generation comparison.

Generates stories from both baseline and quartic models using identical
prompts and seeds, computes quality metrics, saves formatted output.

Usage:
    python generate_comparison.py
"""

import sys, json, re
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from prepare import VOCAB_SIZE, MAX_SEQ_LEN
from evaluate_quality import load_from_checkpoint, generate_text, repetition_rate, unique_token_ratio
from train_r4 import get_device

DEVICE = get_device()

PROMPTS = [
    "Once upon a time there was a",
    "The little girl walked into the",
    "One day, the cat decided to",
    "Tom was very sad because",
    "The teacher told the children to",
    "It was a dark and stormy",
    "Mom said we could go to the",
    "The dog found a big red",
    "Lucy wanted to learn how to",
    "After school, the boy went to",
    "There was a tiny bird who",
    "The old man sat by the",
]

CHECKPOINTS = [
    ("Baseline", "checkpoints/model_baseline_42.pt"),
    ("Quartic",  "checkpoints/model_window_power_4.0_42.pt"),
]


def four_gram_repetition(text):
    """Fraction of 4-grams that are repeated."""
    words = text.split()
    if len(words) < 5:
        return 0.0
    ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
    return 1.0 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def generate_all():
    models = {}
    for label, ckpt_path in CHECKPOINTS:
        print(f"Loading {label}...")
        model, meta = load_from_checkpoint(ckpt_path)
        models[label] = model

    all_generations = {}
    for label, model in models.items():
        all_generations[label] = []
        print(f"\nGenerating from {label}...")
        torch.manual_seed(42)
        for i, prompt in enumerate(PROMPTS):
            torch.manual_seed(42 + i)
            text = generate_text(model, prompt, max_tokens=200,
                                 temperature=0.8, top_k=50, device=DEVICE)
            generated = text[len(prompt):]
            all_generations[label].append({
                "prompt": prompt,
                "generated": generated,
                "full": text,
            })
            # Print inline
            print(f"  [{i+1:2d}] {prompt}...")

    # Print side-by-side
    print("\n" + "=" * 80)
    print("  SIDE-BY-SIDE COMPARISON")
    print("=" * 80)

    output_lines = []
    output_lines.append("# NeuroGen Story Generation Comparison")
    output_lines.append(f"# Baseline vs Quartic (γ=4), seed=42, temperature=0.8, top_k=50")
    output_lines.append(f"# max_tokens=200, model=3.4M params, 20k training steps")
    output_lines.append("")

    for i, prompt in enumerate(PROMPTS):
        print(f"\n{'─'*70}")
        print(f"  PROMPT {i+1}: \"{prompt}\"")
        print(f"{'─'*70}")

        output_lines.append(f"{'='*70}")
        output_lines.append(f"PROMPT {i+1}: \"{prompt}\"")
        output_lines.append(f"{'='*70}")

        for label in ["Baseline", "Quartic"]:
            gen = all_generations[label][i]
            text = gen["generated"]
            print(f"\n  [{label}]:")
            # Word wrap at 70 chars
            words = text.split()
            line = "    "
            for w in words:
                if len(line) + len(w) + 1 > 74:
                    print(line)
                    line = "    " + w
                else:
                    line = line + " " + w if line.strip() else "    " + w
            if line.strip():
                print(line)

            output_lines.append(f"\n[{label}]:")
            output_lines.append(text)
            output_lines.append("")

    # Compute aggregate metrics
    print(f"\n{'='*80}")
    print(f"  AGGREGATE METRICS")
    print(f"{'='*80}")

    output_lines.append(f"\n{'='*70}")
    output_lines.append(f"AGGREGATE METRICS")
    output_lines.append(f"{'='*70}")

    metrics_table = []
    for label in ["Baseline", "Quartic"]:
        gens = all_generations[label]
        texts = [g["generated"] for g in gens]

        total_words = sum(len(t.split()) for t in texts)
        mean_len = total_words / len(texts)
        rep_4gram = sum(four_gram_repetition(t) for t in texts) / len(texts)
        all_words = []
        for t in texts:
            all_words.extend(t.split())
        vocab_div = len(set(all_words)) / len(all_words) if all_words else 0

        metrics = {
            "label": label,
            "mean_words": round(mean_len, 1),
            "repetition_4gram": round(rep_4gram, 4),
            "vocab_diversity": round(vocab_div, 4),
        }
        metrics_table.append(metrics)

        line = f"  {label:<12} words={mean_len:.1f}  4gram_rep={rep_4gram:.4f}  vocab_div={vocab_div:.4f}"
        print(line)
        output_lines.append(line)

    # Save to file
    with open("generated_samples.txt", "w") as f:
        f.write("\n".join(output_lines))
    print(f"\nSaved to generated_samples.txt")

    # Save metrics JSON
    with open("gradient_results/generation_metrics.json", "w") as f:
        json.dump({"metrics": metrics_table, "generations": {
            label: [{"prompt": g["prompt"], "generated": g["generated"]}
                    for g in gens]
            for label, gens in all_generations.items()
        }}, f, indent=2)
    print(f"Saved to gradient_results/generation_metrics.json")

    # Identify most contrasting pairs
    print(f"\n{'='*80}")
    print(f"  MOST CONTRASTING PAIRS (for README)")
    print(f"{'='*80}")

    contrasts = []
    for i, prompt in enumerate(PROMPTS):
        bl_text = all_generations["Baseline"][i]["generated"]
        q4_text = all_generations["Quartic"][i]["generated"]
        bl_rep = four_gram_repetition(bl_text)
        q4_rep = four_gram_repetition(q4_text)
        bl_div = unique_token_ratio(bl_text)
        q4_div = unique_token_ratio(q4_text)
        # Score: higher = more contrasting (quartic better)
        score = (bl_rep - q4_rep) + (q4_div - bl_div) + (len(q4_text.split()) - len(bl_text.split())) / 100
        contrasts.append((score, i, prompt))

    contrasts.sort(reverse=True)
    print("\nTop 4 most contrasting (quartic advantage):")
    for score, i, prompt in contrasts[:4]:
        print(f"  [{i+1}] score={score:+.3f} \"{prompt}\"")
        bl = all_generations["Baseline"][i]["generated"][:150]
        q4 = all_generations["Quartic"][i]["generated"][:150]
        print(f"      BL: {bl}...")
        print(f"      Q4: {q4}...")


if __name__ == "__main__":
    generate_all()
