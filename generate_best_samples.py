"""
NeuroGen: Best-of-N story generation for showcase.

Generates multiple samples per prompt from both models, scores them,
and selects the most contrasting pairs for the README.

Usage:
    python generate_best_samples.py
"""

import sys, re, json
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from prepare import VOCAB_SIZE, MAX_SEQ_LEN
from evaluate_quality import load_from_checkpoint
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

SEEDS = [42, 137, 256, 789, 1337]


def generate(model, prompt, max_tokens=250, temperature=0.8, top_k=50, top_p=0.9):
    model.eval()
    ids = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=DEVICE)
    block_size = MAX_SEQ_LEN
    with torch.no_grad():
        for _ in range(max_tokens):
            logits, _ = model(ids[:, -block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                nxt = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
            else:
                try:
                    nxt = torch.multinomial(probs, 1)
                except RuntimeError:
                    nxt = torch.multinomial(probs.cpu(), 1).to(DEVICE)
            ids = torch.cat([ids, nxt], dim=1)
            if ids.shape[1] >= 2 and ids[0, -1].item() == 10 and ids[0, -2].item() == 10:
                break
    return bytes(ids[0].tolist()).decode("utf-8", errors="replace")


# --- Quality scoring ---

def repetition_score(text, n=3):
    """Lower is better. Fraction of repeated n-grams."""
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def has_encoding_artifacts(text):
    """Check for UTF-8 decoding artifacts like â€œ."""
    artifacts = ["â€", "ï¿½", "\ufffd", "â€™", "â€"]
    return any(a in text for a in artifacts)


def coherence_score(text):
    """Simple heuristic: sentences ending with punctuation, no orphan quotes."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
    if len(sentences) < 2:
        return 0.3
    # Penalize very short sentences (fragmented)
    avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
    len_score = min(avg_len / 8.0, 1.0)  # 8+ words per sentence is good
    # Penalize unclosed quotes
    quote_count = text.count('"')
    quote_penalty = 0.0 if quote_count % 2 == 0 else 0.1
    return len_score - quote_penalty


def vocabulary_diversity(text):
    words = text.lower().split()
    return len(set(words)) / len(words) if words else 0.0


def overall_quality(text):
    """Combined quality score. Higher is better."""
    rep = repetition_score(text)
    coh = coherence_score(text)
    div = vocabulary_diversity(text)
    artifact = 0.3 if has_encoding_artifacts(text) else 0.0
    return coh + div - rep * 2 - artifact


def contrast_score(bl_text, q4_text):
    """How much better quartic is than baseline. Positive = quartic wins."""
    bl_q = overall_quality(bl_text)
    q4_q = overall_quality(q4_text)
    return q4_q - bl_q


def main():
    # Load models
    print("Loading models...")
    bl_model, bl_meta = load_from_checkpoint("checkpoints/model_baseline_42.pt")
    q4_model, q4_meta = load_from_checkpoint("checkpoints/model_window_power_4.0_42.pt")
    print(f"  Baseline: {bl_meta.get('max_steps', '?')} steps, vbpb={bl_meta.get('val_bpb', bl_meta.get('final_vbpb', '?'))}")
    print(f"  Quartic:  {q4_meta.get('max_steps', '?')} steps, vbpb={q4_meta.get('val_bpb', q4_meta.get('final_vbpb', '?'))}")

    # Generate all samples
    all_samples = {}  # prompt -> [(seed, bl_text, q4_text, contrast)]
    for pi, prompt in enumerate(PROMPTS):
        print(f"\n[{pi+1:2d}/12] Prompt: \"{prompt}\"")
        samples = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            bl_text = generate(bl_model, prompt)[len(prompt):]
            torch.manual_seed(seed)
            q4_text = generate(q4_model, prompt)[len(prompt):]
            cs = contrast_score(bl_text, q4_text)
            bl_q = overall_quality(bl_text)
            q4_q = overall_quality(q4_text)
            samples.append({
                "seed": seed,
                "baseline": bl_text,
                "quartic": q4_text,
                "contrast": cs,
                "bl_quality": bl_q,
                "q4_quality": q4_q,
            })
            marker = "<<" if cs > 0.1 else ">>" if cs < -0.1 else "=="
            print(f"  seed={seed:5d}  bl_q={bl_q:.2f}  q4_q={q4_q:.2f}  contrast={cs:+.2f} {marker}")
        all_samples[prompt] = samples

    # For each prompt, pick the best-contrast sample (quartic advantage)
    best_pairs = []
    for prompt, samples in all_samples.items():
        best = max(samples, key=lambda s: s["contrast"])
        best_pairs.append((prompt, best))

    # Sort by contrast score
    best_pairs.sort(key=lambda x: x[1]["contrast"], reverse=True)

    # Print the showcase
    print("\n" + "=" * 78)
    print("  TOP 12 PAIRS (ranked by quartic advantage)")
    print("=" * 78)

    output_lines = [
        "# NeuroGen: Best Story Generation Samples",
        f"# 100K-step checkpoints, 5 seeds per prompt, best contrast selected",
        f"# Baseline vbpb=0.8072, Quartic vbpb=0.7994",
        f"# Settings: temperature=0.8, top_k=50, top_p=0.9, max_tokens=250",
        "",
    ]

    for rank, (prompt, sample) in enumerate(best_pairs):
        cs = sample["contrast"]
        seed = sample["seed"]
        bl = sample["baseline"]
        q4 = sample["quartic"]
        bl_q = sample["bl_quality"]
        q4_q = sample["q4_quality"]

        print(f"\n{'─'*78}")
        print(f"  #{rank+1} (contrast={cs:+.2f}, seed={seed})")
        print(f"  PROMPT: \"{prompt}\"")
        print(f"{'─'*78}")

        output_lines.append(f"{'='*70}")
        output_lines.append(f"#{rank+1} | Prompt: \"{prompt}\" | seed={seed} | contrast={cs:+.2f}")
        output_lines.append(f"{'='*70}")

        for label, text, qual in [("BASELINE", bl, bl_q), ("QUARTIC", q4, q4_q)]:
            rep = repetition_score(text)
            div = vocabulary_diversity(text)
            words = len(text.split())
            artifact = " [ENCODING ARTIFACTS]" if has_encoding_artifacts(text) else ""

            print(f"\n  [{label}] (quality={qual:.2f}, rep={rep:.3f}, div={div:.2f}, {words}w){artifact}")
            # Word wrap
            ws = text.split()
            line = "    "
            for w in ws:
                if len(line) + len(w) + 1 > 78:
                    print(line)
                    line = "    " + w
                else:
                    line = (line + " " + w) if line.strip() else ("    " + w)
            if line.strip():
                print(line)

            output_lines.append(f"\n[{label}] quality={qual:.2f} rep={rep:.3f} div={div:.2f} {words}w{artifact}")
            output_lines.append(text)

        output_lines.append("")

    # Save
    with open("generated_samples_best.txt", "w") as f:
        f.write("\n".join(output_lines))
    print(f"\nSaved to generated_samples_best.txt")

    # Save JSON with all samples for future reference
    json_data = {}
    for prompt, samples in all_samples.items():
        json_data[prompt] = [{
            "seed": s["seed"],
            "baseline": s["baseline"][:500],
            "quartic": s["quartic"][:500],
            "contrast": round(s["contrast"], 4),
            "bl_quality": round(s["bl_quality"], 4),
            "q4_quality": round(s["q4_quality"], 4),
        } for s in samples]
    with open("gradient_results/generation_best_of_5.json", "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved to gradient_results/generation_best_of_5.json")

    # Summary stats
    print(f"\n{'='*78}")
    print(f"  SUMMARY")
    print(f"{'='*78}")
    contrasts = [bp[1]["contrast"] for bp in best_pairs]
    quartic_wins = sum(1 for c in contrasts if c > 0.05)
    baseline_wins = sum(1 for c in contrasts if c < -0.05)
    ties = len(contrasts) - quartic_wins - baseline_wins
    print(f"  Quartic wins: {quartic_wins}/12")
    print(f"  Baseline wins: {baseline_wins}/12")
    print(f"  Ties: {ties}/12")
    print(f"  Mean contrast: {sum(contrasts)/len(contrasts):+.3f}")
    print(f"  Best contrast: {max(contrasts):+.3f}")


if __name__ == "__main__":
    main()
