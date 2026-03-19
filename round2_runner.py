"""
Round 2 experiment runner for NeuroGen.
Orchestrates all three tracks: Horizon Validation, Blend & Pattern, Output Quality.

Usage:
    uv run round2_runner.py              # run all tracks
    uv run round2_runner.py --track 1    # run only Track 1
    uv run round2_runner.py --track 2    # run only Track 2
    uv run round2_runner.py --track 3    # run only Track 3
"""

import argparse
import json
import time
import sys
from collections import defaultdict

import torch

# Import train function directly
from train import train, generate, compute_text_quality, PROMPTS, DEVICE


def run_experiment(init_method: str, minutes: float, seed: int,
                   label: str = "") -> dict:
    """Run a single training experiment and return results."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  init={init_method}  minutes={minutes}  seed={seed}")
    print(f"{'='*60}")
    sys.stdout.flush()

    result = train(
        time_budget=minutes * 60,
        seed=seed,
        init_method=init_method,
        quiet=True,
    )
    print(f"  => val_bpb={result['val_bpb']:.4f}  steps={result['total_steps']}  "
          f"time={result['wall_time_s']:.1f}s")
    sys.stdout.flush()
    return result


def track1_horizon_validation():
    """Track 1: Does the CA advantage hold at longer training?"""
    print("\n" + "="*70)
    print("  TRACK 1: HORIZON VALIDATION")
    print("="*70)

    horizons = [2, 10, 30]
    methods = ["xavier", "xavier_ca10"]
    seeds = [42, 137, 256]

    all_results = {}

    for minutes in horizons:
        for method in methods:
            for seed in seeds:
                key = (minutes, method, seed)
                label = f"Track1: {method} @ {minutes}min, seed={seed}"
                result = run_experiment(method, minutes, seed, label)
                all_results[key] = result

    # Build comparison table
    print("\n\n" + "="*70)
    print("  === CA Init Horizon Validation (3 seeds) ===")
    print("="*70)
    print(f"{'horizon':<10} {'xavier (mean±std)':<22} {'ca10 (mean±std)':<22} {'improvement':<14} {'trend'}")
    print("-" * 80)

    prev_improvement = None
    table_rows = []

    for minutes in horizons:
        xavier_bpbs = [all_results[(minutes, "xavier", s)]["val_bpb"] for s in seeds]
        ca10_bpbs = [all_results[(minutes, "xavier_ca10", s)]["val_bpb"] for s in seeds]

        xavier_mean = sum(xavier_bpbs) / len(xavier_bpbs)
        xavier_std = (sum((x - xavier_mean)**2 for x in xavier_bpbs) / len(xavier_bpbs))**0.5
        ca10_mean = sum(ca10_bpbs) / len(ca10_bpbs)
        ca10_std = (sum((x - ca10_mean)**2 for x in ca10_bpbs) / len(ca10_bpbs))**0.5

        improvement = (xavier_mean - ca10_mean) / xavier_mean * 100

        if prev_improvement is None:
            trend = "-"
        elif improvement > prev_improvement + 0.5:
            trend = "↑"
        elif improvement < prev_improvement - 0.5:
            trend = "↓"
        else:
            trend = "→"
        prev_improvement = improvement

        row = {
            "horizon_min": minutes,
            "xavier_mean": xavier_mean,
            "xavier_std": xavier_std,
            "ca10_mean": ca10_mean,
            "ca10_std": ca10_std,
            "improvement_pct": improvement,
            "trend": trend,
        }
        table_rows.append(row)

        print(f"{minutes:>3} min    {xavier_mean:.4f} ± {xavier_std:.4f}    "
              f"{ca10_mean:.4f} ± {ca10_std:.4f}    "
              f"{improvement:+.1f}%         {trend}")

    # Print loss curves
    print("\n  --- Loss Curves ---")
    for minutes in horizons:
        print(f"\n  {minutes}-minute horizon:")
        for method in methods:
            for seed in seeds:
                curve = all_results[(minutes, method, seed)]["loss_curve"]
                curve_str = "  ".join(f"s{s}:{v:.4f}" for s, _, v in curve[-5:])
                print(f"    {method} seed={seed}: {curve_str}")

    # Determine verdict
    last = table_rows[-1]["improvement_pct"]
    first = table_rows[0]["improvement_pct"]
    if last < -1.0:
        verdict = "HARMFUL — CA init hurts at longer training"
    elif last < 0.5:
        verdict = "HEAD START — CA advantage vanishes with more training"
    elif abs(last - first) < 1.0:
        verdict = "CONSTANT OFFSET — CA gives a fixed improvement"
    elif last > first + 1.0:
        verdict = "LASTING BENEFIT — CA advantage grows with training"
    else:
        verdict = "CONSTANT OFFSET — CA gives a fixed improvement"

    print(f"\n  Verdict: {verdict}")

    return {
        "table": table_rows,
        "all_results": {str(k): {kk: vv for kk, vv in v.items() if kk != "_model"}
                        for k, v in all_results.items()},
        "verdict": verdict,
        "ca_harmful": last < -1.0,
    }


def track2_blend_and_pattern(use_10min: bool = True):
    """Track 2: Optimize blend ratio and CA pattern type."""
    print("\n" + "="*70)
    print("  TRACK 2: BLEND & PATTERN OPTIMIZATION")
    print("="*70)

    minutes = 10 if use_10min else 2
    seeds = [42, 137]

    # Part A: Blend ratio sweep
    print("\n  --- Part A: Blend Ratio Sweep ---")
    blend_methods = ["xavier_ca5", "xavier_ca10", "xavier_ca15", "xavier_ca20", "xavier_ca30"]
    blend_results = {}

    for method in blend_methods:
        bpbs = []
        for seed in seeds:
            label = f"Track2A: {method} @ {minutes}min, seed={seed}"
            result = run_experiment(method, minutes, seed, label)
            bpbs.append(result["val_bpb"])
            blend_results[(method, seed)] = result
        mean_bpb = sum(bpbs) / len(bpbs)
        print(f"  {method}: mean val_bpb = {mean_bpb:.4f}")

    # Find best blend
    blend_means = {}
    for method in blend_methods:
        bpbs = [blend_results[(method, s)]["val_bpb"] for s in seeds]
        blend_means[method] = sum(bpbs) / len(bpbs)
    best_blend = min(blend_means, key=blend_means.get)
    best_blend_pct = best_blend.replace("xavier_ca", "")

    print(f"\n  Best blend ratio: {best_blend_pct}% (val_bpb: {blend_means[best_blend]:.4f})")

    # Part B: CA pattern type comparison at best blend
    print("\n  --- Part B: CA Pattern Type Comparison ---")
    pattern_methods = ["xavier_grid_ca", "xavier_rd_spots", "xavier_rd_stripes",
                       "xavier_block_ca", "xavier_spectral_ca"]
    pattern_results = {}

    for method in pattern_methods:
        bpbs = []
        for seed in seeds:
            label = f"Track2B: {method} @ {minutes}min, seed={seed}"
            result = run_experiment(method, minutes, seed, label)
            bpbs.append(result["val_bpb"])
            pattern_results[(method, seed)] = result
        mean_bpb = sum(bpbs) / len(bpbs)
        print(f"  {method}: mean val_bpb = {mean_bpb:.4f}")

    # Xavier baseline for comparison
    xavier_bpbs = []
    for seed in seeds:
        label = f"Track2B: xavier baseline @ {minutes}min, seed={seed}"
        result = run_experiment("xavier", minutes, seed, label)
        xavier_bpbs.append(result["val_bpb"])
        pattern_results[("xavier", seed)] = result
    xavier_mean = sum(xavier_bpbs) / len(xavier_bpbs)

    # Find best pattern
    pattern_means = {}
    for method in pattern_methods:
        bpbs = [pattern_results[(method, s)]["val_bpb"] for s in seeds]
        pattern_means[method] = sum(bpbs) / len(bpbs)
    best_pattern = min(pattern_means, key=pattern_means.get)

    # Print comparison table
    print(f"\n  === Blend Ratio Comparison ({minutes}min, 2 seeds) ===")
    print(f"  {'method':<18} {'mean val_bpb':<14} {'vs xavier'}")
    print("  " + "-" * 50)

    # Also include xavier in blend table for reference
    for method in ["xavier"] + blend_methods:
        if method == "xavier":
            mean = xavier_mean
        else:
            mean = blend_means[method]
        improvement = (xavier_mean - mean) / xavier_mean * 100
        print(f"  {method:<18} {mean:.4f}        {improvement:+.1f}%")

    print(f"\n  === CA Pattern Comparison ({minutes}min, best blend, 2 seeds) ===")
    print(f"  {'method':<22} {'mean val_bpb':<14} {'vs xavier'}")
    print("  " + "-" * 55)
    for method in ["xavier"] + pattern_methods:
        if method == "xavier":
            mean = xavier_mean
        else:
            mean = pattern_means[method]
        improvement = (xavier_mean - mean) / xavier_mean * 100
        print(f"  {method:<22} {mean:.4f}        {improvement:+.1f}%")

    overall_best_method = best_blend
    overall_best_bpb = blend_means[best_blend]
    if pattern_means[best_pattern] < overall_best_bpb:
        overall_best_method = best_pattern
        overall_best_bpb = pattern_means[best_pattern]

    print(f"\n  Overall best init: {overall_best_method} (val_bpb: {overall_best_bpb:.4f})")

    return {
        "blend_means": blend_means,
        "pattern_means": pattern_means,
        "xavier_mean": xavier_mean,
        "best_blend": best_blend,
        "best_pattern": best_pattern,
        "overall_best": overall_best_method,
        "overall_best_bpb": overall_best_bpb,
        "minutes": minutes,
    }


def track3_output_quality(track1_results: dict):
    """Track 3: Compare generation quality between xavier and best CA."""
    print("\n" + "="*70)
    print("  TRACK 3: OUTPUT QUALITY COMPARISON")
    print("="*70)

    # Use the 30-minute models from Track 1 (seed 42 as representative)
    # Re-train since we don't persist models across tracks
    models = {}
    for method in ["xavier", "xavier_ca10"]:
        label = f"Track3: {method} @ 30min, seed=42 (for quality comparison)"
        result = run_experiment(method, 30, 42, label)
        models[method] = result["_model"]

    # Generate samples
    all_samples = {}
    all_quality = {}

    for method, model in models.items():
        samples = []
        for prompt_text in PROMPTS:
            prompt_bytes = prompt_text.encode("utf-8")
            for _ in range(3):  # 3 samples per prompt
                text = generate(model, prompt_bytes, max_tokens=100,
                               temperature=0.8)
                # Extract just the generated part (after prompt)
                generated = text[len(prompt_text):]
                samples.append({"prompt": prompt_text, "generated": generated,
                                "full": text})
        all_samples[method] = samples
        all_quality[method] = compute_text_quality([s["generated"] for s in samples])

    # Print quality comparison
    print(f"\n  === Quality Comparison (30 samples each) ===")
    print(f"  {'metric':<28} {'xavier':<14} {'xavier_ca10':<14} {'diff'}")
    print("  " + "-" * 65)

    for metric in ["unique_token_ratio", "trigram_repetition_rate", "sentence_completion_rate"]:
        xval = all_quality["xavier"][metric]
        cval = all_quality["xavier_ca10"][metric]
        diff = cval - xval
        print(f"  {metric:<28} {xval:.4f}        {cval:.4f}        {diff:+.4f}")

    # Print side-by-side samples for 3 most different prompts
    print(f"\n  === Sample Comparison (3 most different prompts) ===")
    # Find prompts where outputs differ most (by edit distance approximation)
    prompt_diffs = []
    for i, prompt in enumerate(PROMPTS):
        xavier_texts = [s["generated"] for s in all_samples["xavier"] if s["prompt"] == prompt]
        ca_texts = [s["generated"] for s in all_samples["xavier_ca10"] if s["prompt"] == prompt]
        # Simple difference metric: token overlap
        xavier_tokens = set(" ".join(xavier_texts).split())
        ca_tokens = set(" ".join(ca_texts).split())
        overlap = len(xavier_tokens & ca_tokens) / max(len(xavier_tokens | ca_tokens), 1)
        prompt_diffs.append((1 - overlap, i, prompt))
    prompt_diffs.sort(reverse=True)

    for diff_score, idx, prompt in prompt_diffs[:3]:
        print(f"\n  Prompt: \"{prompt}\"")
        xavier_sample = [s for s in all_samples["xavier"] if s["prompt"] == prompt][0]
        ca_sample = [s for s in all_samples["xavier_ca10"] if s["prompt"] == prompt][0]
        print(f"  Xavier:     ...{xavier_sample['generated'][:120]}")
        print(f"  Xavier+CA:  ...{ca_sample['generated'][:120]}")

    # Determine qualitative difference
    qx = all_quality["xavier"]
    qc = all_quality["xavier_ca10"]
    rep_diff = abs(qx["trigram_repetition_rate"] - qc["trigram_repetition_rate"])
    uniq_diff = abs(qx["unique_token_ratio"] - qc["unique_token_ratio"])

    if rep_diff > 0.05 or uniq_diff > 0.05:
        qual_diff = "noticeable"
    elif rep_diff > 0.02 or uniq_diff > 0.02:
        qual_diff = "subtle"
    else:
        qual_diff = "none"

    print(f"\n  Qualitative difference: {qual_diff}")

    return {
        "quality": all_quality,
        "qualitative_difference": qual_diff,
    }


def write_results_tsv(track1_res, track2_res):
    """Append Round 2 results to results.tsv."""
    rows = []
    exp_num = 51  # continue from Round 1

    if track1_res:
        for row in track1_res["table"]:
            for method in ["xavier", "ca10"]:
                mean_key = f"{method}_mean"
                if method == "xavier":
                    mean = row["xavier_mean"]
                else:
                    mean = row["ca10_mean"]
                tag = f"r2_t1_{method}_{row['horizon_min']}min"
                rows.append(f"{exp_num}\t{tag}\t{mean:.4f}\t0\t{method}\tinit\t{row['improvement_pct']:+.1f}\t0\t"
                           f"Track1 horizon validation, {row['horizon_min']}min, 3 seeds")
                exp_num += 1

    if track2_res:
        for method, mean in sorted(track2_res.get("blend_means", {}).items()):
            tag = f"r2_t2_{method}"
            imp = (track2_res["xavier_mean"] - mean) / track2_res["xavier_mean"] * 100
            rows.append(f"{exp_num}\t{tag}\t{mean:.4f}\t0\t{method}\tinit\t{imp:+.1f}\t0\t"
                       f"Track2 blend sweep, {track2_res['minutes']}min, 2 seeds")
            exp_num += 1
        for method, mean in sorted(track2_res.get("pattern_means", {}).items()):
            tag = f"r2_t2_{method}"
            imp = (track2_res["xavier_mean"] - mean) / track2_res["xavier_mean"] * 100
            rows.append(f"{exp_num}\t{tag}\t{mean:.4f}\t0\t{method}\tinit\t{imp:+.1f}\t0\t"
                       f"Track2 pattern comparison, {track2_res['minutes']}min, 2 seeds")
            exp_num += 1

    if rows:
        with open("results.tsv", "a") as f:
            for row in rows:
                f.write(row + "\n")
        print(f"\nAppended {len(rows)} rows to results.tsv")


def write_summary(track1_res, track2_res, track3_res):
    """Write ROUND2_RESULTS.md summary."""
    lines = ["# Round 2 Results\n"]

    lines.append("## Track 1 — Horizon Validation\n")
    if track1_res:
        lines.append("| horizon | xavier (mean±std) | ca10 (mean±std) | improvement | trend |")
        lines.append("|---------|-------------------|-----------------|-------------|-------|")
        for row in track1_res["table"]:
            lines.append(
                f"| {row['horizon_min']} min | {row['xavier_mean']:.4f} ± {row['xavier_std']:.4f} | "
                f"{row['ca10_mean']:.4f} ± {row['ca10_std']:.4f} | "
                f"{row['improvement_pct']:+.1f}% | {row['trend']} |"
            )
        lines.append(f"\n**Verdict:** {track1_res['verdict']}\n")
    else:
        lines.append("Not run.\n")

    lines.append("## Track 2 — Blend & Pattern Optimization\n")
    if track2_res:
        lines.append(f"Best blend ratio: {track2_res['best_blend']} "
                     f"(val_bpb: {track2_res['blend_means'][track2_res['best_blend']]:.4f})")
        lines.append(f"\nBest CA pattern: {track2_res['best_pattern']} "
                     f"(val_bpb: {track2_res['pattern_means'][track2_res['best_pattern']]:.4f})")
        lines.append(f"\nOverall best init: {track2_res['overall_best']} "
                     f"(val_bpb: {track2_res['overall_best_bpb']:.4f}, "
                     f"improvement over xavier: "
                     f"{(track2_res['xavier_mean'] - track2_res['overall_best_bpb']) / track2_res['xavier_mean'] * 100:+.1f}%)\n")

        lines.append("### Blend Ratio Results\n")
        lines.append("| method | mean val_bpb | vs xavier |")
        lines.append("|--------|-------------|-----------|")
        for method in sorted(track2_res["blend_means"]):
            mean = track2_res["blend_means"][method]
            imp = (track2_res["xavier_mean"] - mean) / track2_res["xavier_mean"] * 100
            lines.append(f"| {method} | {mean:.4f} | {imp:+.1f}% |")

        lines.append("\n### CA Pattern Results\n")
        lines.append("| method | mean val_bpb | vs xavier |")
        lines.append("|--------|-------------|-----------|")
        for method in sorted(track2_res["pattern_means"]):
            mean = track2_res["pattern_means"][method]
            imp = (track2_res["xavier_mean"] - mean) / track2_res["xavier_mean"] * 100
            lines.append(f"| {method} | {mean:.4f} | {imp:+.1f}% |")
        lines.append("")
    else:
        lines.append("Not run.\n")

    lines.append("## Track 3 — Output Quality\n")
    if track3_res:
        lines.append("| metric | xavier | xavier_ca10 | diff |")
        lines.append("|--------|--------|-------------|------|")
        for metric in ["unique_token_ratio", "trigram_repetition_rate", "sentence_completion_rate"]:
            xval = track3_res["quality"]["xavier"][metric]
            cval = track3_res["quality"]["xavier_ca10"][metric]
            diff = cval - xval
            lines.append(f"| {metric} | {xval:.4f} | {cval:.4f} | {diff:+.4f} |")
        lines.append(f"\n**Qualitative difference:** {track3_res['qualitative_difference']}\n")
    else:
        lines.append("Not run.\n")

    lines.append("## Overall Conclusion\n")
    conclusion_parts = []
    if track1_res:
        conclusion_parts.append(f"Horizon validation ({track1_res['verdict'].lower()}).")
    if track2_res:
        conclusion_parts.append(
            f"Best init is {track2_res['overall_best']} "
            f"(val_bpb {track2_res['overall_best_bpb']:.4f})."
        )
    if track3_res:
        conclusion_parts.append(
            f"Qualitative text difference: {track3_res['qualitative_difference']}."
        )
    lines.append(" ".join(conclusion_parts) + "\n")

    with open("ROUND2_RESULTS.md", "w") as f:
        f.write("\n".join(lines))
    print("\nWrote ROUND2_RESULTS.md")


def main():
    parser = argparse.ArgumentParser(description="Round 2 experiment runner")
    parser.add_argument("--track", type=int, default=0,
                        help="Run specific track (1, 2, or 3). 0 = all tracks.")
    args = parser.parse_args()

    t_start = time.time()

    track1_res = None
    track2_res = None
    track3_res = None

    run_all = args.track == 0

    # Track 1: Horizon Validation (priority)
    if run_all or args.track == 1:
        track1_res = track1_horizon_validation()

        # If CA is harmful, skip remaining tracks
        if track1_res.get("ca_harmful"):
            print("\n*** CA init is HARMFUL at longer training. Skipping Tracks 2-3. ***")
            write_results_tsv(track1_res, None)
            write_summary(track1_res, None, None)
            elapsed = time.time() - t_start
            print(f"\nTotal wall time: {elapsed/60:.1f} min")
            return

    # Track 2: Blend & Pattern
    if run_all or args.track == 2:
        # Use 10-min budget if Track 1 showed CA is beneficial
        use_10min = True
        if track1_res:
            last_imp = track1_res["table"][-1]["improvement_pct"]
            if last_imp < 0.5:
                use_10min = False  # CA advantage vanishes, use 2-min
                print("\n  Track 1 showed CA advantage vanishes. Using 2-min budget for Track 2.")
        track2_res = track2_blend_and_pattern(use_10min=use_10min)

    # Track 3: Output Quality
    if run_all or args.track == 3:
        if track1_res and track1_res["table"][-1]["improvement_pct"] < 0.5:
            print("\n  Skipping Track 3: CA advantage too small for quality comparison.")
        else:
            track3_res = track3_output_quality(track1_res or {})

    # Write results
    write_results_tsv(track1_res, track2_res)
    write_summary(track1_res, track2_res, track3_res)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  Round 2 complete. Total wall time: {elapsed/60:.1f} min ({elapsed/3600:.1f} h)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
