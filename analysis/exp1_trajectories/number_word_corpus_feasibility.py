"""Corpus feasibility check for number-word compositional axis (Exp 3).

Exp 0 §13.3 suggested number words (one-ten) as the replacement for the
ruled-out digit-arithmetic axis. Before committing to the axis, check:

  1. Total frequency of each number word — is there enough data?
  2. Next-word distribution for each — are occurrences formulaic (few dominant
     followers like "little pigs") or compositional (many distinct nouns)?
  3. Top noun follower distribution — is there overlap across numbers (so
     "three apples" and "four apples" are both plausible held-out combinations)?
  4. Estimate of compositional diversity: entropy of next-word distribution,
     and count of distinct next-words seen with ≥5 occurrences.
  5. Spot-check random contexts for each number.

If most occurrences of most numbers concentrate on one or two followers
("three little pigs"), the compositional test is thin. If followers are
diverse across a common noun set, the test is viable.

Output: stdout + analysis/exp1_trajectories/number_word_feasibility.json
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO_ROOT / "analysis" / "exp1_trajectories"
DATA_DIR = Path.home() / ".cache" / "neurogen"

NUMBERS = ["one", "two", "three", "four", "five",
           "six", "seven", "eight", "nine", "ten"]

# Stopwords we skip when hunting for the compositional noun after a number.
# We want to count e.g. "three of the little pigs" as (three, pigs), not
# (three, of).
SKIPWORDS = {
    "of", "the", "a", "an", "and", "or", "but", "to", "at", "in", "on",
    "for", "by", "with", "from", "big", "little", "small", "tiny", "huge",
    "very", "more", "other", "own", "these", "those", "this", "that",
    "some", "many", "few", "all", "any", "such", "as", "is", "was", "were",
    "are", "be", "been", "being",
}


def load_train_text(max_bytes: int = 200_000_000) -> str:
    shards = sorted(DATA_DIR.glob("train_*.bin"))
    arrs, total = [], 0
    for p in shards:
        arr = np.fromfile(p, dtype=np.uint8)
        if total + len(arr) > max_bytes:
            arrs.append(arr[: max_bytes - total])
            break
        arrs.append(arr)
        total += len(arr)
        if total >= max_bytes:
            break
    return bytes(np.concatenate(arrs)).decode("utf-8", errors="ignore").lower()


def next_word_after(text: str, word: str, skip_stop: bool = True) -> Counter:
    """For each occurrence of `word` (whole-word), return the next content word.
    If skip_stop, skip stopwords between the number and the noun."""
    c = Counter()
    pattern = re.compile(rf"\b{word}\b")
    for m in pattern.finditer(text):
        tail = text[m.end(): m.end() + 60]
        tokens = re.findall(r"[a-z']+", tail)
        if skip_stop:
            for t in tokens[:6]:
                if t not in SKIPWORDS:
                    c[t] += 1
                    break
        else:
            if tokens:
                c[tokens[0]] += 1
    return c


def main() -> None:
    print("Loading ~200M bytes of training text…")
    text = load_train_text()
    N = len(text)
    print(f"Loaded {N:,} characters\n")

    summary = {}
    print(f"{'number':<8} {'count':>8} {'distinct':>10} "
          f"{'top5_frac':>10} {'ent(bits)':>10}  top-3 followers")
    print("-" * 90)

    for w in NUMBERS:
        pattern = re.compile(rf"\b{w}\b")
        n = len(pattern.findall(text))
        nc = next_word_after(text, w)
        total_followers = sum(nc.values())
        top3 = nc.most_common(3)
        top5 = nc.most_common(5)
        top5_frac = (sum(c for _, c in top5) / total_followers) if total_followers else 0.0
        # Entropy in bits
        if total_followers > 0:
            probs = np.array([c / total_followers for _, c in nc.most_common()])
            entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
        else:
            entropy = 0.0
        distinct_5plus = sum(1 for _, c in nc.items() if c >= 5)
        top3_str = "  ".join(f"{w2}({c})" for w2, c in top3)

        summary[w] = {
            "count": n,
            "distinct_followers": len(nc),
            "distinct_followers_5plus": distinct_5plus,
            "top5_frac": top5_frac,
            "next_word_entropy_bits": entropy,
            "top_followers": nc.most_common(20),
        }
        print(f"{w:<8} {n:>8d} {len(nc):>10d} "
              f"{top5_frac:>10.3f} {entropy:>10.3f}  {top3_str}")

    # Cross-number noun overlap: find nouns that follow at least 3 different
    # numbers. These are compositional slot candidates.
    noun_sets = {}
    for w in NUMBERS:
        cnt = dict(summary[w]["top_followers"])
        noun_sets[w] = {noun for noun, c in cnt.items() if c >= 3}

    # Union across numbers
    all_nouns = Counter()
    for w in NUMBERS:
        for noun in noun_sets[w]:
            all_nouns[noun] += 1
    shared_nouns = [(n, k) for n, k in all_nouns.most_common(40) if k >= 3]
    print()
    print(f"=== Nouns following ≥3 different numbers (candidate compositional slots) ===")
    for noun, k in shared_nouns:
        # which numbers did this noun follow?
        who = [w for w in NUMBERS if noun in noun_sets[w]]
        print(f"  {noun:<15}  follows {k} different numbers: {who}")

    summary["shared_nouns_3plus"] = shared_nouns

    # Compositional-cell coverage: for a sample of compositional slots,
    # count how many of the 10×|slots| number×noun pairs actually occur
    candidate_nouns = [n for n, _ in shared_nouns[:20]]
    coverage = {}
    for noun in candidate_nouns:
        row = {}
        for w in NUMBERS:
            # count exact "word noun" occurrences (with any intermediate stopwords)
            # simple: count followers where noun is the content word
            cnt = dict(summary[w]["top_followers"])
            row[w] = int(cnt.get(noun, 0))
        coverage[noun] = row
    summary["coverage_table"] = coverage

    # Print coverage matrix
    print()
    print(f"=== Number × noun coverage (counts in 200M sample) ===")
    header = f"{'noun':<14} " + " ".join(f"{w:>5}" for w in NUMBERS)
    print(header)
    print("-" * len(header))
    for noun in candidate_nouns:
        row = coverage[noun]
        non_zero = sum(1 for v in row.values() if v > 0)
        print(f"{noun:<14} " + " ".join(f"{row[w]:>5d}" for w in NUMBERS) +
              f"  ({non_zero}/10 cells non-zero)")

    # Feasibility verdict
    print()
    print("=== Feasibility verdict ===")
    total_occur = sum(summary[w]["count"] for w in NUMBERS)
    print(f"Total number-word occurrences: {total_occur:,}")
    mean_entropy = np.mean([summary[w]["next_word_entropy_bits"] for w in NUMBERS])
    print(f"Mean next-word entropy across numbers: {mean_entropy:.2f} bits")
    print(f"Mean top-5 concentration: "
          f"{np.mean([summary[w]['top5_frac'] for w in NUMBERS]):.3f}")
    n_candidate_slots = len(shared_nouns)
    print(f"Candidate compositional slot nouns (≥3 numbers): {n_candidate_slots}")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "number_word_feasibility.json").write_text(json.dumps(summary, indent=2))
    print(f"\nStats → {ANALYSIS_DIR/'number_word_feasibility.json'}")


if __name__ == "__main__":
    main()
