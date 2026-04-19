"""Corpus digit-frequency + context check.

Question from Exp 0 report §13 refinement: is the digit-arithmetic axis
feasible for Exp 3? The +0.60 digit cluster could reflect arithmetic
structure, or merely that digits appear in similar contexts (ages, counts,
quantities).

Checks:
  1. Frequency of each digit byte (48-57) in training data.
  2. Ratio of digit-bytes to number-word tokens (three, four, five, ...).
     Tells us whether TinyStories prefers digits or number words.
  3. Bigram contexts of digits: what precedes and follows each digit? If
     digits appear overwhelmingly in one context (ages, counts of objects),
     the cluster reflects context not composition.
  4. Arithmetic operators: count '+', '-', '=' with digit neighbors. If
     these are near-zero, arithmetic composition cannot be in the corpus
     and the +0.60 digit cluster must be something else.

Output: stdout + analysis/exp1_trajectories/digit_corpus_frequency.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO_ROOT / "analysis" / "exp1_trajectories"
DATA_DIR = Path.home() / ".cache" / "neurogen"

DIGIT_BYTES = list(range(ord("0"), ord("9") + 1))
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five",
                "six", "seven", "eight", "nine", "ten"]


def load_train_bytes(max_bytes: int | None = None) -> np.ndarray:
    shards = sorted(DATA_DIR.glob("train_*.bin"))
    if not shards:
        raise FileNotFoundError(DATA_DIR)
    arrs = []
    total = 0
    for p in shards:
        arr = np.fromfile(p, dtype=np.uint8)
        if max_bytes is not None and total + len(arr) > max_bytes:
            arrs.append(arr[: max_bytes - total])
            break
        arrs.append(arr)
        total += len(arr)
        if max_bytes is not None and total >= max_bytes:
            break
    return np.concatenate(arrs)


def main() -> None:
    # Use a 200M-byte sample for speed; full corpus is ~1.9B bytes. 200M is
    # ~10% of the training data — plenty for frequency estimates.
    print("Loading training bytes (up to 200M)…")
    data = load_train_bytes(max_bytes=200_000_000)
    N = len(data)
    print(f"loaded {N:,} bytes\n")

    # --- 1. Digit byte frequencies ----------------------------------------
    counts = Counter(int(b) for b in data if b in set(DIGIT_BYTES))
    total_digits = sum(counts.values())
    print(f"=== Digit byte counts (out of {N:,} total bytes) ===")
    for d in DIGIT_BYTES:
        c = counts.get(d, 0)
        print(f"  '{chr(d)}'  (byte {d:>3d}):  {c:>8d}  ({100*c/N:.4f}% of corpus)")
    print(f"  total digits: {total_digits:,}  ({100*total_digits/N:.4f}% of corpus)")

    # Compare against common letters
    for ch in "aeotn":
        b = ord(ch)
        c = int((data == b).sum())
        print(f"  '{ch}' reference:   byte {b:>3d}:  {c:>8d}  ({100*c/N:.4f}%)")

    # --- 2. Number words vs digits ----------------------------------------
    # Decode as UTF-8 (actually ASCII in TinyStories) and search for exact
    # number word with word boundaries.
    text = bytes(data).decode("utf-8", errors="ignore").lower()
    print(f"\n=== Number words vs digits ===")
    word_counts = {}
    for w in NUMBER_WORDS:
        # cheap whole-word match: preceded by space/BOL and followed by non-letter
        # Not perfect but good enough for order-of-magnitude
        key = f" {w} "
        c = text.count(key)
        word_counts[w] = c
        print(f"  '{w:<6}' : {c:>8d}")
    digit_vs_words = {
        "total_digit_bytes": total_digits,
        "total_number_words_01_10": sum(word_counts.values()),
    }
    print(f"\n  total digit bytes: {total_digits:,}")
    print(f"  total number-word matches (0-10): {sum(word_counts.values()):,}")
    print(f"  ratio digit:word = {total_digits/max(sum(word_counts.values()),1):.2f}")

    # --- 3. Bigram contexts of digits -------------------------------------
    print(f"\n=== Digit byte bigram neighborhoods (what precedes each digit) ===")
    # For each digit d, sample a few preceding bytes and tally
    ctx = {}
    for d in DIGIT_BYTES:
        idxs = np.where(data == d)[0]
        idxs = idxs[(idxs > 0) & (idxs < N - 1)]
        if len(idxs) == 0:
            ctx[d] = {"prev": [], "next": []}
            continue
        prev_bytes = data[idxs - 1]
        next_bytes = data[idxs + 1]
        prev_c = Counter(int(b) for b in prev_bytes)
        next_c = Counter(int(b) for b in next_bytes)
        top_prev = prev_c.most_common(5)
        top_next = next_c.most_common(5)
        ctx[d] = {"prev": top_prev, "next": top_next}
        print(f"  '{chr(d)}' (n={len(idxs)}):")
        print(f"     prev: " + ", ".join(
            f"{repr(chr(b)) if 32 <= b < 127 else f'0x{b:02x}'}:{c}"
            for b, c in top_prev))
        print(f"     next: " + ", ".join(
            f"{repr(chr(b)) if 32 <= b < 127 else f'0x{b:02x}'}:{c}"
            for b, c in top_next))

    # --- 4. Arithmetic operator usage -------------------------------------
    print(f"\n=== Arithmetic operators in corpus ===")
    # Patterns: 'N + M', 'N - M', 'N = M', also 'N and M' (additive context)
    patterns = {
        "'digit + digit'": [(data[i-2:i+3] == np.array([d1, ord(' '), ord('+'), ord(' '), d2], dtype=np.uint8)).all()
                            for i in []],
    }
    # Simpler: count pair occurrences of ' + ', ' - ', ' = ', ' x '
    ops = [" + ", " - ", " = ", " x ", " * ", " / "]
    for op in ops:
        print(f"  '{op}' : {text.count(op):>6d}")
    # And: digit-space-plus-space-digit sequences
    import re
    for op_rx in [r"\d\s*\+\s*\d", r"\d\s*-\s*\d", r"\d\s*=\s*\d",
                  r"\d\s*x\s*\d", r"\d\s*\*\s*\d"]:
        matches = re.findall(op_rx, text)
        print(f"  regex {op_rx!r}: {len(matches):>6d}  "
              f"(examples: {matches[:3]})")

    # --- 5. Sample digit contexts (windowed) ------------------------------
    print(f"\n=== Sample digit contexts (30 random windows per digit '3') ===")
    d = ord("3")
    idxs = np.where(data == d)[0]
    idxs = idxs[(idxs > 20) & (idxs < N - 20)]
    rng = np.random.default_rng(0)
    sample = rng.choice(idxs, size=min(30, len(idxs)), replace=False)
    for i in sorted(sample):
        window = bytes(data[i - 10 : i + 10]).decode("utf-8", errors="replace")
        print(f"   ...{window!r}...")

    # Save
    out = {
        "digit_counts": {chr(d): counts.get(d, 0) for d in DIGIT_BYTES},
        "total_digit_bytes": int(total_digits),
        "number_word_counts": word_counts,
        "bigram_contexts": {chr(d): ctx[d] for d in DIGIT_BYTES},
        "arithmetic_operator_counts": {op: text.count(op) for op in ops},
        "scanned_bytes": int(N),
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "digit_corpus_frequency.json").write_text(json.dumps(out, indent=2))
    print(f"\nStats → {ANALYSIS_DIR/'digit_corpus_frequency.json'}")


if __name__ == "__main__":
    main()
