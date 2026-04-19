"""Interpretation diagnostics: distinguish regime A (genuinely high-D
structure), B (local structure only), or C (noise) for the Exp 0 displacement
finding (flat spectrum, uniform magnitudes).

Two checks:

1. Pairwise cosine similarity — within-class byte groups vs random pairs, on
   both final embeddings and on displacement vectors. If within-class
   similarity is systematically higher than random, we're in A or B; if not,
   we're in C.

2. Nearest-neighbor coherence — for ~20 anchor bytes, print the 10 nearest
   neighbors in final embedding space (cosine). Human eyeballs the semantic
   coherence.

Byte-level vocab (V=256) means "expected-similar" groups are orthographic,
not lexical: lowercase vs uppercase letters, digits, whitespace,
punctuation, etc. If the model has learned anything, these groups should
cluster.

Outputs:
  - stdout: tables, nearest neighbors
  - reports/exp0/interpretation_stats.json
"""

from __future__ import annotations

import json
import string
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "runs" / "exp0_baseline"
OUT_DIR = REPO_ROOT / "reports" / "exp0"


# ---------------------------------------------------------------------------
# Byte classes (on the 0-255 byte vocabulary)
# ---------------------------------------------------------------------------
def byte_classes() -> dict[str, list[int]]:
    lowercase = [ord(c) for c in string.ascii_lowercase]
    uppercase = [ord(c) for c in string.ascii_uppercase]
    digits = [ord(c) for c in string.digits]
    whitespace = [ord(c) for c in " \t\n\r"]
    sentence_punct = [ord(c) for c in ".!?"]
    clause_punct = [ord(c) for c in ",;:"]
    quotes = [ord(c) for c in "\"'"]
    brackets = [ord(c) for c in "()[]{}"]
    # Vowels / consonants — finer partition of lowercase
    vowels = [ord(c) for c in "aeiou"]
    consonants = [ord(c) for c in "bcdfghjklmnpqrstvwxyz"]
    return {
        "lowercase":      lowercase,
        "uppercase":      uppercase,
        "digits":         digits,
        "whitespace":     whitespace,
        "sentence_punct": sentence_punct,
        "clause_punct":   clause_punct,
        "quotes":         quotes,
        "brackets":       brackets,
        "vowels":         vowels,
        "consonants":     consonants,
    }


def byte_repr(b: int) -> str:
    """Human-readable display of a byte."""
    if 32 <= b < 127:
        return repr(chr(b))          # e.g. 'a', "'", '"'
    return f"\\x{b:02x}"              # e.g. \x0a


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------
def cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity matrix between rows of A and rows of B."""
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return An @ Bn.T


def pair_mean_cos(vecs: np.ndarray, idxs: list[int]) -> float | None:
    """Mean pairwise cosine similarity among rows indexed by idxs (distinct pairs)."""
    if len(idxs) < 2:
        return None
    M = cosine(vecs[idxs], vecs[idxs])
    iu = np.triu_indices(len(idxs), k=1)
    return float(M[iu].mean())


def random_pair_mean_cos(vecs: np.ndarray, n_pairs: int, rng: np.random.Generator) -> float:
    """Mean cosine over n_pairs uniformly sampled distinct pairs."""
    V = vecs.shape[0]
    i = rng.integers(0, V, size=n_pairs)
    j = rng.integers(0, V, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    sims = (cosine(vecs[i], vecs[j])).diagonal()
    return float(sims.mean())


def cross_class_mean_cos(vecs: np.ndarray, A: list[int], B: list[int]) -> float:
    """Mean cosine between rows in A and rows in B (across groups)."""
    M = cosine(vecs[A], vecs[B])
    return float(M.mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    snaps = sorted((RUN_DIR / "snapshots").glob("wte_step_*.npy"))
    w0 = np.load(snaps[0])
    wL = np.load(snaps[-1])
    disp = wL - w0

    classes = byte_classes()
    rng = np.random.default_rng(42)

    # ----- Diagnostic 1: within-class vs random, for final embeds and disps --
    random_n = 500   # large enough to stabilize mean
    rand_cos_wL = random_pair_mean_cos(wL, random_n, rng)
    rand_cos_disp = random_pair_mean_cos(disp, random_n, rng)

    rows = []
    for name, idxs in classes.items():
        wl_mean = pair_mean_cos(wL, idxs)
        d_mean = pair_mean_cos(disp, idxs)
        rows.append(
            {
                "class": name,
                "n_tokens": len(idxs),
                "wL_within": wl_mean,
                "wL_lift_vs_random": (wl_mean - rand_cos_wL) if wl_mean is not None else None,
                "disp_within": d_mean,
                "disp_lift_vs_random": (d_mean - rand_cos_disp) if d_mean is not None else None,
            }
        )

    # Cross-class similarity: within vs between for the two biggest classes
    cross_pairs = [
        ("lowercase", "uppercase"),
        ("lowercase", "digits"),
        ("digits", "whitespace"),
        ("sentence_punct", "clause_punct"),
        ("vowels", "consonants"),
    ]
    cross_rows = []
    for a, b in cross_pairs:
        cross_rows.append(
            {
                "a": a,
                "b": b,
                "wL_cross": cross_class_mean_cos(wL, classes[a], classes[b]),
                "disp_cross": cross_class_mean_cos(disp, classes[a], classes[b]),
            }
        )

    # Print diagnostic 1
    print("=" * 78)
    print("DIAGNOSTIC 1 — pairwise cosine similarity")
    print("=" * 78)
    print(f"Random-pair baseline (n={random_n}):")
    print(f"  final embedding cosine : {rand_cos_wL:+.4f}")
    print(f"  displacement cosine    : {rand_cos_disp:+.4f}")
    print()
    print(f"{'class':<18} {'n':>4}  "
          f"{'wL_within':>10} {'lift':>8}   {'disp_within':>12} {'lift':>8}")
    print("-" * 78)
    for r in rows:
        wl_w = f"{r['wL_within']:+.4f}" if r["wL_within"] is not None else "  n/a  "
        wl_l = f"{r['wL_lift_vs_random']:+.4f}" if r["wL_lift_vs_random"] is not None else "  n/a  "
        d_w = f"{r['disp_within']:+.4f}" if r["disp_within"] is not None else "  n/a  "
        d_l = f"{r['disp_lift_vs_random']:+.4f}" if r["disp_lift_vs_random"] is not None else "  n/a  "
        print(f"{r['class']:<18} {r['n_tokens']:>4}  {wl_w:>10} {wl_l:>8}   {d_w:>12} {d_l:>8}")

    print()
    print(f"{'pair':<34}  {'wL_cross':>10}  {'disp_cross':>12}")
    print("-" * 78)
    for cr in cross_rows:
        print(f"{cr['a'] + ' × ' + cr['b']:<34}  {cr['wL_cross']:+.4f}     {cr['disp_cross']:+.4f}")

    # ----- Diagnostic 2: nearest-neighbor coherence -------------------------
    print()
    print("=" * 78)
    print("DIAGNOSTIC 2 — k=10 nearest neighbors in final embedding space (cosine)")
    print("=" * 78)

    anchors = [
        ord("a"), ord("e"), ord("t"), ord("s"), ord("m"),       # common lowercase
        ord("A"), ord("T"), ord("S"),                            # uppercase
        ord("0"), ord("1"), ord("5"),                            # digits
        ord(" "), ord("\n"),                                     # whitespace
        ord("."), ord(","), ord("!"), ord("?"),                  # punctuation
        ord('"'), ord("'"), ord("("),                            # quotes / brackets
    ]

    wL_norm = wL / (np.linalg.norm(wL, axis=1, keepdims=True) + 1e-12)
    cos_full = wL_norm @ wL_norm.T
    np.fill_diagonal(cos_full, -np.inf)  # exclude self

    nn_dump = []
    for a in anchors:
        nn_idx = np.argsort(-cos_full[a])[:10]
        nn_sims = cos_full[a, nn_idx]
        nn_chars = [byte_repr(int(j)) for j in nn_idx]
        print(f"  {byte_repr(a):<6}  →  " +
              "  ".join(f"{c}({s:+.2f})" for c, s in zip(nn_chars, nn_sims)))
        nn_dump.append(
            {
                "anchor": a,
                "anchor_char": byte_repr(a),
                "neighbors": [{"byte": int(j), "char": nn_chars[k], "cos": float(nn_sims[k])}
                              for k, j in enumerate(nn_idx)],
            }
        )

    # ----- Save stats -------------------------------------------------------
    out = {
        "random_baseline": {
            "final_embed_cos": rand_cos_wL,
            "displacement_cos": rand_cos_disp,
            "n_pairs": random_n,
        },
        "within_class": rows,
        "cross_class": cross_rows,
        "nearest_neighbors": nn_dump,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "interpretation_stats.json").write_text(json.dumps(out, indent=2))
    print()
    print(f"Stats written to {OUT_DIR/'interpretation_stats.json'}")


if __name__ == "__main__":
    main()
