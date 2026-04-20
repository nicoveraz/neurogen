"""Anchor-driven convergence verification on additional triads.

The `.!?` case study (Exp 1 §2) established a specific trajectory dynamic:
one member (the "anchor") reaches its final-centroid cosine early, and the
other two migrate toward it later. Anchor identified by high initial-cosine
with the final-centroid; correlates with corpus frequency.

This script:
  1. Finds candidate triads in w_final by searching for triples (i, j, k)
     where all three pairwise cosines exceed a threshold (tight mutual
     cluster).
  2. For each candidate, tracks the three tokens' cosine with the
     final-triad centroid across training snapshots.
  3. Identifies the anchor: member with highest init-centroid cosine.
  4. Reports step at which each member reaches centroid cosine ≥ 0.6.
  5. Correlates anchor-ness (init cosine rank) with corpus frequency.

Writes anchor_verification.json with per-triad trajectories and a
summary table.
"""

from __future__ import annotations

import json
import string
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "runs" / "exp0_baseline"
ANALYSIS_DIR = REPO_ROOT / "analysis" / "exp1_trajectories"
DATA_DIR = Path.home() / ".cache" / "neurogen"


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float((a @ b) / (na * nb))


def byte_label(b: int) -> str:
    if 32 <= b < 127 and chr(b) not in " \t\n\r":
        return repr(chr(b))
    if b == ord(" "):
        return "' '"
    if b == ord("\n"):
        return "'\\n'"
    return f"\\x{b:02x}"


def load_byte_frequencies(n_bytes: int = 200_000_000) -> np.ndarray:
    """Raw byte frequencies from a sample of training data."""
    shards = sorted(DATA_DIR.glob("train_*.bin"))
    arrs = []
    total = 0
    for p in shards:
        arr = np.fromfile(p, dtype=np.uint8)
        if total + len(arr) > n_bytes:
            arrs.append(arr[: n_bytes - total])
            break
        arrs.append(arr)
        total += len(arr)
        if total >= n_bytes:
            break
    data = np.concatenate(arrs)
    counts = np.bincount(data, minlength=256)
    return counts.astype(np.int64)


def find_candidate_triads(
    W_final: np.ndarray,
    n_candidates: int = 20,
    min_triad_cos: float = 0.30,
) -> list[tuple[int, int, int, float]]:
    """Find triads (i,j,k) where all three pairwise cosines ≥ min_triad_cos.
    Returns list of (i, j, k, min_pairwise_cos) sorted by min_pairwise_cos
    descending. De-duplicated by canonical sort."""
    Wn = W_final / (np.linalg.norm(W_final, axis=1, keepdims=True) + 1e-12)
    C = Wn @ Wn.T
    np.fill_diagonal(C, -np.inf)

    V = W_final.shape[0]
    # Only consider bytes that are "common" (skip high-bytes to avoid noise)
    common = list(range(32, 127))  # printable ASCII
    triads: list[tuple[int, int, int, float]] = []

    for i_idx, i in enumerate(common):
        for j_idx in range(i_idx + 1, len(common)):
            j = common[j_idx]
            if C[i, j] < min_triad_cos:
                continue
            # For each (i, j), find k with good cos to both
            for k_idx in range(j_idx + 1, len(common)):
                k = common[k_idx]
                if C[i, k] < min_triad_cos or C[j, k] < min_triad_cos:
                    continue
                min_c = min(C[i, j], C[i, k], C[j, k])
                triads.append((i, j, k, float(min_c)))

    triads.sort(key=lambda t: -t[3])
    # Skip the `.!?` triad which is already reported (step 3000/12000/16000)
    known = {(ord("."), ord("!"), ord("?"))}
    canonical = [t for t in triads if tuple(sorted(t[:3])) not in {tuple(sorted(k)) for k in known}]
    return canonical[:n_candidates]


def analyze_triad(
    i: int, j: int, k: int,
    Ws: list[np.ndarray],
    steps: list[int],
    byte_freqs: np.ndarray,
) -> dict:
    """Trajectory analysis for a single triad."""
    W_final = Ws[-1]
    centroid = W_final[[i, j, k]].mean(axis=0)

    tokens = [i, j, k]
    labels = {t: byte_label(t) for t in tokens}

    # Centroid cosine per token per snapshot
    centroid_series = {labels[t]: [] for t in tokens}
    for W in Ws:
        for t in tokens:
            centroid_series[labels[t]].append(cos(W[t], centroid))

    # Anchor identification: init cosine with final centroid (i.e. series[0])
    init_cos = {t: centroid_series[labels[t]][0] for t in tokens}
    ordered = sorted(tokens, key=lambda t: -init_cos[t])
    anchor = ordered[0]

    # Step at which each reaches centroid cos ≥ 0.6
    threshold = 0.6
    threshold_step = {}
    for t in tokens:
        series = np.asarray(centroid_series[labels[t]])
        hit = np.argmax(series >= threshold) if (series >= threshold).any() else -1
        threshold_step[labels[t]] = steps[hit] if hit >= 0 else None

    # Final pairwise cosines
    final_pairs = {
        f"{labels[tokens[0]]}-{labels[tokens[1]]}": cos(W_final[tokens[0]], W_final[tokens[1]]),
        f"{labels[tokens[0]]}-{labels[tokens[2]]}": cos(W_final[tokens[0]], W_final[tokens[2]]),
        f"{labels[tokens[1]]}-{labels[tokens[2]]}": cos(W_final[tokens[1]], W_final[tokens[2]]),
    }

    return {
        "tokens": [int(t) for t in tokens],
        "labels": labels,
        "frequencies": {labels[t]: int(byte_freqs[t]) for t in tokens},
        "init_cos_with_final_centroid": {labels[t]: round(v, 4) for t, v in init_cos.items()},
        "predicted_anchor": labels[anchor],
        "anchor_frequency_rank": [(labels[t], int(byte_freqs[t])) for t in ordered],
        "threshold_step_0.6": threshold_step,
        "final_pairwise_cosines": {k: round(v, 4) for k, v in final_pairs.items()},
        "centroid_series": centroid_series,
    }


def main() -> None:
    snaps = sorted((RUN_DIR / "snapshots").glob("wte_step_*.npy"))
    steps = [int(p.stem.split("_")[-1]) for p in snaps]
    print(f"Loading {len(snaps)} snapshots...")
    Ws = [np.load(p) for p in snaps]

    print("Loading byte frequencies from training corpus sample...")
    byte_freqs = load_byte_frequencies(n_bytes=200_000_000)

    print("Finding candidate triads in final embeddings...")
    candidates = find_candidate_triads(Ws[-1], n_candidates=20, min_triad_cos=0.30)
    print(f"Found {len(candidates)} candidate triads (printable ASCII only)")
    print(f"\n{'triad':<14}  {'min_pair_cos':>14}")
    print("-" * 30)
    for i, j, k, mc in candidates[:10]:
        label = f"{byte_label(i)}+{byte_label(j)}+{byte_label(k)}"
        print(f"{label:<14}  {mc:>14.4f}")

    # Pick the top 5 for analysis
    analyses = []
    print(f"\n=== Detailed analysis of top 5 triads ===\n")
    for i, j, k, mc in candidates[:5]:
        result = analyze_triad(i, j, k, Ws, steps, byte_freqs)
        analyses.append(result)

        label_str = "+".join(result["labels"][t] for t in [i, j, k])
        anchor = result["predicted_anchor"]
        print(f"Triad {label_str}  (min_pair_cos={mc:.3f})")
        print(f"  Byte frequencies: "
              + ", ".join(f"{lab}:{f:,}" for lab, f in result["anchor_frequency_rank"]))
        print(f"  Predicted anchor (highest init-centroid cos): {anchor}")
        print(f"  Init cosines with final centroid: "
              + ", ".join(f"{lab}:{v:+.3f}"
                          for lab, v in result["init_cos_with_final_centroid"].items()))
        print(f"  Step each crosses 0.6 centroid cos: "
              + ", ".join(f"{lab}:{s}"
                          for lab, s in result["threshold_step_0.6"].items()))
        # Validation: is anchor also the most-frequent member?
        freq_sorted = sorted(
            result["anchor_frequency_rank"],
            key=lambda x: -x[1],
        )
        most_frequent = freq_sorted[0][0]
        match = "✓ match" if anchor == most_frequent else "✗ mismatch"
        print(f"  Most-frequent member: {most_frequent}  →  anchor-vs-frequency: {match}")
        # Time-spread: how separated are the threshold crossings?
        thresh_steps = [v for v in result["threshold_step_0.6"].values() if v is not None]
        if len(thresh_steps) == 3:
            first, last = min(thresh_steps), max(thresh_steps)
            print(f"  Time spread: first@{first}, last@{last}, "
                  f"ratio={last/max(first,1):.1f}x")
        print()

    # Save
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "anchor_verification.json").write_text(
        json.dumps({"steps": steps, "triads": analyses}, indent=2)
    )
    print(f"Saved → {ANALYSIS_DIR/'anchor_verification.json'}")

    # Summary table
    print(f"\n=== Summary: anchor-vs-frequency pattern across triads ===\n")
    n_match = 0
    n_sequential = 0
    for a in analyses:
        tokens_labels = list(a["labels"].values())
        ts_steps = [v for v in a["threshold_step_0.6"].values() if v is not None]
        if len(ts_steps) < 3:
            seq_note = "incomplete (not all crossed 0.6)"
        else:
            seq_note = (
                "sequential" if (max(ts_steps) / max(min(ts_steps), 1)) >= 2
                else "simultaneous"
            )
            if max(ts_steps) / max(min(ts_steps), 1) >= 2:
                n_sequential += 1
        freq_sorted = sorted(a["anchor_frequency_rank"], key=lambda x: -x[1])
        most_freq = freq_sorted[0][0]
        if a["predicted_anchor"] == most_freq:
            n_match += 1
        print(f"  {'+'.join(tokens_labels):<12}  anchor-is-most-frequent: "
              f"{'yes' if a['predicted_anchor']==most_freq else 'no':<4}  "
              f"dynamics: {seq_note}")
    print(f"\n  {n_match} / {len(analyses)} triads: anchor-ness predicts most-frequent member")
    print(f"  {n_sequential} / {len(analyses)} triads: sequential (≥2× time spread)")


if __name__ == "__main__":
    main()
