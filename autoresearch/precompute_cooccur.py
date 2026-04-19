"""Precompute byte-level co-occurrence matrix for Exp 2 topographic regularizer.

Window = 5 (bytes within 5 positions of each other). Symmetric, self-pairs
excluded, normalized to sum to 1 over off-diagonal entries.

Operates on the full training corpus. One-shot preprocessing; saves the
256x256 matrix to disk so all three Exp 2 conditions load the same
co-occurrence data.

Output:
  runs/exp2_cooccur/cooccur_w5.npy           # (256, 256) float32
  runs/exp2_cooccur/cooccur_w5_raw.npy       # unnormalized counts
  runs/exp2_cooccur/metadata.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "runs" / "exp2_cooccur"
DATA_DIR = Path.home() / ".cache" / "neurogen"

WINDOW = 5
VOCAB = 256


def compute_cooccur(
    data: np.ndarray, window: int = 5, vocab: int = 256,
    chunk_size: int = 50_000_000,
) -> np.ndarray:
    """Symmetric co-occurrence in a window of `window` bytes either side.
    Excludes self-pairs (same position). Returns raw counts (int64).

    Chunked to avoid allocating full copies of a 1.9B-byte array. Each chunk
    loads ≤ chunk_size bytes cast to uint32 (enough to hold left*256+right
    flat indices, max 65535 < 2^32)."""
    mat = np.zeros((vocab, vocab), dtype=np.int64)
    N = len(data)
    # Include `window` bytes of overlap between chunks so we don't miss
    # cross-chunk pairs (only matters at boundaries).
    overlap = window
    start = 0
    while start < N:
        end = min(N, start + chunk_size)
        chunk = data[start:end].astype(np.uint32, copy=False)
        L = len(chunk)
        for offset in range(1, window + 1):
            if L <= offset:
                break
            left = chunk[:-offset]
            right = chunk[offset:]
            flat = (left * vocab) + right
            counts = np.bincount(flat, minlength=vocab * vocab)
            # Accumulate as int64 to avoid int32 overflow across the full corpus.
            mat += counts.astype(np.int64).reshape(vocab, vocab)
        if end >= N:
            break
        # Advance by chunk_size − overlap so adjacent bytes at the boundary
        # are counted exactly once across (this chunk's tail, next chunk's head).
        start += chunk_size - overlap
        del chunk
    # Symmetrize and zero diagonal.
    mat = mat + mat.T
    np.fill_diagonal(mat, 0)
    return mat


def main() -> None:
    shards = sorted(DATA_DIR.glob("train_*.bin"))
    if not shards:
        raise FileNotFoundError(DATA_DIR)
    print(f"Loading {len(shards)} training shards…")
    arrs = [np.fromfile(p, dtype=np.uint8) for p in shards]
    data = np.concatenate(arrs)
    N = len(data)
    print(f"  {N:,} bytes")

    t0 = time.time()
    raw = compute_cooccur(data, window=WINDOW, vocab=VOCAB)
    print(f"Computed co-occurrence in {time.time()-t0:.1f}s")

    # Frequency normalization — used only as diagnostic / alternative.
    total = raw.sum()
    if total == 0:
        raise RuntimeError("empty co-occurrence")
    freq = (raw.astype(np.float64) / total).astype(np.float32)

    # DEFAULT: log1p + normalize. Raw frequency is dominated by a few pairs
    # (space-letter bigrams). Top-1% of pairs carry 57% of raw mass and the
    # space-letter rows alone carry ~17%. The topographic regularizer would
    # be dominated by those pairs and could collapse letters toward space.
    # log1p(raw) / sum flattens the distribution (max/median drops from 3.7M
    # to 4.3) while preserving ordering.
    log_raw = np.log1p(raw.astype(np.float64))
    np.fill_diagonal(log_raw, 0.0)
    cooccur = (log_raw / log_raw.sum()).astype(np.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "cooccur_w5.npy", cooccur)           # DEFAULT
    np.save(OUT_DIR / "cooccur_w5_freq.npy", freq)          # freq-normalized
    np.save(OUT_DIR / "cooccur_w5_raw.npy", raw)            # raw counts

    # Diagnostics
    row_sums = cooccur.sum(axis=1)
    nnz_frac = (cooccur > 0).mean()
    top_pairs = np.dstack(np.unravel_index(np.argsort(cooccur.ravel())[::-1][:20], cooccur.shape))[0]
    print(f"Off-diagonal nnz fraction: {nnz_frac:.3f}")
    print(f"Top 20 pairs by cooccur weight:")
    for i, j in top_pairs:
        ci = repr(chr(int(i))) if 32 <= int(i) < 127 else f"\\x{int(i):02x}"
        cj = repr(chr(int(j))) if 32 <= int(j) < 127 else f"\\x{int(j):02x}"
        print(f"  {ci} ↔ {cj}  weight={cooccur[i, j]:.6f}  raw={raw[i, j]:,}")

    meta = {
        "window": WINDOW,
        "vocab": VOCAB,
        "total_bytes": int(N),
        "total_pairs": int(total),
        "nnz_fraction": float(nnz_frac),
        "max_weight": float(cooccur.max()),
        "shape": list(cooccur.shape),
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved to {OUT_DIR}")
    print(f"Metadata: {meta}")


if __name__ == "__main__":
    main()
