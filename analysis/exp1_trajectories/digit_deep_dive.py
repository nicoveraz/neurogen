"""Exp 1 — digit deep dive.

The within-class coherence analysis showed digits peaked at cosine +0.49 at
step 47K, then declined to +0.44 by step 100K. If this decline carries
numerical-ordering information (the model learning that `0` is closer to
`1` than to `9`), then at late training the pairwise digit cosine should
correlate with |i - j|.

Specifically:
  - Compute 10x10 pairwise cosine matrix of digit tokens at each checkpoint.
  - At each checkpoint, fit slope / correlation of digit-digit cosine vs
    numerical distance |i - j|. If the slope becomes increasingly negative
    over training, the cluster's loosening is ordinal.
  - Compare the matrix at peak (step 47K) vs final (step 100K) visually.

This is the headline case study for Exp 1 because it demonstrates phase-
structured representation formation — coarse-category discovery followed by
within-category refinement — visible only in the trajectory, not in the
endpoints.

Outputs:
  analysis/exp1_trajectories/digit_deep_dive.json
  reports/exp1/digit_pairwise_peak_vs_final.png
  reports/exp1/digit_ordering_correlation.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    r = float(np.corrcoef(x, y)[0, 1])
    return r, float("nan")


def spearmanr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return pearsonr(rx.astype(float), ry.astype(float))

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "runs" / "exp0_baseline"
ANALYSIS_DIR = REPO_ROOT / "analysis" / "exp1_trajectories"
REPORT_DIR = REPO_ROOT / "reports" / "exp1"

DIGITS = [ord(c) for c in "0123456789"]  # byte ids 48..57
DIGIT_CHARS = list("0123456789")


def cosine_matrix(A: np.ndarray) -> np.ndarray:
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    return An @ An.T


def digit_cosine(W: np.ndarray) -> np.ndarray:
    return cosine_matrix(W[DIGITS])


def ordering_correlation(C: np.ndarray) -> tuple[float, float]:
    """Correlation between digit-digit cosine and numerical distance |i-j|.
    Returns (pearson_r, spearman_r). Negative correlation = higher cosine at
    smaller numerical distance = ordinal structure."""
    n = C.shape[0]
    iu = np.triu_indices(n, k=1)
    cos_pairs = C[iu]
    num_dist = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))[iu]
    pr, _ = pearsonr(cos_pairs, num_dist)
    sr, _ = spearmanr(cos_pairs, num_dist)
    return float(pr), float(sr)


def main() -> None:
    snaps = sorted((RUN_DIR / "snapshots").glob("wte_step_*.npy"))
    steps = [int(p.stem.split("_")[-1]) for p in snaps]

    # Per-checkpoint: mean digit-digit cos and ordering correlations
    mean_cos, pearson_series, spearman_series = [], [], []
    matrices_at = {}  # cache full matrices at some checkpoints

    want_steps = {0, 2000, 5000, 10000, 20000, 30000, 47000, 70000, 100000}

    for step, path in zip(steps, snaps):
        W = np.load(path)
        C = digit_cosine(W)
        iu = np.triu_indices(10, k=1)
        mean_cos.append(float(C[iu].mean()))
        pr, sr = ordering_correlation(C)
        pearson_series.append(pr)
        spearman_series.append(sr)
        if step in want_steps:
            matrices_at[step] = C.tolist()

    # Identify peak step from the coherence curve
    peak_idx = int(np.argmax(mean_cos))
    peak_step = steps[peak_idx]
    final_step = steps[-1]
    print(f"Digit coherence peak: step={peak_step} mean_cos={mean_cos[peak_idx]:.4f}")
    print(f"Digit coherence final: step={final_step} mean_cos={mean_cos[-1]:.4f}")

    # Ensure the actual peak step is in the matrices cache
    if peak_step not in matrices_at:
        W = np.load(snaps[peak_idx])
        matrices_at[peak_step] = digit_cosine(W).tolist()

    # Headline matrices
    C_peak = np.asarray(matrices_at[peak_step])
    C_final = np.asarray(matrices_at[final_step])
    delta = C_final - C_peak

    print()
    print("Ordering correlation (cos vs |i-j|, negative = ordinal):")
    for step, p, s in zip(steps, pearson_series, spearman_series):
        if step in (0, 2000, 10000, 20000, 47000, 70000, 100000):
            print(f"  step={step:>6d}  pearson={p:+.3f}  spearman={s:+.3f}  "
                  f"mean_cos={mean_cos[steps.index(step)]:+.3f}")

    # Save stats
    out = {
        "steps": steps,
        "mean_digit_cos": mean_cos,
        "pearson_cos_vs_numdist": pearson_series,
        "spearman_cos_vs_numdist": spearman_series,
        "peak_step": peak_step,
        "final_step": final_step,
        "matrices_at": matrices_at,
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "digit_deep_dive.json").write_text(json.dumps(out, indent=2))
    print(f"\nStats → {ANALYSIS_DIR/'digit_deep_dive.json'}")

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        # --- Plot 1: pairwise cosine matrices peak vs final, plus delta ----
        fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
        for ax, M, title in zip(
            axs,
            [C_peak, C_final, delta],
            [f"cos matrix @ peak (step {peak_step})",
             f"cos matrix @ final (step {final_step})",
             f"Δ = final − peak"],
        ):
            vmin = -0.3 if "Δ" in title else -0.2
            vmax =  0.3 if "Δ" in title else  1.0
            im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap="RdBu_r" if "Δ" in title else "viridis")
            ax.set_xticks(range(10)); ax.set_xticklabels(DIGIT_CHARS)
            ax.set_yticks(range(10)); ax.set_yticklabels(DIGIT_CHARS)
            ax.set_title(title)
            plt.colorbar(im, ax=ax, shrink=0.85)
            # Annotate cells
            for i in range(10):
                for j in range(10):
                    v = M[i, j]
                    color = "white" if abs(v) > (0.5 if "Δ" not in title else 0.15) else "black"
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=7, color=color)
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "digit_pairwise_peak_vs_final.png", dpi=140)
        print(f"Plot → {REPORT_DIR/'digit_pairwise_peak_vs_final.png'}")

        # --- Plot 2: ordering correlation over training --------------------
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.plot(steps, mean_cos, lw=1.6, color="C0")
        ax1.axvline(peak_step, ls="--", color="gray", alpha=0.6,
                    label=f"peak @ step {peak_step}")
        ax1.set_ylabel("mean digit pairwise cos")
        ax1.set_title("Digit class coherence over training")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(steps, pearson_series, lw=1.4, color="C3", label="Pearson")
        ax2.plot(steps, spearman_series, lw=1.4, color="C2", label="Spearman", alpha=0.7)
        ax2.axhline(0, color="black", lw=0.5)
        ax2.axvline(peak_step, ls="--", color="gray", alpha=0.6)
        ax2.set_xlabel("training step")
        ax2.set_ylabel("corr(cos, |i−j|)")
        ax2.set_title("Ordering correlation — negative = ordinal (close numbers more similar)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.tight_layout()
        fig.savefig(REPORT_DIR / "digit_ordering_correlation.png", dpi=140)
        print(f"Plot → {REPORT_DIR/'digit_ordering_correlation.png'}")
    except ImportError:
        print("matplotlib / scipy issue; skipping plots")


if __name__ == "__main__":
    main()
