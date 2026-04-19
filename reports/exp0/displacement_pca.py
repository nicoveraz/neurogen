"""Displacement-vector PCA diagnostic (Exp 0 → Exp 1 bridge).

Computes PCA on the (w_final - w_initial) displacement vectors across all tokens.
Eigenvalue spectrum tells us whether trajectory structure is low-rank (few
dominant axes, strong structure) or flat (many independent directions, weaker
structure). This calibrates how elaborate Exp 1's analyses need to be.

Outputs:
  - stdout: top-k eigenvalues, cumulative variance, participation ratio
  - reports/exp0/displacement_pca.png: eigenvalue spectrum + cumulative variance
  - reports/exp0/displacement_stats.json: numbers for the report
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "runs" / "exp0_baseline"
OUT_DIR = REPO_ROOT / "reports" / "exp0"


def main() -> None:
    snaps = sorted((RUN_DIR / "snapshots").glob("wte_step_*.npy"))
    w0 = np.load(snaps[0])     # (V, D) at init
    wL = np.load(snaps[-1])    # (V, D) at final
    V, D = w0.shape
    disp = wL - w0             # (V, D) per-token displacement

    # Magnitudes (already reported in evaluation, reproduce here)
    mag = np.linalg.norm(disp, axis=1)

    # PCA on displacement vectors (tokens as samples, dims as features)
    # Center across tokens (remove any global translation component)
    disp_mean = disp.mean(axis=0, keepdims=True)
    disp_c = disp - disp_mean
    translation_frac = np.linalg.norm(disp_mean) / np.linalg.norm(disp)

    # SVD instead of cov-eigh: numerically cleaner, same eigenvalues via s**2/(V-1)
    _, s, _ = np.linalg.svd(disp_c, full_matrices=False)
    eigvals = s ** 2 / (V - 1)
    total_var = eigvals.sum()
    explained = eigvals / total_var
    cumulative = np.cumsum(explained)

    # Participation ratio — intuitive "effective rank"
    part_ratio = (eigvals.sum() ** 2) / (eigvals ** 2).sum()

    top_k = 10
    stats = {
        "vocab": int(V),
        "dim": int(D),
        "displacement_magnitude": {
            "min": float(mag.min()),
            "median": float(np.median(mag)),
            "mean": float(mag.mean()),
            "max": float(mag.max()),
            "std": float(mag.std()),
        },
        "translation_fraction": float(translation_frac),
        "participation_ratio": float(part_ratio),
        "top_eigenvalues": [float(x) for x in eigvals[:top_k]],
        "explained_variance_top_k": [float(x) for x in explained[:top_k]],
        "cumulative_variance_top_k": [float(x) for x in cumulative[:top_k]],
        "cumvar_at": {
            "k=1": float(cumulative[0]),
            "k=3": float(cumulative[2]),
            "k=5": float(cumulative[4]),
            "k=10": float(cumulative[9]),
            "k=20": float(cumulative[19]),
            "k=50": float(cumulative[49]),
        },
    }

    print(f"Vocab×Dim: {V}×{D}")
    print(f"Displacement magnitude: min={mag.min():.3f}  median={np.median(mag):.3f}  "
          f"mean={mag.mean():.3f}  max={mag.max():.3f}  std={mag.std():.3f}")
    print(f"Translation fraction ||mean(disp)|| / ||disp||: {translation_frac:.4f}  "
          f"({'<<1: token-specific' if translation_frac < 0.3 else 'substantial global drift'})")
    print(f"Participation ratio (effective rank of displacements): {part_ratio:.2f} / {D}")
    print(f"\nTop-{top_k} eigenvalue spectrum:")
    for i in range(top_k):
        print(f"  PC{i+1:>2d}  eigval={eigvals[i]:>8.4f}  "
              f"explained={explained[i]*100:5.2f}%  "
              f"cum={cumulative[i]*100:6.2f}%")
    print(f"\nCumulative variance:")
    for k, v in stats["cumvar_at"].items():
        print(f"  {k}: {v*100:5.2f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "displacement_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nStats written to {OUT_DIR/'displacement_stats.json'}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        k_plot = min(64, D)
        xs = np.arange(1, k_plot + 1)
        ax1.semilogy(xs, eigvals[:k_plot], marker="o", markersize=3)
        ax1.set_xlabel("Component")
        ax1.set_ylabel("Eigenvalue (log)")
        ax1.set_title(f"Displacement spectrum (top {k_plot})\n"
                      f"participation ratio = {part_ratio:.1f}")
        ax1.grid(True, alpha=0.3)

        ax2.plot(xs, cumulative[:k_plot] * 100, marker="o", markersize=3)
        ax2.axhline(50, ls="--", color="gray", alpha=0.5, label="50%")
        ax2.axhline(90, ls="--", color="gray", alpha=0.5, label="90%")
        ax2.set_xlabel("Components kept")
        ax2.set_ylabel("Cumulative variance explained (%)")
        ax2.set_title("Cumulative variance")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.tight_layout()
        fig.savefig(OUT_DIR / "displacement_pca.png", dpi=140)
        print(f"Plot saved to {OUT_DIR/'displacement_pca.png'}")
    except ImportError:
        print("matplotlib not available; skipping plot")


if __name__ == "__main__":
    main()
