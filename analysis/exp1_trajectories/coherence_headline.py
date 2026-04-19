"""Exp 1 — headline coherence figure.

Single figure conveying the within-class coherence finding: log-step x-axis,
one trace per class, color-coded by class type. Candidate for paper Figure 1.

Pulls from the existing within_class_coherence.json produced by
`within_class_coherence.py`.

Outputs:
  reports/exp1/headline_coherence.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO_ROOT / "analysis" / "exp1_trajectories"
REPORT_DIR = REPO_ROOT / "reports" / "exp1"


# Color families: letters cool, digits/punct warm, ws/brackets neutral
STYLES = {
    "sentence_punct": ("#D7263D", "-",  2.0, "letters/punct"),
    "clause_punct":   ("#F46036", "-",  1.8, "letters/punct"),
    "digits":         ("#2E86AB", "-",  2.2, "digits"),
    "uppercase":      ("#4F86C6", "-",  1.5, "letters/punct"),
    "lowercase":      ("#6AA8E6", "-",  1.5, "letters/punct"),
    "vowels":         ("#7FB3D5", "--", 1.2, "letters/punct"),
    "consonants":     ("#9CC5E4", "--", 1.2, "letters/punct"),
    "quotes":         ("#8E44AD", ":",  1.3, "special"),
    "whitespace":     ("#7F7F7F", ":",  1.2, "special"),
    "brackets":       ("#AAAAAA", ":",  1.0, "special"),
}
CLASSES_ORDER = list(STYLES.keys())


def main() -> None:
    stats = json.loads((ANALYSIS_DIR / "within_class_coherence.json").read_text())
    steps = np.asarray(stats["steps"])
    curves = stats["curves"]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for name in CLASSES_ORDER:
        color, ls, lw, _ = STYLES[name]
        ax.plot(steps, curves[name], color=color, ls=ls, lw=lw, label=name, alpha=0.95)
    ax.plot(steps, curves["__random__"], color="black", ls="--", lw=0.9,
            label="random baseline", alpha=0.6)

    # Annotate peak-then-decline signatures
    for name in ["digits", "clause_punct"]:
        c = np.asarray(curves[name])
        pi = int(np.argmax(c))
        ax.annotate(
            f"{name}: peak @ step {steps[pi]:,}",
            xy=(steps[pi], c[pi]),
            xytext=(steps[pi] * 1.15, c[pi] + 0.03),
            fontsize=9, ha="left",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8, alpha=0.7),
        )

    ax.set_xscale("symlog", linthresh=500)
    ax.set_xlabel("training step (log / symlog)")
    ax.set_ylabel("mean pairwise cosine (within class)")
    ax.set_title(
        "Within-class trajectory coherence — byte-level 3.4M on TinyStories\n"
        "Peak-then-decline (digits, clause_punct) signals coarse-category then "
        "within-category refinement"
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(ncol=2, fontsize=9, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(REPORT_DIR / "headline_coherence.png", dpi=150)
    print(f"Plot → {REPORT_DIR/'headline_coherence.png'}")


if __name__ == "__main__":
    main()
