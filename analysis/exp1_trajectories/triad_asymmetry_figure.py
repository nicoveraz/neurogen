"""Comparison figure for §3.3: anchor-driven convergence conditional on asymmetry.

Overlays centroid-approach trajectories for:
  - `.!?` (primary case, high asymmetry, sequential)
  - `,;:` (confirming case, very high asymmetry, modest spread)
  - A digit triple (symmetric, simultaneous — representative null)
  - A letter triple (low asymmetry, doesn't form tight cluster — representative null)

Each trajectory: cosine-with-final-centroid per token over training steps.
Vertical lines mark the 0.6 threshold crossings when visible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "runs" / "exp0_baseline"
REPORT_DIR = REPO_ROOT / "reports" / "exp1"


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float((a @ b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def centroid_series(Ws, tokens):
    W_final = Ws[-1]
    centroid = W_final[tokens].mean(axis=0)
    return {t: [cos(W[t], centroid) for W in Ws] for t in tokens}


def main() -> None:
    snaps = sorted((RUN_DIR / "snapshots").glob("wte_step_*.npy"))
    steps = np.asarray([int(p.stem.split("_")[-1]) for p in snaps])
    Ws = [np.load(p) for p in snaps]

    triads = {
        ".!? (anchor .; 30× asymm; spread 5.3×)": {
            "tokens": [ord("."), ord("!"), ord("?")],
            "labels": {ord("."): ".", ord("!"): "!", ord("?"): "?"},
            "anchor": ord("."),
        },
        ",;: (anchor ,; 978× asymm; spread 1.3×)": {
            "tokens": [ord(","), ord(";"), ord(":")],
            "labels": {ord(","): ",", ord(";"): ";", ord(":"): ":"},
            "anchor": ord(","),
        },
        "5 6 8 (digit symm; 1.6× asymm; spread 1.2×)": {
            "tokens": [ord("5"), ord("6"), ord("8")],
            "labels": {ord("5"): "5", ord("6"): "6", ord("8"): "8"},
            "anchor": None,
        },
        "e s d (letter; 2.3× asymm; cluster too loose)": {
            "tokens": [ord("e"), ord("s"), ord("d")],
            "labels": {ord("e"): "e", ord("s"): "s", ord("d"): "d"},
            "anchor": None,
        },
    }

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    colors = ["C0", "C3", "C2"]

    for ax, (title, spec) in zip(axs.flatten(), triads.items()):
        series = centroid_series(Ws, spec["tokens"])
        for t, color in zip(spec["tokens"], colors):
            label = f"{spec['labels'][t]!r}" + (" [anchor]" if t == spec["anchor"] else "")
            ax.plot(steps, series[t], label=label, color=color, lw=1.6)
            # Mark threshold crossing at 0.6 if achieved
            s = np.asarray(series[t])
            if (s >= 0.6).any():
                hit = int(np.argmax(s >= 0.6))
                ax.axvline(steps[hit], color=color, ls=":", lw=0.8, alpha=0.5)
        ax.axhline(0.6, ls="--", color="gray", lw=0.7, alpha=0.6,
                   label="0.6 threshold" if ax is axs[0, 0] else None)
        ax.set_xscale("symlog", linthresh=500)
        ax.set_ylim(-0.3, 1.05)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower right")

    for ax in axs[1]:
        ax.set_xlabel("training step")
    for ax in axs[:, 0]:
        ax.set_ylabel("cosine with final-triad centroid")

    fig.suptitle(
        "Anchor-driven convergence is asymmetry-gated\n"
        ".!? and ,;: anchor and migrate; symmetric digit triple converges together; "
        "loose letter cluster never reaches 0.6",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out = REPORT_DIR / "triad_asymmetry_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Plot → {out}")


if __name__ == "__main__":
    main()
