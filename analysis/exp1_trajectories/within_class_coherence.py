"""Exp 1 analysis — within-class trajectory coherence over training.

For each byte class that showed strong final-embedding clustering in Exp 0
(digits, sentence-punct, uppercase, lowercase, vowels, clause-punct, quotes,
whitespace, consonants, brackets), compute the mean pairwise cosine similarity
of member tokens' *positions* (not displacements — see Exp 0 §11) at every
checkpoint snapshot. The resulting curves show when each class forms as a
coherent cluster.

Operates on positions, not displacements (per methodological principle from
Exp 0 §11). Random-pair baseline at each checkpoint included as null.

Outputs:
  analysis/exp1_trajectories/within_class_coherence.json
  reports/exp1/within_class_coherence.png
"""

from __future__ import annotations

import json
import string
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "runs" / "exp0_baseline"
ANALYSIS_DIR = REPO_ROOT / "analysis" / "exp1_trajectories"
REPORT_DIR = REPO_ROOT / "reports" / "exp1"


def byte_classes() -> dict[str, list[int]]:
    return {
        "digits":         [ord(c) for c in string.digits],
        "sentence_punct": [ord(c) for c in ".!?"],
        "uppercase":      [ord(c) for c in string.ascii_uppercase],
        "lowercase":      [ord(c) for c in string.ascii_lowercase],
        "vowels":         [ord(c) for c in "aeiou"],
        "consonants":     [ord(c) for c in "bcdfghjklmnpqrstvwxyz"],
        "clause_punct":   [ord(c) for c in ",;:"],
        "quotes":         [ord(c) for c in "\"'"],
        "whitespace":     [ord(c) for c in " \t\n\r"],
        "brackets":       [ord(c) for c in "()[]{}"],
    }


def cosine_matrix(A: np.ndarray) -> np.ndarray:
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    return An @ An.T


def within_mean(C: np.ndarray) -> float:
    n = C.shape[0]
    if n < 2:
        return float("nan")
    iu = np.triu_indices(n, k=1)
    return float(C[iu].mean())


def random_pair_mean(vecs: np.ndarray, n_pairs: int, rng: np.random.Generator) -> float:
    V = vecs.shape[0]
    i = rng.integers(0, V, size=n_pairs)
    j = rng.integers(0, V, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    vi = vecs[i] / (np.linalg.norm(vecs[i], axis=1, keepdims=True) + 1e-12)
    vj = vecs[j] / (np.linalg.norm(vecs[j], axis=1, keepdims=True) + 1e-12)
    return float((vi * vj).sum(axis=1).mean())


def main() -> None:
    snaps = sorted((RUN_DIR / "snapshots").glob("wte_step_*.npy"))
    steps = [int(p.stem.split("_")[-1]) for p in snaps]
    print(f"Loading {len(snaps)} snapshots, steps {steps[0]} … {steps[-1]}")

    classes = byte_classes()
    rng = np.random.default_rng(42)
    n_pairs = 500

    curves: dict[str, list[float]] = {name: [] for name in classes}
    curves["__random__"] = []

    for path in snaps:
        W = np.load(path)
        C = cosine_matrix(W)
        for name, idxs in classes.items():
            subC = C[np.ix_(idxs, idxs)]
            curves[name].append(within_mean(subC))
        curves["__random__"].append(random_pair_mean(W, n_pairs, rng))

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "steps": steps,
        "curves": {k: v for k, v in curves.items()},
        "n_random_pairs": n_pairs,
    }
    (ANALYSIS_DIR / "within_class_coherence.json").write_text(json.dumps(out, indent=2))
    print(f"Stats → {ANALYSIS_DIR/'within_class_coherence.json'}")

    # Summary stats per class
    print()
    print(f"{'class':<18} {'init':>8} {'final':>8} {'peak':>8} {'peak_step':>9} {'lift@final':>11}")
    print("-" * 72)
    final_rand = curves["__random__"][-1]
    for name in classes:
        c = np.asarray(curves[name])
        peak_idx = int(np.nanargmax(c))
        lift_final = c[-1] - final_rand
        print(f"{name:<18} {c[0]:+.4f} {c[-1]:+.4f} {c[peak_idx]:+.4f} {steps[peak_idx]:>9d}  {lift_final:+.4f}")
    print(f"{'__random__':<18} {curves['__random__'][0]:+.4f} {curves['__random__'][-1]:+.4f}")

    # Find "formation step" per class: first step where within-class cosine
    # exceeds the random baseline by at least 0.05 AND stays above that
    # threshold for the rest of training. This is the "cluster has formed"
    # moment.
    print()
    print("Formation step (first step where class cos > random_baseline + 0.05 sustainedly):")
    for name in classes:
        c = np.asarray(curves[name])
        r = np.asarray(curves["__random__"])
        lift = c - r
        above = lift > 0.05
        # Find first step after which `above` stays True
        form = None
        for k in range(len(above)):
            if above[k] and bool(above[k:].all()):
                form = steps[k]
                break
        if form is None:
            # fallback: first step where lift crosses 0.05 (not necessarily sustained)
            idx = np.argmax(lift > 0.05)
            form = steps[int(idx)] if bool((lift > 0.05).any()) else None
        print(f"  {name:<18} {form}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

        # Full curve
        palette = {
            "digits":         "C0",
            "sentence_punct": "C3",
            "uppercase":      "C2",
            "lowercase":      "C1",
            "vowels":         "C4",
            "consonants":     "C5",
            "clause_punct":   "C6",
            "quotes":         "C7",
            "whitespace":     "C8",
            "brackets":       "C9",
        }
        for name in classes:
            ax1.plot(steps, curves[name], label=name, color=palette[name], lw=1.4)
        ax1.plot(steps, curves["__random__"], label="random baseline",
                 color="black", ls="--", lw=1.0, alpha=0.6)
        ax1.set_ylabel("mean pairwise cosine (within class)")
        ax1.set_title("Within-class trajectory coherence — full run")
        ax1.grid(True, alpha=0.3)
        ax1.legend(ncol=3, fontsize=8, loc="lower right")

        # Zoom on first 20K steps — where formation is expected to happen
        mask_early = np.asarray(steps) <= 20_000
        steps_e = np.asarray(steps)[mask_early]
        for name in classes:
            ax2.plot(steps_e, np.asarray(curves[name])[mask_early],
                     label=name, color=palette[name], lw=1.4)
        ax2.plot(steps_e, np.asarray(curves["__random__"])[mask_early],
                 color="black", ls="--", lw=1.0, alpha=0.6, label="random")
        ax2.set_xlabel("training step")
        ax2.set_ylabel("mean pairwise cosine (within class)")
        ax2.set_title("Zoom: first 20K steps (formation window)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(ncol=3, fontsize=8, loc="lower right")

        fig.tight_layout()
        fig.savefig(REPORT_DIR / "within_class_coherence.png", dpi=140)
        print(f"Plot → {REPORT_DIR/'within_class_coherence.png'}")
    except ImportError:
        print("matplotlib not available; skipping plot")


if __name__ == "__main__":
    main()
