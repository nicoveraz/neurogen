"""Exp 1 — bracket openers/closers sub-class coherence.

Exp 1 within_class_coherence found brackets '()[]{}' never form a cluster.
Hypothesis: openers `([{` and closers `)]}` are each internally coherent,
dissimilar to the other group. Re-run coherence on the two sub-classes.

Also check: the openers are pairwise with their matching closers. Compute
mean cosine for the three pair groups (, ) | [ , ] | { , }.

Outputs:
  analysis/exp1_trajectories/bracket_subclass.json
  reports/exp1/bracket_subclass_coherence.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "runs" / "exp0_baseline"
ANALYSIS_DIR = REPO_ROOT / "analysis" / "exp1_trajectories"
REPORT_DIR = REPO_ROOT / "reports" / "exp1"


GROUPS = {
    "openers ([{":     [ord(c) for c in "([{"],
    "closers )]}":     [ord(c) for c in ")]}"],
    "brackets (all 6)": [ord(c) for c in "()[]{}"],
}
PAIRS = {
    "() pair": [ord("("), ord(")")],
    "[] pair": [ord("["), ord("]")],
    "{} pair": [ord("{"), ord("}")],
}
# Cross-group: opener_i × closer_j for all i,j — should be LOW if our
# hypothesis is right (openers and closers are distinct sub-classes).
OPEN_IDS = [ord(c) for c in "([{"]
CLOSE_IDS = [ord(c) for c in ")]}"]


def cosine_matrix(A: np.ndarray) -> np.ndarray:
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    return An @ An.T


def within_mean(vecs: np.ndarray, idxs: list[int]) -> float:
    if len(idxs) < 2:
        return float("nan")
    C = cosine_matrix(vecs[idxs])
    iu = np.triu_indices(len(idxs), k=1)
    return float(C[iu].mean())


def cross_mean(vecs: np.ndarray, A: list[int], B: list[int]) -> float:
    vA = vecs[A] / (np.linalg.norm(vecs[A], axis=1, keepdims=True) + 1e-12)
    vB = vecs[B] / (np.linalg.norm(vecs[B], axis=1, keepdims=True) + 1e-12)
    return float((vA @ vB.T).mean())


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

    curves: dict[str, list[float]] = {name: [] for name in GROUPS}
    curves.update({name: [] for name in PAIRS})
    curves["open × close (cross)"] = []
    curves["__random__"] = []

    rng = np.random.default_rng(42)
    for path in snaps:
        W = np.load(path)
        for name, idxs in GROUPS.items():
            curves[name].append(within_mean(W, idxs))
        for name, idxs in PAIRS.items():
            curves[name].append(within_mean(W, idxs))
        curves["open × close (cross)"].append(cross_mean(W, OPEN_IDS, CLOSE_IDS))
        curves["__random__"].append(random_pair_mean(W, 500, rng))

    print(f"{'group':<22} {'init':>8} {'final':>8} {'peak':>8} {'peak_step':>9}")
    print("-" * 62)
    for name in list(GROUPS) + list(PAIRS) + ["open × close (cross)", "__random__"]:
        c = np.asarray(curves[name])
        pi = int(np.argmax(c))
        print(f"{name:<22} {c[0]:+.4f} {c[-1]:+.4f} {c[pi]:+.4f} {steps[pi]:>9d}")

    out = {"steps": steps, "curves": curves}
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "bracket_subclass.json").write_text(json.dumps(out, indent=2))
    print(f"\nStats → {ANALYSIS_DIR/'bracket_subclass.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
        styles = {
            "openers ([{":        ("C2", "-",  1.8),
            "closers )]}":        ("C3", "-",  1.8),
            "brackets (all 6)":   ("C0", "-",  1.3),
            "open × close (cross)": ("C5", "--", 1.3),
            "() pair":            ("C1", ":",  1.0),
            "[] pair":            ("C4", ":",  1.0),
            "{} pair":            ("C6", ":",  1.0),
            "__random__":         ("black", "--", 0.8),
        }
        for name, (color, ls, lw) in styles.items():
            ax.plot(steps, curves[name], color=color, ls=ls, lw=lw, label=name, alpha=0.9)
        ax.set_xlabel("training step")
        ax.set_ylabel("mean pairwise cosine")
        ax.set_title("Bracket sub-class coherence: openers vs closers vs combined")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "bracket_subclass_coherence.png", dpi=140)
        print(f"Plot → {REPORT_DIR/'bracket_subclass_coherence.png'}")
    except ImportError:
        print("matplotlib unavailable")


if __name__ == "__main__":
    main()
