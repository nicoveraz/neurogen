"""Exp 1 — sentence-punct `.!?` triad: sequential vs simultaneous convergence.

Named case study from Exp 0 §12: the triad ends at mutual cosine 0.47-0.62
and Δ cos = −0.11 ("different starts, shared destination"). The
within-class coherence curve showed the triad reached formation threshold
at step ~3000.

Open question: do all three tokens converge simultaneously, or does one act
as anchor and the others migrate to it?

Analysis:
  1. Pairwise cosine curves: '.'–'!', '.'–'?', '!'–'?' over all snapshots.
     Compare when each pair reaches threshold.
  2. Per-token nearest-neighbor: at each snapshot, compute each triad
     token's top-5 nearest neighbors (excluding self). Count how often each
     triad token appears in the others' top-K over time.
  3. Centroid-pull test: define triad centroid at final step as c =
     mean(w_L['.'], w_L['!'], w_L['?']). At each prior snapshot, compute
     each token's cosine with the final centroid. If one token approaches
     the centroid first, it's the anchor.

Outputs:
  analysis/exp1_trajectories/triad_sequential.json
  reports/exp1/triad_pairwise_cos.png
  reports/exp1/triad_centroid_pull.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "runs" / "exp0_baseline"
ANALYSIS_DIR = REPO_ROOT / "analysis" / "exp1_trajectories"
REPORT_DIR = REPO_ROOT / "reports" / "exp1"

DOT, BANG, QUE = ord("."), ord("!"), ord("?")
TRIAD = [DOT, BANG, QUE]
TRIAD_CHARS = {DOT: ".", BANG: "!", QUE: "?"}


def cos(a: np.ndarray, b: np.ndarray) -> float:
    an = a / (np.linalg.norm(a) + 1e-12)
    bn = b / (np.linalg.norm(b) + 1e-12)
    return float(an @ bn)


def main() -> None:
    snaps = sorted((RUN_DIR / "snapshots").glob("wte_step_*.npy"))
    steps = [int(p.stem.split("_")[-1]) for p in snaps]

    # Load final-step for centroid
    W_final = np.load(snaps[-1])
    triad_centroid = W_final[TRIAD].mean(axis=0)

    pair_names = [(DOT, BANG), (DOT, QUE), (BANG, QUE)]
    pair_labels = [f"{TRIAD_CHARS[a]}–{TRIAD_CHARS[b]}" for a, b in pair_names]

    pair_curves = {lab: [] for lab in pair_labels}
    centroid_curves = {TRIAD_CHARS[t]: [] for t in TRIAD}
    # Nearest-neighbor tracking
    nn_counts = {TRIAD_CHARS[t]: {TRIAD_CHARS[o]: 0 for o in TRIAD if o != t}
                 for t in TRIAD}
    nn_per_step = []  # per-step top-3 for each triad token

    for step, path in zip(steps, snaps):
        W = np.load(path)
        # Pairwise cos
        for (a, b), lab in zip(pair_names, pair_labels):
            pair_curves[lab].append(cos(W[a], W[b]))
        # Each triad token's cos with final centroid
        for t in TRIAD:
            centroid_curves[TRIAD_CHARS[t]].append(cos(W[t], triad_centroid))
        # Nearest neighbors of each triad token (exclude self + other triad members
        # to see *external* neighbors, but ALSO record triad rank among all neighbors)
        wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
        sims = wn @ wn.T
        np.fill_diagonal(sims, -np.inf)
        step_nn = {}
        for t in TRIAD:
            top5 = np.argsort(-sims[t])[:5]
            step_nn[TRIAD_CHARS[t]] = [int(i) for i in top5]
            # rank of each other triad member among all neighbors
            for o in TRIAD:
                if o == t:
                    continue
                rank_o = int((sims[t] > sims[t, o]).sum())  # 0 = best
                nn_counts[TRIAD_CHARS[t]].setdefault(f"rank_of_{TRIAD_CHARS[o]}", [])
        nn_per_step.append({"step": step, "top5": step_nn})

    # Per-step, track rank of each other triad member (lower = closer)
    rank_curves = {
        TRIAD_CHARS[t]: {TRIAD_CHARS[o]: [] for o in TRIAD if o != t}
        for t in TRIAD
    }
    for step, path in zip(steps, snaps):
        W = np.load(path)
        wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
        sims = wn @ wn.T
        np.fill_diagonal(sims, -np.inf)
        for t in TRIAD:
            for o in TRIAD:
                if o == t:
                    continue
                # rank of o among all 256 tokens (0 = nearest)
                rank_o = int((sims[t] > sims[t, o]).sum())
                rank_curves[TRIAD_CHARS[t]][TRIAD_CHARS[o]].append(rank_o)

    # Printed summary: which pair forms first (above 0.3 threshold)?
    print("Pair cosine trajectory (threshold ≥ 0.3 means 'clustered'):")
    thr = 0.3
    for (a, b), lab in zip(pair_names, pair_labels):
        vals = np.asarray(pair_curves[lab])
        hit = int(np.argmax(vals >= thr)) if (vals >= thr).any() else -1
        first_step = steps[hit] if hit >= 0 else None
        print(f"  {lab}:  init={vals[0]:+.3f}  final={vals[-1]:+.3f}  "
              f"first ≥{thr} @ step {first_step}")

    print("\nCentroid approach (cosine with final-step centroid):")
    for t in TRIAD:
        v = np.asarray(centroid_curves[TRIAD_CHARS[t]])
        print(f"  {TRIAD_CHARS[t]}:  init={v[0]:+.3f}  mid={v[len(v)//2]:+.3f}  "
              f"final={v[-1]:+.3f}  first ≥0.6 @ step "
              f"{steps[int(np.argmax(v >= 0.6))] if (v >= 0.6).any() else None}")

    # Save
    out = {
        "steps": steps,
        "pair_curves": pair_curves,
        "centroid_curves": centroid_curves,
        "rank_curves": rank_curves,
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "triad_sequential.json").write_text(json.dumps(out, indent=2))
    print(f"\nStats → {ANALYSIS_DIR/'triad_sequential.json'}")

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        # Plot 1: pairwise cosines over time
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for lab, color in zip(pair_labels, ["C0", "C1", "C2"]):
            ax1.plot(steps, pair_curves[lab], lw=1.6, color=color, label=lab)
        ax1.axhline(0.3, ls="--", color="gray", alpha=0.5, label="0.3 threshold")
        ax1.set_xlabel("training step")
        ax1.set_ylabel("pairwise cosine")
        ax1.set_title(".!? triad — pairwise cosine over training")
        ax1.set_xscale("symlog", linthresh=500)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        for t, color in zip(TRIAD, ["C0", "C1", "C2"]):
            ax2.plot(steps, centroid_curves[TRIAD_CHARS[t]], lw=1.6, color=color,
                     label=f"{TRIAD_CHARS[t]!r}")
        ax2.axhline(0.6, ls="--", color="gray", alpha=0.5, label="0.6 threshold")
        ax2.set_xlabel("training step")
        ax2.set_ylabel("cosine with final-triad centroid")
        ax2.set_title(".!? centroid approach — which token arrives first?")
        ax2.set_xscale("symlog", linthresh=500)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "triad_pairwise_cos.png", dpi=140)
        print(f"Plot → {REPORT_DIR/'triad_pairwise_cos.png'}")

        # Plot 2: rank of each triad member among neighbors of the others
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
        for ax, t in zip(axs, TRIAD):
            for o in TRIAD:
                if o == t:
                    continue
                ax.plot(steps, rank_curves[TRIAD_CHARS[t]][TRIAD_CHARS[o]],
                        label=f"rank of {TRIAD_CHARS[o]!r}", lw=1.6)
            ax.set_xscale("symlog", linthresh=500)
            ax.set_yscale("log")
            ax.set_xlabel("training step")
            ax.set_title(f"Neighbors-of {TRIAD_CHARS[t]!r}")
            ax.grid(True, alpha=0.3)
            ax.legend()
        axs[0].set_ylabel("rank of neighbor (0 = nearest; log)")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "triad_centroid_pull.png", dpi=140)
        print(f"Plot → {REPORT_DIR/'triad_centroid_pull.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
