"""Exp 1 — consecutive-snapshot CKA (linear CKA on embedding matrices).

Trajectory smoothness: how similar is the token embedding layout at step t
to step t+1? Falling CKA = the embedding is actively reorganizing; rising
CKA = the embedding has stabilized and changes are small.

Computed on positions, not displacements (per methodological principle
§11). Linear CKA(A, B) = ‖Aᵀ B‖_F² / (‖Aᵀ A‖_F · ‖Bᵀ B‖_F).

Outputs:
  analysis/exp1_trajectories/consecutive_cka.json
  reports/exp1/consecutive_cka.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "runs" / "exp0_baseline"
ANALYSIS_DIR = REPO_ROOT / "analysis" / "exp1_trajectories"
REPORT_DIR = REPO_ROOT / "reports" / "exp1"


def center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    """Linear CKA between two embedding matrices A, B of shape (V, D).
    Scale-invariant, row-centered."""
    A = center(A)
    B = center(B)
    hsic_ab = float(np.linalg.norm(A.T @ B) ** 2)
    hsic_aa = float(np.linalg.norm(A.T @ A))
    hsic_bb = float(np.linalg.norm(B.T @ B))
    return hsic_ab / (hsic_aa * hsic_bb + 1e-12)


def cka_anchored(A_ref: np.ndarray, A_t: np.ndarray) -> float:
    """CKA of snapshot A_t to a fixed reference A_ref."""
    return linear_cka(A_ref, A_t)


def main() -> None:
    snaps = sorted((RUN_DIR / "snapshots").glob("wte_step_*.npy"))
    steps = [int(p.stem.split("_")[-1]) for p in snaps]
    print(f"Loading {len(snaps)} snapshots…")

    Ws = [np.load(p) for p in snaps]
    W_final = Ws[-1]
    W_init = Ws[0]

    # Consecutive CKA
    cka_consec = [1.0]  # self-CKA at step 0
    for i in range(1, len(Ws)):
        cka_consec.append(linear_cka(Ws[i - 1], Ws[i]))

    # CKA vs final
    cka_to_final = [linear_cka(W, W_final) for W in Ws]
    # CKA vs init
    cka_to_init = [linear_cka(W, W_init) for W in Ws]

    print(f"{'step':>8}  {'CKA(prev,t)':>12}  {'CKA(t, final)':>14}  {'CKA(t, init)':>12}")
    pick = {0, 500, 2000, 5000, 10000, 20000, 47000, 70000, 100000}
    for s, cc, cf, ci in zip(steps, cka_consec, cka_to_final, cka_to_init):
        if s in pick:
            print(f"{s:>8d}  {cc:>12.4f}  {cf:>14.4f}  {ci:>12.4f}")

    out = {
        "steps": steps,
        "cka_consecutive": cka_consec,
        "cka_to_final": cka_to_final,
        "cka_to_init": cka_to_init,
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "consecutive_cka.json").write_text(json.dumps(out, indent=2))
    print(f"\nStats → {ANALYSIS_DIR/'consecutive_cka.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax1.plot(steps, cka_consec, lw=1.6, color="C0", label="CKA(t−1, t) — smoothness")
        ax1.set_xscale("symlog", linthresh=500)
        ax1.set_xlabel("training step")
        ax1.set_ylabel("consecutive CKA")
        ax1.set_title("Trajectory smoothness (consecutive-snapshot CKA)")
        ax1.grid(True, alpha=0.3, which="both")
        ax1.legend()

        ax2.plot(steps, cka_to_final, lw=1.6, color="C2", label="CKA(t, final)")
        ax2.plot(steps, cka_to_init, lw=1.6, color="C3", label="CKA(t, init)")
        ax2.set_xscale("symlog", linthresh=500)
        ax2.set_xlabel("training step")
        ax2.set_ylabel("CKA")
        ax2.set_title("CKA vs fixed reference — when has the layout 'arrived'?")
        ax2.grid(True, alpha=0.3, which="both")
        ax2.legend()

        fig.tight_layout()
        fig.savefig(REPORT_DIR / "consecutive_cka.png", dpi=140)
        print(f"Plot → {REPORT_DIR/'consecutive_cka.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
