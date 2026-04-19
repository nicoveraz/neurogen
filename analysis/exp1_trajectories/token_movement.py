"""Exp 1 — per-token movement magnitude over training.

For each token, compute ‖w_{t+1} − w_t‖ at each checkpoint transition.
This is the per-step velocity.

Note: this is a displacement-based quantity. Per the methodological
principle (Exp 0 §11), the interpretation must account for initialization
variance. To partially correct: also normalize each token's velocity by its
own mean velocity over training. The resulting curve shows when each token
moved relatively more or less than its average — finding its "active" period.

Also produce summary statistics:
  - Per-token total path length ∑‖Δ_t‖
  - Per-token stabilization step: first step after which remaining velocity
    is < 20% of peak velocity

Finally: contrast movement curves for "common" bytes (ASCII letters, space,
digits, common punct) vs rare bytes (high-byte UTF-8, control chars).

Outputs:
  analysis/exp1_trajectories/token_movement.json
  reports/exp1/token_movement.png
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


COMMON_BYTES = set(
    [ord(c) for c in string.ascii_letters + string.digits]
    + [ord(c) for c in " \n\t.,!?;:'\"()"]
)


def main() -> None:
    snaps = sorted((RUN_DIR / "snapshots").glob("wte_step_*.npy"))
    steps = np.asarray([int(p.stem.split("_")[-1]) for p in snaps])
    Ws = np.stack([np.load(p) for p in snaps])    # (T, V, D)
    T, V, D = Ws.shape

    # Per-step velocity: ‖w_{t+1} − w_t‖ per token
    delta = np.diff(Ws, axis=0)                    # (T-1, V, D)
    vel = np.linalg.norm(delta, axis=2)            # (T-1, V)
    transition_steps = steps[1:]                   # step at the END of each diff

    # Per-token total path length
    path_len = vel.sum(axis=0)                     # (V,)
    # Per-token peak velocity and stabilization step
    peak_v = vel.max(axis=0)                       # (V,)
    stabilization = np.full(V, -1, dtype=int)
    for v_idx in range(V):
        # first time after which vel stays < 20% of peak for the remainder
        threshold = 0.2 * peak_v[v_idx]
        below = vel[:, v_idx] < threshold
        # find first k such that below[k:] is all True
        for k in range(len(below)):
            if below[k:].all():
                stabilization[v_idx] = int(transition_steps[k])
                break
        if stabilization[v_idx] == -1:
            stabilization[v_idx] = int(transition_steps[-1])

    # Partition into common vs rare byte groups
    common_mask = np.array([b in COMMON_BYTES for b in range(256)])
    rare_mask = ~common_mask

    # Summary stats
    print(f"per-token path length:   "
          f"min={path_len.min():.2f}  mean={path_len.mean():.2f}  "
          f"max={path_len.max():.2f}")
    print(f"stabilization step:      "
          f"common bytes median = {int(np.median(stabilization[common_mask]))}  "
          f"rare bytes median = {int(np.median(stabilization[rare_mask]))}")
    print(f"total path (common) = {path_len[common_mask].sum():.1f}  "
          f"total path (rare) = {path_len[rare_mask].sum():.1f}")

    # Aggregate velocity curves: median per group across transition steps
    vel_common_med = np.median(vel[:, common_mask], axis=1)
    vel_rare_med = np.median(vel[:, rare_mask], axis=1)
    vel_common_q = np.quantile(vel[:, common_mask], [0.25, 0.75], axis=1)
    vel_rare_q = np.quantile(vel[:, rare_mask], [0.25, 0.75], axis=1)

    out = {
        "transition_steps": transition_steps.tolist(),
        "path_length": path_len.tolist(),
        "peak_velocity": peak_v.tolist(),
        "stabilization_step": stabilization.tolist(),
        "vel_common_median": vel_common_med.tolist(),
        "vel_rare_median": vel_rare_med.tolist(),
        "n_common": int(common_mask.sum()),
        "n_rare": int(rare_mask.sum()),
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "token_movement.json").write_text(json.dumps(out, indent=2))
    print(f"\nStats → {ANALYSIS_DIR/'token_movement.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        # Left: median velocity common vs rare, with IQR bands
        ax1.plot(transition_steps, vel_common_med, lw=1.8, color="C0",
                 label=f"common bytes (n={common_mask.sum()}) — median")
        ax1.fill_between(transition_steps, vel_common_q[0], vel_common_q[1],
                         color="C0", alpha=0.2, label="common IQR")
        ax1.plot(transition_steps, vel_rare_med, lw=1.8, color="C3",
                 label=f"rare bytes (n={rare_mask.sum()}) — median")
        ax1.fill_between(transition_steps, vel_rare_q[0], vel_rare_q[1],
                         color="C3", alpha=0.2, label="rare IQR")
        ax1.set_xscale("symlog", linthresh=500)
        ax1.set_xlabel("training step")
        ax1.set_ylabel("per-step ‖Δw‖ (token velocity)")
        ax1.set_title("Token velocity over training — common vs rare bytes")
        ax1.grid(True, alpha=0.3, which="both")
        ax1.legend(fontsize=9)

        # Right: stabilization-step histogram common vs rare
        bins = np.geomspace(500, 100000, 25)
        ax2.hist(stabilization[common_mask], bins=bins, alpha=0.6,
                 label=f"common (n={common_mask.sum()})", color="C0")
        ax2.hist(stabilization[rare_mask], bins=bins, alpha=0.6,
                 label=f"rare (n={rare_mask.sum()})", color="C3")
        ax2.set_xscale("log")
        ax2.set_xlabel("stabilization step (first where vel < 20% of peak, sustained)")
        ax2.set_ylabel("count")
        ax2.set_title("When does each token stabilize?")
        ax2.grid(True, alpha=0.3, which="both")
        ax2.legend()

        fig.tight_layout()
        fig.savefig(REPORT_DIR / "token_movement.png", dpi=140)
        print(f"Plot → {REPORT_DIR/'token_movement.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
