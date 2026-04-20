"""§4.2 overlay figure — three Exp 2 pilots, three pathologies.

Each pilot trained with topographic regularization on a 10K-step schedule.
Different formulations → different pathologies:
  - exp2_pilot (pilot 1): Gaussian pure attractive, grid_lr_scale=10 → collapse
  - exp2_pilot2C_mse: MSE-simple with target=0 for non-cooccur → escape
  - exp2_pilot2D_eq: equilibrium MSE with max(er, cap·w) target → bimodal

Plots three panels:
  1. Grid spread ratio (pairwise mean distance / init)
  2. Topographic loss magnitude (normalized so all three are on [0, 1])
  3. Fraction of positions actively moving (>0.1 grid units per log interval)

Outputs reports/exp2/pilot_comparison.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports" / "exp2"


def load_steps(run_name: str):
    path = REPO_ROOT / "runs" / run_name / "logs" / "train.jsonl"
    steps = []
    for l in path.read_text().splitlines():
        try:
            d = json.loads(l)
            if d.get("kind") == "step":
                steps.append(d)
        except Exception:
            pass
    return steps


PILOTS = {
    "Gaussian pure attractive\n(pilot 1 · collapse)": {
        "run": "exp2_pilot",
        "color": "C3",
        "final_loss_sign": -1,  # Gaussian loss is negative
    },
    "MSE-simple (target=0 for non-cooccur)\n(pilot 2C · escape)": {
        "run": "exp2_pilot2C_mse",
        "color": "C1",
        "final_loss_sign": 1,
    },
    "Equilibrium MSE (target=max(er, cap·w))\n(pilot 2D · bimodal)": {
        "run": "exp2_pilot2D_eq",
        "color": "C2",
        "final_loss_sign": 1,
    },
}


def main() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))

    for name, spec in PILOTS.items():
        rows = load_steps(spec["run"])
        if not rows:
            print(f"no data for {spec['run']}, skipping")
            continue
        steps = np.asarray([r["step"] for r in rows])
        # Grid spread
        spread = np.asarray([r.get("grid_spread_ratio", float("nan")) for r in rows])
        axs[0].plot(steps, spread, lw=1.8, color=spec["color"], label=name)

        # Topographic loss at the working sigma (topo_loss_raw). For Gaussian
        # pure attractive the loss is negative; for MSE it's positive. Plot
        # |loss| normalized to its absolute max so the curves overlap.
        loss_key = "topo_loss_raw" if any("topo_loss_raw" in r for r in rows) else "topo_loss_sigma1"
        loss = np.asarray([abs(r.get(loss_key, float("nan"))) for r in rows])
        loss_norm = loss / max(np.nanmax(loss), 1e-9)
        axs[1].plot(steps, loss_norm, lw=1.8, color=spec["color"], label=name)

        # Fraction moved
        moved = np.asarray([r.get("frac_positions_moved", float("nan")) for r in rows])
        axs[2].plot(steps, moved, lw=1.8, color=spec["color"], label=name)

    axs[0].set_title("Grid spread ratio\n(mean pairwise distance / init)")
    axs[0].axhline(1.0, color="gray", lw=0.7, ls="--", alpha=0.5)
    axs[0].set_xscale("symlog", linthresh=500)
    axs[0].grid(True, alpha=0.3, which="both")
    axs[0].set_xlabel("training step")
    axs[0].legend(fontsize=8, loc="upper left")

    axs[1].set_title("|Topographic loss| (normalized per pilot)")
    axs[1].set_xscale("symlog", linthresh=500)
    axs[1].grid(True, alpha=0.3, which="both")
    axs[1].set_xlabel("training step")

    axs[2].set_title("frac positions moved >0.1 / log interval")
    axs[2].set_xscale("symlog", linthresh=500)
    axs[2].grid(True, alpha=0.3, which="both")
    axs[2].set_xlabel("training step")

    fig.suptitle(
        "Three Gaussian-kernel topographic formulations, three pathologies. "
        "Each converges to a frozen state (right-most points of panel 3 → 0), "
        "but reaches it via different failure modes.",
        fontsize=10, y=1.05,
    )
    fig.tight_layout()
    out = REPORT_DIR / "pilot_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Plot → {out}")


if __name__ == "__main__":
    main()
