"""§4.3 illustration — |∇L| vs distance for three loss formulations.

Visualizes why kernel-based topographic losses have vanishing gradient at
both d → 0 and d → ∞, while distance-based losses have monotone gradient.

For a pair (i, j) with target similarity T_s (or target distance T_d):
  - Gaussian-pure-attractive   L_1 = -w · K(d)
    |∂L_1/∂d| ∝ w · K(d) · d / σ²         (peaks at d = σ, vanishes at both ends)
  - Gaussian-MSE               L_2 = (K(d) - T_s)²
    |∂L_2/∂d| ∝ |K(d) - T_s| · K(d) · d / σ²   (vanishes at d→0 and d→∞)
  - Distance-MSE               L_3 = (d - T_d)²
    |∂L_3/∂d| = 2|d - T_d|                (monotone; zero only at equilibrium)

σ = 3, T_s = 0.4 (arbitrary moderate), T_d = 2.77 (= σ·√(-2 ln T_s) so
the Gaussian-MSE equilibrium matches).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports" / "exp2"


def main() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable")
        return

    sigma = 3.0
    Ts = 0.4
    Td = sigma * np.sqrt(-2 * np.log(Ts))   # ≈ 2.77

    d = np.linspace(0.001, 12.0, 400)

    # Absolute gradient magnitude wrt d for each loss
    K = np.exp(-d ** 2 / (2 * sigma ** 2))
    # Gaussian pure attractive: |grad| = K · d / σ²  (w set to 1)
    g1 = K * d / sigma ** 2
    # Gaussian MSE: |grad| = |K - Ts| · K · d / σ²
    g2 = np.abs(K - Ts) * K * d / sigma ** 2
    # Distance MSE: |grad| = 2|d - Td|
    g3 = 2 * np.abs(d - Td)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(d, g1, label=r"$L=-K(d)$ (Gaussian attractive)", color="C3", lw=1.8)
    ax1.plot(d, g2, label=rf"$L=(K(d)-T_s)^2$ (Gaussian-MSE, $T_s={Ts}$)",
             color="C1", lw=1.8)
    ax1.axvline(Td, color="gray", lw=0.7, ls="--",
                label=f"target distance d* ≈ {Td:.2f}")
    ax1.set_xlabel("pairwise distance d")
    ax1.set_ylabel(r"|gradient of loss wrt d|")
    ax1.set_title("Kernel-based losses: gradient vanishes at both tails\n"
                  "(d → 0: (pos_i − pos_j) → 0;  d → ∞: K(d) → 0)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    ax2.plot(d, g3, label=rf"$L=(d-T_d)^2$ (distance-MSE, $T_d={Td:.2f}$)",
             color="C2", lw=1.8)
    ax2.axvline(Td, color="gray", lw=0.7, ls="--", label=f"target distance d* = {Td:.2f}")
    ax2.set_xlabel("pairwise distance d")
    ax2.set_ylabel(r"|gradient of loss wrt d|")
    ax2.set_title("Distance-based loss: gradient magnitude linear in error\n"
                  "Monotone attraction toward d*, no vanishing basin")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "Why Gaussian-kernel topographic losses fail: gradient vanishes at both d→0 and d→∞\n"
        "Distance-based formulation (right) has globally well-behaved gradient.",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    out = REPORT_DIR / "gradient_envelope.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Plot → {out}")


if __name__ == "__main__":
    main()
