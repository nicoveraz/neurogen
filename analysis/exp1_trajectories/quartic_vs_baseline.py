"""Compare embedding geometry of quartic-window vs baseline training.

Question: does the quartic-window architectural constraint produce emergent
topographic-like organization in token embeddings, without an explicit
topographic regularizer?

Comparisons (all on final w_final, plus trajectory comparisons over snapshots):

  1. Co-occurrence × final-embedding-similarity correlation
     - Spearman rho(cooccur[i,j], cos(w_final[i], w_final[j]))
     - If quartic has higher rho, its geometry reflects co-occurrence structure
       more strongly — i.e., emergent topographic organization.

  2. Within-class coherence (repeat §3.1 for both runs)
     - Does quartic produce tighter final clusters for byte classes?
     - Does it produce phase-structure (peak-then-decline) on the same
       classes as baseline, or on different ones?

  3. Tracked-pair distances in final embedding space
     - Space-letter pairs, .!? triad, digit cluster, brackets.
     - Does quartic put them at positions more aligned with co-occurrence
       than baseline does?

  4. 2D PCA of final embeddings
     - Visual: do classes cluster spatially in quartic? How does the layout
       compare?

Outputs:
  analysis/exp1_trajectories/quartic_vs_baseline.json
  reports/exp_quartic/quartic_comparison.png
"""

from __future__ import annotations

import json
import string
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "runs" / "exp0_baseline"
QUARTIC_DIR = REPO_ROOT / "runs" / "exp_quartic"
COOCCUR_PATH = REPO_ROOT / "runs" / "exp2_cooccur" / "cooccur_w5.npy"
OUT_ANALYSIS = REPO_ROOT / "analysis" / "exp1_trajectories"
OUT_REPORT = REPO_ROOT / "reports" / "exp_quartic"


def byte_classes() -> dict[str, list[int]]:
    return {
        "digits":         [ord(c) for c in string.digits],
        "sentence_punct": [ord(c) for c in ".!?"],
        "uppercase":      [ord(c) for c in string.ascii_uppercase],
        "lowercase":      [ord(c) for c in string.ascii_lowercase],
        "vowels":         [ord(c) for c in "aeiou"],
        "consonants":     [ord(c) for c in "bcdfghjklmnpqrstvwxyz"],
        "clause_punct":   [ord(c) for c in ",;:"],
        "whitespace":     [ord(c) for c in " \t\n\r"],
    }


def cosine_matrix(A: np.ndarray) -> np.ndarray:
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    return An @ An.T


def within_class_mean(M: np.ndarray, idxs: list[int]) -> float:
    if len(idxs) < 2:
        return float("nan")
    C = M[np.ix_(idxs, idxs)]
    iu = np.triu_indices(len(idxs), k=1)
    return float(C[iu].mean())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra.astype(float), rb.astype(float))[0, 1])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def analyze_run(run_dir: Path, cooccur: np.ndarray) -> dict:
    snaps = sorted((run_dir / "snapshots").glob("wte_step_*.npy"))
    steps = [int(p.stem.split("_")[-1]) for p in snaps]
    W_final = np.load(snaps[-1])
    V = W_final.shape[0]
    # Final pairwise cosine matrix
    C_final = cosine_matrix(W_final)
    iu = np.triu_indices(V, k=1)
    cos_pairs = C_final[iu]
    cooccur_pairs = cooccur[iu]

    # Overall cooccur × cosine relationship
    rho = spearman(cooccur_pairs, cos_pairs)
    r = pearson(cooccur_pairs, cos_pairs)

    # Restrict to non-trivial co-occurrence (cooccur > median nonzero)
    nz = cooccur_pairs > 0
    cooccur_nz = cooccur_pairs[nz]
    cos_nz = cos_pairs[nz]
    rho_nz = spearman(cooccur_nz, cos_nz)
    r_nz = pearson(cooccur_nz, cos_nz)

    # Within-class coherence at final
    classes = byte_classes()
    within_class = {name: within_class_mean(C_final, idxs)
                    for name, idxs in classes.items()}

    # Tracked pairs
    tracked = {
        "' '-'e'": (ord(" "), ord("e")),
        "' '-'t'": (ord(" "), ord("t")),
        "' '-'a'": (ord(" "), ord("a")),
        "'.'-'!'": (ord("."), ord("!")),
        "'.'-'?'": (ord("."), ord("?")),
        "'0'-'5'": (ord("0"), ord("5")),
        "'(-')'": (ord("("), ord(")")),
    }
    tracked_cos = {k: float(C_final[i, j]) for k, (i, j) in tracked.items()}

    # Trajectory: within-class coherence at each snapshot (for digits +
    # sentence_punct as the canonical cases)
    trajectory = {}
    for name in ["digits", "sentence_punct", "uppercase", "lowercase",
                 "clause_punct", "vowels"]:
        idxs = classes[name]
        traj = []
        for p in snaps:
            W = np.load(p)
            Cm = cosine_matrix(W)
            traj.append(within_class_mean(Cm, idxs))
        trajectory[name] = traj

    # Digit ordinal correlation trajectory
    digits = classes["digits"]
    digit_ordinal_r = []
    for p in snaps:
        W = np.load(p)
        Cd = cosine_matrix(W[digits])
        iu_d = np.triu_indices(10, k=1)
        cos_d = Cd[iu_d]
        numdist_d = np.abs(np.subtract.outer(np.arange(10), np.arange(10)))[iu_d]
        digit_ordinal_r.append(pearson(cos_d, numdist_d.astype(float)))

    return {
        "steps": steps,
        "cooccur_cos_spearman_all": round(rho, 4),
        "cooccur_cos_pearson_all": round(r, 4),
        "cooccur_cos_spearman_nz": round(rho_nz, 4),
        "cooccur_cos_pearson_nz": round(r_nz, 4),
        "final_within_class": {k: round(v, 4) for k, v in within_class.items()},
        "tracked_pair_cos_final": {k: round(v, 4) for k, v in tracked_cos.items()},
        "within_class_trajectory": trajectory,
        "digit_ordinal_r_trajectory": digit_ordinal_r,
    }


def main() -> None:
    if not BASELINE_DIR.exists():
        raise FileNotFoundError(BASELINE_DIR)
    if not QUARTIC_DIR.exists():
        print(f"Quartic run not yet available at {QUARTIC_DIR}; running baseline only")
    if not COOCCUR_PATH.exists():
        raise FileNotFoundError(COOCCUR_PATH)

    cooccur = np.load(COOCCUR_PATH)

    print("=== Baseline analysis ===")
    baseline = analyze_run(BASELINE_DIR, cooccur)
    print_results("baseline", baseline)

    if QUARTIC_DIR.exists() and any((QUARTIC_DIR / "snapshots").glob("wte_step_*.npy")):
        print("\n=== Quartic analysis ===")
        quartic = analyze_run(QUARTIC_DIR, cooccur)
        print_results("quartic", quartic)

        print("\n=== Comparison ===")
        print_comparison(baseline, quartic)

        # Save side-by-side
        combined = {"baseline": baseline, "quartic": quartic}
        OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
        (OUT_ANALYSIS / "quartic_vs_baseline.json").write_text(json.dumps(combined, indent=2))
        print(f"\nSaved → {OUT_ANALYSIS/'quartic_vs_baseline.json'}")

        # Plot
        plot_comparison(baseline, quartic)
    else:
        OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
        (OUT_ANALYSIS / "baseline_geometry.json").write_text(json.dumps(baseline, indent=2))
        print(f"\n(Quartic snapshots not available yet; baseline saved for reference)")


def print_results(label: str, r: dict) -> None:
    print(f"  [{label}] cooccur × final cos correlation:")
    print(f"     Spearman (all pairs): {r['cooccur_cos_spearman_all']:+.3f}")
    print(f"     Spearman (non-zero cooccur only): {r['cooccur_cos_spearman_nz']:+.3f}")
    print(f"  [{label}] final within-class cosine:")
    for name, v in r["final_within_class"].items():
        print(f"     {name:<18} {v:+.4f}")
    print(f"  [{label}] tracked pair cosines (final):")
    for name, v in r["tracked_pair_cos_final"].items():
        print(f"     {name:<12} {v:+.4f}")
    dor = r["digit_ordinal_r_trajectory"]
    print(f"  [{label}] digit ordinal r: init={dor[0]:+.3f}  mid={dor[len(dor)//2]:+.3f}  final={dor[-1]:+.3f}  min={min(dor):+.3f}")


def print_comparison(b: dict, q: dict) -> None:
    print(f"{'metric':<45} {'baseline':>12} {'quartic':>12} {'delta':>10}")
    print("-" * 82)
    m = "cooccur_cos_spearman_nz"
    print(f"{'cooccur×cos Spearman (nonzero pairs)':<45} {b[m]:>12.4f} {q[m]:>12.4f} {q[m]-b[m]:>+10.4f}")
    m = "cooccur_cos_pearson_nz"
    print(f"{'cooccur×cos Pearson (nonzero pairs)':<45} {b[m]:>12.4f} {q[m]:>12.4f} {q[m]-b[m]:>+10.4f}")
    for name in b["final_within_class"]:
        bw = b["final_within_class"][name]
        qw = q["final_within_class"][name]
        print(f"{'within-class final cos · ' + name:<45} {bw:>12.4f} {qw:>12.4f} {qw-bw:>+10.4f}")

    bor = b["digit_ordinal_r_trajectory"]
    qor = q["digit_ordinal_r_trajectory"]
    print(f"{'digit ordinal r (final)':<45} {bor[-1]:>12.4f} {qor[-1]:>12.4f} {qor[-1]-bor[-1]:>+10.4f}")
    print(f"{'digit ordinal r (most negative)':<45} {min(bor):>12.4f} {min(qor):>12.4f} {min(qor)-min(bor):>+10.4f}")


def plot_comparison(b: dict, q: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    OUT_REPORT.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))

    # Within-class coherence trajectory overlay for digits + sentence_punct
    for name, color in [("digits", "C0"), ("sentence_punct", "C3"), ("uppercase", "C2"), ("clause_punct", "C1")]:
        axs[0].plot(b["steps"], b["within_class_trajectory"][name],
                    color=color, lw=1.2, ls="-",
                    label=f"{name} · baseline")
        axs[0].plot(q["steps"], q["within_class_trajectory"][name],
                    color=color, lw=1.8, ls="--",
                    label=f"{name} · quartic")
    axs[0].set_xscale("symlog", linthresh=500)
    axs[0].set_xlabel("training step")
    axs[0].set_ylabel("mean pairwise cosine (within class)")
    axs[0].set_title("Within-class coherence — baseline vs quartic")
    axs[0].grid(True, alpha=0.3, which="both")
    axs[0].legend(fontsize=7, loc="upper left", ncol=2)

    # Digit ordinal r trajectory
    axs[1].plot(b["steps"], b["digit_ordinal_r_trajectory"],
                color="C0", lw=1.4, label="baseline")
    axs[1].plot(q["steps"], q["digit_ordinal_r_trajectory"],
                color="C3", lw=1.8, label="quartic")
    axs[1].axhline(0, color="black", lw=0.5)
    axs[1].set_xscale("symlog", linthresh=500)
    axs[1].set_xlabel("training step")
    axs[1].set_ylabel("r(cos, |i−j|)")
    axs[1].set_title("Digit ordinal correlation")
    axs[1].grid(True, alpha=0.3, which="both")
    axs[1].legend()

    fig.tight_layout()
    out = OUT_REPORT / "quartic_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Plot → {out}")


if __name__ == "__main__":
    main()
