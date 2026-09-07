"""Every figure in this repo, from committed data, in one style.

Before this existed, eight of the nine committed charts had no generating
script under their committed name -- analyze_all.py wrote figures/fig1_*.svg
while the repo shipped charts/learning_curves.svg -- so no figure could be
regenerated or checked against the data it claimed to show.

Each figure writes SVG and PNG (for the README) and PDF (for LaTeX, which
cannot take SVG). Any figure showing val_bpb states its evaluation harness in
the title, because this repo has two and never mixes them (README, "Which
evaluation harness a number comes from").

Usage:
    uv run python analyze_figures.py           # all figures
    uv run python analyze_figures.py --only shock
"""
import argparse
import glob
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("gradient_results")
VALID = Path("validation_results")
OUT = Path("charts")
SEEDS = [42, 137, 256, 789, 1337]

# One colour per arm, used identically in every figure.
C = {"A": "#4a5568", "B": "#2b6cb0", "F": "#c05621", "R": "#2f855a",
     "accent": "#97266d", "muted": "#a0aec0"}
LABEL = {"A": "A: full attention", "B": "B: quartic throughout",
         "F": "F: hard switch", "R": "R: ramped release"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150,
})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("svg", "png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote charts/{name}.{{svg,png,pdf}}")


def _fixed():
    return json.load(open(RESULTS / "fixed_eval_remeasure.json"))["runs"]


# ---------------------------------------------------------------- removal
def fig_removal():
    """A/B/F/R training curves for one seed, plus paired endpoints."""
    d = json.load(open(RESULTS / "exp7_curriculum_sw10000_5seed.json"))["runs"]
    r = json.load(open(RESULTS / "exp7_curriculum_sw10000_ramp5seed.json"))["runs"]
    seed = 137                       # completed in every arm
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0),
                                  gridspec_kw={"width_ratios": [1.55, 1]})
    for key, arm, src in (("A_full", "A", d), ("B_quartic", "B", d),
                          ("F_switch", "F", d), ("F_switch", "R", r)):
        run = src.get(f"{key}_s{seed}")
        if not run or "curve" not in run:
            continue
        xs = [p[0] for p in run["curve"]]; ys = [p[1] for p in run["curve"]]
        ax.plot(xs, ys, color=C[arm], lw=1.7, label=LABEL[arm])
    ax.axvline(10000, color=C["muted"], ls="--", lw=1.1)
    # The hard switch spikes far off this scale (3.29 bpb at seed 137) and
    # recovers within ~30 steps. Clipping keeps the four curves legible; the
    # spike itself is the subject of the switch_shock figure.
    ax.set_ylim(0.85, 1.35)
    ax.annotate("release at step 10,000\nF spikes to 3.29 bpb here,\nrecovered by step 10,500",
                xy=(10000, 1.20), xytext=(11600, 1.24), fontsize=7.5,
                color=C["muted"], ha="left",
                arrowprops=dict(arrowstyle="->", color=C["muted"], lw=1))
    ax.set_xlabel("training step"); ax.set_ylabel("validation bpb")
    ax.set_title(f"Removal at the halfway point, seed {seed}\nlegacy harness (training-time curves)")
    ax.legend(loc="upper right")

    fx = _fixed()
    g = lambda k: fx[k]["fixed_bpb"]
    arms = {"A": {s: g(f"model_baseline_{s}_fixedeval.pt") for s in SEEDS},
            "B": {s: g(f"model_window_power_4.0_{s}_fixedeval.pt") for s in SEEDS},
            "R": {s: g(f"model_F_switch_{s}_sw10000_ramp5seed.pt") for s in SEEDS}}
    for i, s in enumerate(SEEDS):
        for arm in ("A", "B", "R"):
            ax2.plot(i, arms[arm][s], "o", color=C[arm], ms=6,
                     label=LABEL[arm] if i == 0 else None)
        ax2.plot([i, i], [arms["A"][s], arms["R"][s]], color=C["muted"], lw=0.8, zorder=0)
    ax2.set_xticks(range(len(SEEDS))); ax2.set_xticklabels([str(s) for s in SEEDS])
    ax2.set_xlabel("seed"); ax2.set_ylabel("final validation bpb")
    ax2.set_title("Paired endpoints, all five seeds\n(fixed harness)")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "removal_experiment")


# ------------------------------------------------------------------ shock
def fig_shock():
    """Loss, pre-clip grad norm and max Adam update around the switch."""
    def trace(tag, seed=256):
        p = RESULTS / f"exp7_switch_trace_s{seed}_sw10000_{tag}.json"
        return json.load(open(p))["trace"] if p.exists() else None
    # The hard-switch trace is the from-scratch 5-seed run; the ramp resumes
    # from that run's own pre-switch checkpoint. Verified interchangeable: the
    # resumed kernel_control trace reproduces the from-scratch one to 1e-6 at
    # all 501 traced steps, which is the log's rounding precision.
    hard, ramp = trace("5seed"), trace("ramp500")
    if hard is None or ramp is None:
        print("  shock: traces missing, skipped"); return
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    fields = [("loss", "training loss", None),
              ("grad_norm_preclip", "gradient norm (pre-clip)", 1.0),
              ("max_adam_update", "max Adam update", None)]
    for ax, (key, lab, hline) in zip(axes, fields):
        for tr, arm, name in ((hard, "F", "hard switch"), (ramp, "R", "500-step ramp")):
            xs = [p["step"] - 10000 for p in tr]
            ax.plot(xs, [p[key] for p in tr], color=C[arm], lw=1.3, label=name)
        ax.axvline(0, color=C["muted"], ls="--", lw=1.1)
        if hline:
            ax.axhline(hline, color=C["accent"], ls=":", lw=1.1)
            ax.text(260, hline * 1.06, "clip threshold", color=C["accent"], fontsize=7)
        ax.set_xlabel("steps from release"); ax.set_ylabel(lab)
        if key == "max_adam_update":
            ax.set_yscale("log")
        ax.set_xlim(-100, 500)
    axes[0].legend(loc="upper right")
    fig.suptitle("The switch is a shock; the ramp removes it\n"
                 "seed 256 \u2014 the ramp resumes from the hard switch's own pre-switch state",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "switch_shock")


# --------------------------------------------------------------- ablation
def fig_ablation():
    """Per-layer window ablation, seed 42 (legacy harness, single seed)."""
    import analyze_ablation as ab
    rows = []
    for arch, windows, desc in ab.CONFIGS:
        v = ab.final_bpb(arch, 42)
        if v is not None:
            rows.append((desc, windows, v))
    if not rows:
        print("  ablation: no data, skipped"); return
    base = next(v for d, w, v in rows if "reference" in d and "full" in d)
    rows.sort(key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    pct = [(base - v) / base * 100 for _, _, v in rows]
    cols = [C["F"] if p < 0 else (C["B"] if "quartic" in d else C["R"])
            for p, (d, _, _) in zip(pct, rows)]
    ax.barh(range(len(rows)), pct, color=cols)
    ax.axvline(0, color="#2d3748", lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{d}\n{w}" for d, w, _ in rows])
    ax.set_xlabel("gain over full-attention baseline (%)")
    ax.set_title("Where the effect lives: per-layer window ablation\n"
                 "seed 42 only, legacy harness")
    for i, p in enumerate(pct):
        ax.text(p + (0.05 if p >= 0 else -0.05), i, f"{p:+.2f}%",
                va="center", ha="left" if p >= 0 else "right", fontsize=8)
    ax.set_xlim(min(pct) - 0.5, max(pct) + 0.5)
    fig.tight_layout()
    save(fig, "layer_ablation")


# ------------------------------------------------------------ grad rank
def fig_gradrank():
    """Gradient covariance effective rank vs window size."""
    d = json.load(open(RESULTS / "mechanism_disambiguation.json"))["exp5"]
    d = sorted(d, key=lambda r: r["window"])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot([r["window"] for r in d], [r["eff_rank"] for r in d],
            "o-", color=C["B"], lw=1.8, ms=6)
    ax.set_xscale("log", base=2)
    ax.set_xticks([r["window"] for r in d])
    ax.set_xticklabels([str(r["window"]) for r in d])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("attention window at layer 0 (tokens)")
    ax.set_ylabel("effective rank of Q/K gradient covariance")
    ax.set_title("The constraint makes early updates coherent\n"
                 "50 gradient samples, frozen baseline checkpoint")
    ax2 = ax.twinx()
    ax2.plot([r["window"] for r in d], [r["var_top1"] * 100 for r in d],
             "s--", color=C["F"], lw=1.3, ms=5, alpha=0.85)
    ax2.set_ylabel("variance in top component (%)", color=C["F"])
    ax2.tick_params(axis="y", colors=C["F"]); ax2.grid(False)
    fig.tight_layout()
    save(fig, "gradient_rank")


# ------------------------------------------------------------------ 125M
def fig_125m():
    """Per-seed 125M gap at 20k steps (legacy harness; never re-measured)."""
    import analyze_125m as a
    rows = []
    for s in SEEDS:
        try:
            b = json.load(open(f"results_125m/baseline_s{s}.json"))
            q = json.load(open(f"results_125m/window_power_4.0_s{s}.json"))
        except FileNotFoundError:
            continue
        at = lambda d, step: next((p["val_bpb"] for p in d["curve"]
                                   if p["step"] == step), None)
        bb, qq = at(b, 20000), at(q, 20000)
        if bb and qq:
            rows.append((s, bb, qq, (bb - qq) / bb * 100))
    if not rows:
        print("  125m: no data, skipped"); return
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    cols = [C["F"] if g < 0 else C["B"] for *_, g in rows]
    ax.bar(range(len(rows)), [g for *_, g in rows], color=cols)
    ax.axhline(0, color="#2d3748", lw=1)
    ax.axhline(st.mean([g for *_, g in rows]), color=C["accent"], ls="--", lw=1.2,
               label=f"mean {st.mean([g for *_, g in rows]):+.2f}%")
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([str(r[0]) for r in rows])
    ax.set_xlabel("seed"); ax.set_ylabel("quartic gain over baseline (%)")
    ax.set_title("125M at 20k steps: suggestive, not converged\n"
                 "legacy harness; one seed negative")
    ax.legend()
    fig.tight_layout()
    save(fig, "125m_per_seed_gap")


# -------------------------------------------------------- window schedule
def fig_schedule():
    """The depth-wise window schedule itself. Concept figure, no measurements."""
    import windows
    T, L = 256, 4
    modes = [("power_4.0", "quartic ($\\gamma=4$), used throughout", C["B"]),
             ("quadratic", "quadratic ($\\gamma=2$)", C["R"])]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for mode, lab, col in modes:
        w = [windows.compute_window_size(i, L, T, mode) for i in range(L)]
        ax.plot(range(L), w, "o-", color=col, lw=1.8, ms=7, label=lab)
        if mode == "power_4.0":
            for i, v in enumerate(w):
                ax.annotate(str(v), (i, v), textcoords="offset points",
                            xytext=(6, -12), ha="left", fontsize=8, color=col)
    ax.axhline(T, color=C["muted"], ls="--", lw=1.1)
    ax.text(L - 1.05, T * 1.04, f"full attention ({T} tokens)", fontsize=8,
            color=C["muted"], ha="right")
    ax.set_xticks(range(L)); ax.set_xticklabels([f"layer {i}" for i in range(L)])
    ax.set_ylabel("attention window (tokens)")
    ax.set_ylim(0, T * 1.12)
    ax.set_title("Depth-wise window schedule at depth 4, sequence length 256\n"
                 "early layers see a short span, the last sees everything")
    ax.legend(loc="center right")
    fig.tight_layout()
    save(fig, "window_schedule")


# ------------------------------------------------------ entropy persistence
def fig_entropy():
    """Per-layer attention entropy at 20k and 100k steps."""
    try:
        d20 = json.load(open(RESULTS / "attention_entropy.json"))
        d100 = json.load(open(RESULTS / "attention_entropy_100k.json"))
    except FileNotFoundError:
        print("  entropy: data missing, skipped"); return
    layers = sorted(d100["Baseline"], key=int)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = range(len(layers))
    w = 0.2
    series = [(d20, "Baseline", "baseline, 20k", C["A"], -1.5),
              (d20, "Quartic", "quartic, 20k", C["B"], -0.5),
              (d100, "Baseline", "baseline, 100k", C["A"], 0.5),
              (d100, "Quartic", "quartic, 100k", C["B"], 1.5)]
    for src, arm, lab, col, off in series:
        if arm not in src:
            continue
        ys = [src[arm][l]["mean_entropy"] for l in layers]
        ax.bar([i + off * w for i in x], ys, w, label=lab, color=col,
               alpha=1.0 if "100k" in lab else 0.55,
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(list(x)); ax.set_xticklabels([f"layer {l}" for l in layers])
    ax.set_ylabel("mean attention entropy (nats)")
    ax.set_title("What the constraint leaves behind\n"
                 "early-layer entropy stays low at 5$\\times$ longer training")
    ax.legend(ncol=2, fontsize=7.5)
    fig.tight_layout()
    save(fig, "attention_entropy_persistence")


FIGURES = {"schedule": fig_schedule, "removal": fig_removal, "shock": fig_shock,
           "entropy": fig_entropy, "ablation": fig_ablation,
           "gradrank": fig_gradrank, "125m": fig_125m}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=sorted(FIGURES))
    args = ap.parse_args()
    for name, fn in FIGURES.items():
        if args.only and name != args.only:
            continue
        print(f"{name}:")
        fn()


if __name__ == "__main__":
    main()
