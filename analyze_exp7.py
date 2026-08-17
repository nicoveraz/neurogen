"""Report the window-removal sweep (Exp 7) and the switch-shock diagnosis.

Reads the files experiment_mechanism.py writes and emits (a) the per-seed
paired table with signs visible, (b) the pre-registered verdicts, (c) the
switch-shock comparison across A2 treatments, and (d) doc-ready table bodies
so numbers reach README/paper without hand transcription.

Usage:
    uv run python analyze_exp7.py
    uv run python analyze_exp7.py --tag 5seed --emit latex
"""
import argparse
import glob
import json
import statistics as st
from pathlib import Path

from analyze_all import paired_permutation_p, paired_t_p, cohens_dz
from experiment_mechanism import summarize_exp7

RESULTS = Path("gradient_results")
SEED_ORDER = [42, 137, 256, 789, 1337]


def load_sweep(tag, switch_step=10000):
    p = RESULTS / f"exp7_curriculum_sw{switch_step}_{tag}.json"
    if not p.exists():
        return None, None
    d = json.load(open(p))
    return d.get("config", {}), d["runs"]


def _rows(runs):
    """Per-seed (A, B, F) with F=None when the run diverged or is missing."""
    out = []
    seeds = sorted({r["seed"] for r in runs.values()},
                   key=lambda s: SEED_ORDER.index(s) if s in SEED_ORDER else 99)
    for s in seeds:
        A = runs.get(f"A_full_s{s}", {}).get("final_bpb")
        B = runs.get(f"B_quartic_s{s}", {}).get("final_bpb")
        f = runs.get(f"F_switch_s{s}", {})
        F = f.get("final_bpb")
        ok = F is not None and F == F  # NaN-safe
        out.append((s, A, B, F if ok else None, f.get("diverged_at")))
    return out


def emit_markdown(rows):
    print("\n--- README block ---")
    print("```")
    print(f"{'seed':<7}{'A: full':<10}{'B: quartic':<12}{'F: quartic→full':<18}"
          f"{'B vs A':<9}{'F vs A':<9}{'F vs B'}")
    for s, A, B, F, div in rows:
        pct = lambda ref, v: f"{(ref - v) / ref * 100:+.2f}%" if (ref and v) else "—"
        fs = f"{F:.4f}" if F else "diverged (NaN)"
        print(f"{s:<7}{A:<10.4f}{B:<12.4f}{fs:<18}"
              f"{pct(A,B):<9}{pct(A,F):<9}{pct(B,F)}")
    done = [(A, B, F) for _, A, B, F, _ in rows if F]
    if done:
        mA = st.mean(a for a, _, _ in done)
        mB = st.mean(b for _, b, _ in done)
        mF = st.mean(f for _, _, f in done)
        print(f"\n{'mean':<7}{mA:<10.4f}{mB:<12.4f}{mF:<18.4f}"
              f"{(mA-mB)/mA*100:+.2f}%  {(mA-mF)/mA*100:+.2f}%  {(mB-mF)/mB*100:+.2f}%"
              f"   (n={len(done)})")
    print("```")


def emit_latex(rows):
    print("\n--- LaTeX table body ---")
    for s, A, B, F, div in rows:
        pct = lambda ref, v: f"${(ref - v) / ref * 100:+.2f}$\\%" if (ref and v) else "---"
        fs = f"{F:.4f}" if F else "\\textit{diverged (NaN)}"
        print(f"{s} & {A:.4f} & {B:.4f} & {fs} & "
              f"{pct(A,B)} & {pct(A,F)} & {pct(B,F)} \\\\")
    done = [(A, B, F) for _, A, B, F, _ in rows if F]
    if done:
        mA = st.mean(a for a, _, _ in done)
        mB = st.mean(b for _, b, _ in done)
        mF = st.mean(f for _, _, f in done)
        print("\\midrule")
        print(f"Mean & {mA:.4f} & {mB:.4f} & \\textbf{{{mF:.4f}}} & "
              f"${(mA-mB)/mA*100:+.2f}$\\% & $\\mathbf{{{(mA-mF)/mA*100:+.2f}}}$\\% & "
              f"${(mB-mF)/mB*100:+.2f}$\\% \\\\")


def shock_table(switch_step=10000):
    """Compare the switch shock across A2 treatments (and the plain sweep)."""
    rows = []
    for f in sorted(glob.glob(str(RESULTS / f"exp7_switch_trace_s*_sw{switch_step}_*.json"))):
        d = json.load(open(f))
        tr, cfg = d["trace"], d.get("config", {})
        pre = [r for r in tr if r["step"] < switch_step]
        post = [r for r in tr if r["step"] >= switch_step]
        if not post:
            continue
        near = [r for r in post if r["step"] < switch_step + 100]
        tail = [r for r in post if r["step"] >= switch_step + 400]
        label = Path(f).stem.split(f"_sw{switch_step}_")[-1]
        rows.append({
            "treatment": label,
            "seed": d["seed"],
            "pre_upd": max((r["max_adam_update"] for r in pre), default=float("nan")),
            "peak_loss": max(r["loss"] for r in near),
            "peak_gnorm": max(r["grad_norm_preclip"] for r in near),
            "peak_upd": max(r["max_adam_update"] for r in near),
            "tail_upd": max((r["max_adam_update"] for r in tail), default=float("nan")),
            "reset": cfg.get("reset_optimizer"), "ramp": cfg.get("ramp_steps"),
            "warmup": cfg.get("post_switch_warmup"), "mode": cfg.get("switch_mode"),
        })
    if not rows:
        return
    print("\n" + "=" * 92)
    print("  SWITCH SHOCK  (peaks over the 100 steps after the switch)")
    print("=" * 92)
    print(f"  {'treatment':<22}{'seed':>6}{'peak loss':>11}{'peak gnorm':>12}"
          f"{'pre upd':>11}{'peak upd':>11}{'x pre':>7}{'by +400':>10}")
    for r in sorted(rows, key=lambda x: (x["treatment"], x["seed"])):
        ratio = r["peak_upd"] / r["pre_upd"] if r["pre_upd"] == r["pre_upd"] else float("nan")
        pre = f"{r['pre_upd']:.2e}" if r["pre_upd"] == r["pre_upd"] else "—"
        rat = f"{ratio:.2f}" if ratio == ratio else "—"
        print(f"  {r['treatment']:<22}{r['seed']:>6}{r['peak_loss']:>11.4f}"
              f"{r['peak_gnorm']:>12.3f}{pre:>11}{r['peak_upd']:>11.2e}"
              f"{rat:>7}{r['tail_upd']:>10.2e}")
    print("\n  Criterion (pre-registered): the treatment that removes the stale Adam")
    print("  second moment should pull 'peak upd' back toward 'pre upd' (ratio -> 1).")


def compare_arms(tag_a, tag_b, label_a, label_b, switch_step=10000):
    """Paired comparison of two arms measured under the same tag scheme.

    Used for ramp-vs-hard-switch: both resume from the same pre-switch state
    with the same restored RNG, so they see identical batches and identical
    eval draws. The endpoint noise that limits other comparisons is
    common-mode here and largely cancels.
    """
    _, ra = load_sweep(tag_a, switch_step)
    _, rb = load_sweep(tag_b, switch_step)
    if not ra or not rb:
        return
    seeds = sorted({r["seed"] for r in ra.values() if r["config"] == "F_switch"}
                   & {r["seed"] for r in rb.values() if r["config"] == "F_switch"},
                   key=lambda s: SEED_ORDER.index(s) if s in SEED_ORDER else 99)
    if not seeds:
        return
    print("\n" + "=" * 92)
    print(f"  {label_a} vs {label_b}  (paired, same pre-switch state and RNG)")
    print("=" * 92)
    print(f"  {'seed':>6} {'baseline':>10} {label_a:>12} {label_b:>12} "
          f"{'B-arm vs A':>11} {'diff':>10} {label_b+' vs '+label_a:>16}")
    diffs, vsA = [], []
    for s in seeds:
        A = ra[f"A_full_s{s}"]["final_bpb"]
        a = ra[f"F_switch_s{s}"]["final_bpb"]
        b = rb[f"F_switch_s{s}"]["final_bpb"]
        diffs.append(a - b)                 # positive => label_b is better
        vsA.append(A - b)
        print(f"  {s:>6} {A:>10.4f} {a:>12.4f} {b:>12.4f} "
              f"{(A-a)/A*100:>+10.2f}% {a-b:>+10.5f} {(a-b)/a*100:>+15.2f}%")
    n = len(diffs)
    if n >= 2:
        p, c, t = paired_permutation_p(diffs)
        npos = sum(1 for d in diffs if d > 0)
        print(f"\n  {label_b} vs {label_a}: {npos}/{n} favour {label_b}, "
              f"perm {c}/{t}={p:.3f}, t_p={paired_t_p(diffs):.4f}, "
              f"dz={cohens_dz(diffs):.2f}")
        print(f"    mean difference {st.mean(diffs):+.5f} bpb "
              f"(sd {st.stdev(diffs):.5f})" if n > 1 else "")
        pA, cA, tA = paired_permutation_p(vsA)
        nposA = sum(1 for d in vsA if d > 0)
        print(f"  {label_b} vs baseline: {nposA}/{n} positive, perm {cA}/{tA}={pA:.3f}, "
              f"t_p={paired_t_p(vsA):.4f}, dz={cohens_dz(vsA):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"),
                    help="Paired comparison of two arms, e.g. --compare 5seed ramp5seed")
    ap.add_argument("--tag", default="5seed")
    ap.add_argument("--switch-step", type=int, default=10000)
    ap.add_argument("--emit", choices=["markdown", "latex", "both", "none"],
                    default="both")
    args = ap.parse_args()

    if args.compare:
        a, b = args.compare
        labels = {"5seed": "F (switch)", "ramp5seed": "R (ramp)"}
        compare_arms(a, b, labels.get(a, a), labels.get(b, b), args.switch_step)
        return

    cfg, runs = load_sweep(args.tag, args.switch_step)
    if runs is None:
        print(f"No sweep found for tag '{args.tag}' at switch {args.switch_step}.")
        return
    rows = _rows(runs)
    n_done = sum(1 for r in rows if r[3])
    print("=" * 92)
    print(f"  EXPERIMENT 7 — window removal at step {args.switch_step} "
          f"({n_done}/{len(SEED_ORDER)} seeds complete)")
    print("=" * 92)

    summarize_exp7(runs)

    if args.emit in ("markdown", "both"):
        emit_markdown(rows)
    if args.emit in ("latex", "both"):
        emit_latex(rows)

    shock_table(args.switch_step)


if __name__ == "__main__":
    main()
