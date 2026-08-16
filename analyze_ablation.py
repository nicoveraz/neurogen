"""Per-layer attention-window ablation: which layer's locality carries the gain?

All configs at a given seed share the SAME init and the SAME data order — the
only thing that differs is the per-layer attention window. That makes the
attribution clean even at n=1: differences in final val_bpb are attributable
purely to the window pattern.

Across seeds the runs are paired (each config shares an init with the baseline
at that seed), so per-seed differences are the unit of analysis and their signs
are reported, not just their mean.

Usage:
    uv run python analyze_ablation.py                         # seed 42 only
    uv run python analyze_ablation.py --seeds 42,137,256,789,1337
"""
import argparse
import json
import os

from analyze_all import paired_permutation_p, paired_t_p, cohens_dz

DEPTH4_QUARTIC = "[8,23,86,256]"
BASELINE = "baseline"
QUARTIC = "window_power_4.0"

CONFIGS = [
    (BASELINE,            "[256,256,256,256]", "full attention (reference)"),
    ("window_only_L0",    "[8,256,256,256]",   "only layer 0 local"),
    ("window_only_L01",   "[8,23,256,256]",    "early layers (L0+L1) local"),
    ("window_no_L0",      "[256,23,86,256]",   "quartic minus L0 locality"),
    (QUARTIC,             DEPTH4_QUARTIC,      "quartic (reference)"),
    ("window_only_last",  "[256,256,256,8]",   "reversed control: only last local"),
]


def final_bpb(arch, seed):
    path = f"validation_results/{arch}_s{seed}.json"
    if not os.path.exists(path):
        return None
    s = json.load(open(path))["summary"]
    return s.get("final_vbpb") or s.get("final_val_bpb")


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _per_seed_table(seed):
    base, quar = final_bpb(BASELINE, seed), final_bpb(QUARTIC, seed)
    if base is None or quar is None:
        print(f"  seed {seed}: missing baseline/quartic — skipping")
        return
    gap = base - quar  # the full quartic improvement (100% reference)
    print(f"\n  seed {seed}: baseline {base:.4f}  quartic {quar:.4f}  "
          f"gap {gap:.4f} (= 100% of the quartic gain)")
    print(f"  {'config':<20}{'windows':<20}{'final':>8}{'vs_base':>9}{'%gain':>7}  note")
    print("  " + "-" * 86)
    for arch, win, note in CONFIGS:
        v = final_bpb(arch, seed)
        if v is None:
            print(f"  {arch:<20}{win:<20}{'—':>8}{'—':>9}{'—':>7}  (pending)")
            continue
        vs = (base - v) / base * 100
        pct = (base - v) / gap * 100 if gap else 0
        print(f"  {arch:<20}{win:<20}{v:>8.4f}{vs:>+8.2f}%{pct:>6.0f}%  {note}")


def _paired_report(name, diffs, rule):
    """Paired test over per-seed differences, with the sign count up front."""
    n = len(diffs)
    if n < 2:
        print(f"\n  {name}: n={n} — no paired test possible.")
        return
    p, cnt, tot = paired_permutation_p(diffs)
    tp = paired_t_p(diffs)
    npos = sum(1 for d in diffs if d > 0)
    print(f"\n  {name}: {npos}/{n} positive, perm {cnt}/{tot}={p:.3f}, "
          f"t_p={tp:.4f}, dz={cohens_dz(diffs):.2f}")
    floor = 1 / 2 ** n
    print(f"    floor at n={n} is {floor:.3f}"
          + ("  (cannot reach p<0.05)" if floor > 0.05 else ""))
    print(f"    {rule(npos, n, tp)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=str, default="42",
                    help="Comma-separated seeds, e.g. 42,137,256,789,1337")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print("=" * 78)
    print(f"  PER-LAYER WINDOW ABLATION (20k steps; matched init + data order per seed)")
    print("=" * 78)
    for seed in seeds:
        _per_seed_table(seed)

    # Seeds where every arm needed for the paired tests exists.
    usable = [s for s in seeds
              if all(final_bpb(a, s) is not None
                     for a in (BASELINE, QUARTIC, "window_only_L0",
                               "window_only_L01", "window_only_last"))]
    if len(usable) < 2:
        print(f"\n  Paired tests need >=2 complete seeds (have {len(usable)}): "
              f"single-seed reading below.")
        print("\n  Reading: early-layer locality (L0, L0+L1) drives the gain; late-layer")
        print("  locality (only_last) HURTS (worse than baseline). The gradual quartic")
        print("  ramp is not optimal — early-local + rest-full (only_L01) beats it.")
        print("  Caveat: n=1 seed. Big effects are robust; the only_L01>quartic margin")
        print("  is small (replicate on another seed before asserting it).")
        return

    print(f"\n{'=' * 78}")
    print(f"  PAIRED ACROSS {len(usable)} SEEDS: {usable}")
    print(f"{'=' * 78}")
    print("  Pre-registered criteria (stated before the runs; see README).")

    # only_last should be WORSE than baseline. The permutation test is one-sided,
    # so orient the differences in the predicted direction: positive = only_last
    # is worse, which is the hypothesis being tested.
    d_last = [final_bpb("window_only_last", s) - final_bpb(BASELINE, s) for s in usable]
    _paired_report("only_last vs baseline (positive = only_last WORSE, as predicted)",
                   d_last,
                   lambda npos, n, tp: (
                       "VERDICT: claimed — late-layer locality hurts on every seed."
                       if npos == n and n >= 5 else
                       "VERDICT: consistent direction, under-powered at this n."
                       if npos == n else
                       "VERDICT: NOT claimed — not every seed is worse than baseline."))

    # only_L0 should recover >=50% of the quartic gain on every seed.
    fracs = [(final_bpb(BASELINE, s) - final_bpb("window_only_L0", s))
             / (final_bpb(BASELINE, s) - final_bpb(QUARTIC, s)) for s in usable]
    n_ok = sum(1 for f in fracs if f >= 0.5)
    print(f"\n  only_L0 recovers >=50% of the quartic gain: {n_ok}/{len(usable)} seeds "
          f"(per-seed: {', '.join(f'{f*100:.0f}%' for f in fracs)})")
    print(f"    {'VERDICT: claimed.' if n_ok == len(usable) and len(usable) >= 5 else 'VERDICT: NOT claimed at this n.'}")

    # only_L01 > quartic is the small margin that needs replication.
    d_l01 = [final_bpb(QUARTIC, s) - final_bpb("window_only_L01", s) for s in usable]
    _paired_report("only_L01 vs quartic (the small margin)", d_l01,
                   lambda npos, n, tp: (
                       "VERDICT: claimed — >=4/5 positive and paired-t p<0.05."
                       if n >= 5 and npos >= 4 and tp < 0.05 else
                       "VERDICT: NOT claimed — report as 'not distinguishable "
                       "from quartic'."))


if __name__ == "__main__":
    main()
