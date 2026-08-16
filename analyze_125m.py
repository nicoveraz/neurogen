"""
Analyze 125M validation results.

Usage: python analyze_125m.py
"""

import json, glob, math
from collections import defaultdict
from pathlib import Path

# Reuse the paired statistics (exact sign-flip permutation + paired-t + dz)
# from analyze_all so there is one source of truth for the test we run.
from analyze_all import paired_permutation_p, paired_t_p, cohens_dz

def mean(x): return sum(x)/len(x) if x else 0
def std(x):
    m = mean(x)
    return (sum((v-m)**2 for v in x)/max(len(x)-1,1))**0.5 if len(x)>1 else 0

def _load_3_4m_means(steps=20000):
    """Mean final val_bpb per arch at 3.4M, read from validation_results/.

    Previously these four numbers sat in the source as literals. Reading them
    keeps the scale-comparison table honest if the underlying runs change.
    """
    per_arch = defaultdict(list)
    for f in sorted(glob.glob("validation_results/*_s*.json")):
        if f.endswith("_100k.json"):
            continue
        d = json.load(open(f))
        s = d["summary"]
        if s.get("max_steps") != steps:
            continue
        v = s.get("final_vbpb", s.get("final_val_bpb"))
        if v is not None:
            per_arch[s["arch"]].append(v)
    return {a: mean(v) for a, v in per_arch.items()}


def _paired_table(by_seed, label):
    """Print a paired baseline-vs-variant table for one training horizon."""
    base = by_seed.get("baseline", {})
    if not base:
        print(f"  (no baseline runs at {label})")
        return
    bl_mean = mean(list(base.values()))
    print(f"\n{label}  (baseline mean {bl_mean:.4f}, n={len(base)})")
    print(f"{'config':<24} {'n':>3} {'mean bpb':>10} {'std':>8} {'vs bl':>8} {'perm_p':>11} {'t_p':>8} {'dz':>6}")
    print("-" * 84)
    known = ["baseline", "window_quadratic", "window_power_3.0", "window_power_4.0",
             "window_power_6.0", "window_power_8.0", "window_power_10.0",
             "window_power_12.0", "window_quad_induction", "window_p4_induction"]
    # Anything else present (e.g. curriculum arms "…@sw25000") is listed after.
    for arch in known + sorted(set(by_seed) - set(known)):
        d = by_seed.get(arch, {})
        if not d: continue
        vals = list(d.values())
        m, s = mean(vals), std(vals)
        if arch == "baseline":
            print(f"{arch:<24} {len(vals):>3} {m:>10.4f} {s:>8.4f} {'—':>8} {'—':>11} {'—':>8} {'—':>6}")
            continue
        diff = (bl_mean - m)/bl_mean * 100
        seeds = sorted(set(base) & set(d))
        diffs = [base[sd] - d[sd] for sd in seeds]
        if len(diffs) >= 2:
            perm_p, cnt, tot = paired_permutation_p(diffs)
            t_p = paired_t_p(diffs)
            dz = cohens_dz(diffs)
            npos = sum(1 for x in diffs if x > 0)
            print(f"{arch:<24} {len(vals):>3} {m:>10.4f} {s:>8.4f} {diff:>+7.1f}% "
                  f"{cnt}/{tot}={perm_p:>5.3f} {t_p:>8.4f} {dz:>6.2f}  ({npos}/{len(diffs)}+)")
        else:
            print(f"{arch:<24} {len(vals):>3} {m:>10.4f} {s:>8.4f} {diff:>+7.1f}% "
                  f"{'n<2':>11} {'—':>8} {'—':>6}")

def main():
    by_seed = defaultdict(lambda: defaultdict(dict))  # max_steps -> arch -> {seed: bpb}
    results = defaultdict(list)                         # arch -> all bpb (for sweep/scale)
    curves = defaultdict(list)
    curve_by_seed = defaultdict(dict)                   # arch -> {seed: curve}

    for f in sorted(glob.glob("results_125m/*.json")):
        d = json.load(open(f))
        s = d["summary"]
        # A curriculum run (--switch-step) shares the "arch" field with the
        # windows-throughout run. Give it a distinct label so the two are never
        # averaged together.
        arch = s["arch"]
        if s.get("switch_step") is not None:
            arch = f"{arch}@sw{s['switch_step']}"
        by_seed[s["max_steps"]][arch][s["seed"]] = s["final_val_bpb"]
        curves[arch].append(d["curve"])
        curve_by_seed[arch][s["seed"]] = d["curve"]

    if not by_seed:
        print("No results found in results_125m/. Run experiments first.")
        return

    # For the exponent sweep / scale comparison below we use the largest common
    # horizon (20k, where all seeds exist). Mixing horizons inflates the mean.
    results_20k = by_seed.get(20000, {})
    for arch, d in results_20k.items():
        results[arch] = list(d.values())
    bl = results.get("baseline", [])
    bl_mean = mean(bl) if bl else 1.0

    print("=" * 90)
    print("  NEUROGEN 125M VALIDATION RESULTS")
    print("=" * 90)
    print("\nPaired across matched seeds, reported PER training horizon (mixing 20k and")
    print("50k runs into one mean inflates it). perm_p = exact sign-flip permutation")
    print("(floor 1/2^n); t_p = paired-t two-sided; dz = paired effect size.")

    # Final performance tables, separated by horizon
    for ms in sorted(by_seed):
        _paired_table(by_seed[ms], f"{ms//1000}k steps")

    # Unified per-seed gap at a matched step (20k), across ALL seeds. Each seed's
    # baseline and quartic arms share the same LR schedule, so the within-seed gap
    # is valid even though seeds 42/137 run a 50k schedule (read at step 20k) while
    # 256/789/1337 are dedicated 20k runs. This is the honest "grows with scale"
    # headline number: do NOT report a single cherry-picked seed.
    def _bpb_at_step(curve, step):
        for p in curve:
            if p.get("step") == step:
                return p.get("val_bpb")
        return None
    MATCH_STEP = 20000
    base_c, quar_c = curve_by_seed.get("baseline", {}), curve_by_seed.get("window_power_4.0", {})
    seeds = sorted(set(base_c) & set(quar_c))
    print(f"\nPer-seed gap at step {MATCH_STEP} (baseline vs quartic, all seeds):")
    print(f"  {'seed':>6} {'baseline':>10} {'quartic':>10} {'gap%':>8}")
    gaps = []
    for sd in seeds:
        b = _bpb_at_step(base_c[sd], MATCH_STEP); q = _bpb_at_step(quar_c[sd], MATCH_STEP)
        if b is None or q is None: continue
        g = (b - q) / b * 100; gaps.append(g)
        print(f"  {sd:>6} {b:>10.4f} {q:>10.4f} {g:>+7.2f}% {'✓' if g > 0 else '✗'}")
    if gaps:
        npos = sum(1 for g in gaps if g > 0)
        # Paired test on the raw per-seed differences, not on the percentages.
        pdiffs = [_bpb_at_step(base_c[sd], MATCH_STEP) - _bpb_at_step(quar_c[sd], MATCH_STEP)
                  for sd in seeds
                  if _bpb_at_step(base_c[sd], MATCH_STEP) is not None
                  and _bpb_at_step(quar_c[sd], MATCH_STEP) is not None]
        print(f"  mean gap {mean(gaps):+.2f}% (sd {std(gaps):.2f}), {npos}/{len(gaps)} seeds positive")
        if len(pdiffs) >= 2:
            perm_p, cnt, tot = paired_permutation_p(pdiffs)
            print(f"  paired: perm {cnt}/{tot}={perm_p:.3f}  t_p={paired_t_p(pdiffs):.4f}  "
                  f"dz={cohens_dz(pdiffs):.2f}")
        print(f"  → headline: +{mean(gaps):.1f}% at 125M ({MATCH_STEP//1000}k, n={len(gaps)}), "
              f"{npos}/{len(gaps)} seeds positive.")

    # 50k extension: report it, and report why it does not support a scaling claim.
    # The convergence gate is the mean decline over the final 10k steps, in bpb per
    # 1k steps. A run counts as converged below GATE. Both arms fail it, and the
    # quartic arms fail it harder — a widening gap between two curves that are both
    # still falling is uninformative about the asymptotic gap.
    GATE = 0.002
    ext = sorted(sd for sd in seeds
                 if _bpb_at_step(base_c[sd], 50000) is not None
                 and _bpb_at_step(quar_c[sd], 50000) is not None)
    if ext:
        print(f"\n50k extension (n={len(ext)}) — UNCONVERGED, do not cite as a scaling result:")
        print(f"  {'seed':>6} {'baseline':>10} {'quartic':>10} {'gap%':>8} "
              f"{'bl decl/1k':>11} {'q decl/1k':>10}")
        bl50, q50 = [], []
        for sd in ext:
            b, q = _bpb_at_step(base_c[sd], 50000), _bpb_at_step(quar_c[sd], 50000)
            bl50.append(b); q50.append(q)
            bd = (_bpb_at_step(base_c[sd], 40000) - b) / 10
            qd = (_bpb_at_step(quar_c[sd], 40000) - q) / 10
            print(f"  {sd:>6} {b:>10.4f} {q:>10.4f} {(b-q)/b*100:>+7.2f}% "
                  f"{bd:>11.4f} {qd:>10.4f}")
        gap50 = (mean(bl50) - mean(q50)) / mean(bl50) * 100
        print(f"  mean {mean(bl50):.4f} vs {mean(q50):.4f} → +{gap50:.1f}% "
              f"(n={len(ext)}, both arms above the {GATE} bpb/1k convergence gate)")

    # Exponent sweep summary
    print(f"\nExponent Sweep (window_power_X.0):")
    print(f"  {'exponent':<10} {'layers [d12] windows':>40} {'mean bpb':>10} {'vs baseline':>12}")
    print("  " + "-" * 75)
    for exp in [2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]:
        arch = f"window_power_{exp}"
        vals = results.get(arch, [])
        if not vals: continue
        # Compute window sizes for 12 layers, 1024 seq, base 16
        windows = [int(16 + ((i+1)/12)**exp * (1024-16)) for i in range(12)]
        win_str = f"[{windows[0]},{windows[3]},{windows[7]},{windows[11]}]"
        m = mean(vals)
        diff = (bl_mean - m)/bl_mean*100
        print(f"  {exp:<10} {win_str:>40} {m:>10.4f} {diff:>+11.1f}%")

    # 95% CI
    print(f"\n95% Confidence Intervals:")
    for arch in ["baseline", "window_quadratic", "window_power_4.0", "window_power_6.0",
                 "window_power_8.0", "window_power_10.0", "window_power_12.0",
                 "window_quad_induction", "window_p4_induction"]:
        vals = results.get(arch, [])
        if not vals: continue
        m, s = mean(vals), std(vals)
        n = len(vals)
        t_crit = 2.776 if n == 5 else 2.262 if n == 10 else 1.96
        ci = t_crit * s / math.sqrt(n)
        print(f"  {arch:<28} {m:.4f} ± {ci:.4f}  [{m-ci:.4f}, {m+ci:.4f}]")

    # Dominance check
    print(f"\nDominance (every seed beats baseline mean {bl_mean:.4f}):")
    for arch in ["window_quadratic", "window_power_4.0", "window_power_6.0",
                 "window_power_8.0", "window_power_10.0", "window_power_12.0",
                 "window_quad_induction", "window_p4_induction"]:
        vals = results.get(arch, [])
        if not vals: continue
        n_beat = sum(1 for v in vals if v < bl_mean)
        print(f"  {arch:<28} {n_beat}/{len(vals)} seeds beat baseline")

    # Learning curve snapshots
    if curves:
        print(f"\nLearning curve snapshots (mean bpb):")
        checkpoints = [1000, 5000, 10000, 25000, 50000]
        print(f"{'config':<28}", end="")
        for cp in checkpoints:
            print(f" {'step '+str(cp//1000)+'k':>10}", end="")
        print()
        print("-" * (28 + 10*len(checkpoints)))
        for arch in ["baseline", "window_quadratic", "window_power_4.0", "window_power_8.0",
                     "window_quad_induction"]:
            if arch not in curves: continue
            print(f"{arch:<28}", end="")
            for cp in checkpoints:
                vals = []
                for curve in curves[arch]:
                    for p in curve:
                        if p["step"] == cp:
                            vals.append(p["val_bpb"])
                            break
                print(f" {mean(vals):>10.4f}" if vals else f" {'—':>10}", end="")
            print()

    # Comparison with 3.4M results
    print(f"\n{'=' * 90}")
    print(f"  SCALE COMPARISON: 3.4M vs 125M")
    print(f"{'=' * 90}")
    small = _load_3_4m_means()
    print(f"{'config':<28} {'3.4M bpb':>10} {'125M bpb':>10} {'3.4M vs bl':>12} {'125M vs bl':>12} {'scales?':>8}")
    print("-" * 85)
    for arch in ["baseline", "window_quadratic", "window_power_4.0",
                 "window_power_6.0", "window_power_8.0", "window_power_10.0",
                 "window_power_12.0", "window_quad_induction", "window_p4_induction"]:
        s_val = small.get(arch)
        l_vals = results.get(arch, [])
        l_val = mean(l_vals) if l_vals else None
        # An arch with no run at a scale prints "—", not 0.0000/+100% — a missing
        # run is not a measurement.
        if s_val is None and l_val is None:
            continue
        s_diff = (small["baseline"] - s_val)/small["baseline"]*100 if s_val else None
        l_diff = (bl_mean - l_val)/bl_mean*100 if l_val else None
        scales = ("YES" if (s_diff > 0.5 and l_diff > 0.5) else "NO") \
            if (s_diff is not None and l_diff is not None) else "?"
        f = lambda v, w, suf="": (f"{v:>{w}.4f}" if suf == "" else f"{v:>+{w-1}.1f}%") \
            if v is not None else f"{'—':>{w}}"
        print(f"{arch:<28} {f(s_val,10)} {f(l_val,10)} {f(s_diff,12,'%')} "
              f"{f(l_diff,12,'%')} {scales:>8}")

    print()

if __name__ == "__main__":
    main()
