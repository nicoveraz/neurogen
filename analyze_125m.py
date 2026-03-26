"""
Analyze 125M validation results.

Usage: python analyze_125m.py
"""

import json, glob, math
from collections import defaultdict
from pathlib import Path

def mean(x): return sum(x)/len(x) if x else 0
def std(x):
    m = mean(x)
    return (sum((v-m)**2 for v in x)/max(len(x)-1,1))**0.5 if len(x)>1 else 0

def welch_t_pvalue(a, b):
    """Approximate p-value from Welch's t-test (normal approx for df>4)."""
    na, nb = len(a), len(b)
    ma, mb = mean(a), mean(b)
    sa, sb = std(a), std(b)
    se = (sa**2/na + sb**2/nb)**0.5
    if se == 0: return 0, 1.0
    t = (ma - mb) / se
    p = 2 * 0.5 * (1 + math.erf(-abs(t)/math.sqrt(2)))
    return t, p

def cohens_d(a, b):
    pooled = ((std(a)**2 + std(b)**2)/2)**0.5
    return (mean(a) - mean(b))/pooled if pooled > 0 else 0

def main():
    results = defaultdict(list)
    curves = defaultdict(list)

    for f in sorted(glob.glob("results_125m/*.json")):
        d = json.load(open(f))
        s = d["summary"]
        results[s["arch"]].append(s["final_val_bpb"])
        curves[s["arch"]].append(d["curve"])

    if not results:
        print("No results found in results_125m/. Run experiments first.")
        return

    bl = results.get("baseline", [])
    bl_mean = mean(bl) if bl else 1.0

    print("=" * 90)
    print("  NEUROGEN 125M VALIDATION RESULTS")
    print("=" * 90)

    # Final performance table
    print(f"\n{'config':<28} {'n':>3} {'mean bpb':>10} {'std':>8} {'vs baseline':>12} {'p-value':>8} {'Cohen d':>8}")
    print("-" * 85)

    for arch in ["baseline", "window_quadratic", "window_power_3.0", "window_power_4.0",
                 "window_power_6.0", "window_power_8.0", "window_power_10.0",
                 "window_power_12.0", "window_quad_induction", "window_p4_induction"]:
        vals = results.get(arch, [])
        if not vals: continue
        m, s = mean(vals), std(vals)
        if arch == "baseline":
            print(f"{arch:<28} {len(vals):>3} {m:>10.4f} {s:>8.4f} {'—':>12} {'—':>8} {'—':>8}")
        else:
            diff = (bl_mean - m)/bl_mean * 100
            t, p = welch_t_pvalue(bl, vals)
            cd = cohens_d(bl, vals)
            sig = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
            print(f"{arch:<28} {len(vals):>3} {m:>10.4f} {s:>8.4f} {diff:>+11.1f}% {p:>7.4f}{sig:>1} {cd:>8.2f}")

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
    small = {"baseline": 0.9002, "window_power_4.0": 0.8866, "window_quadratic": 0.8911, "window_quad_induction": 0.8899}
    print(f"{'config':<28} {'3.4M bpb':>10} {'125M bpb':>10} {'3.4M vs bl':>12} {'125M vs bl':>12} {'scales?':>8}")
    print("-" * 85)
    for arch in ["baseline", "window_quadratic", "window_power_4.0",
                 "window_power_6.0", "window_power_8.0", "window_power_10.0",
                 "window_power_12.0", "window_quad_induction", "window_p4_induction"]:
        s_val = small.get(arch, 0)
        l_vals = results.get(arch, [])
        l_val = mean(l_vals) if l_vals else 0
        s_diff = (small["baseline"] - s_val)/small["baseline"]*100
        l_diff = (bl_mean - l_val)/bl_mean*100 if l_val else 0
        scales = "YES" if (s_diff > 0.5 and l_diff > 0.5) else "NO" if l_val else "?"
        print(f"{arch:<28} {s_val:>10.4f} {l_val:>10.4f} {s_diff:>+11.1f}% {l_diff:>+11.1f}% {scales:>8}")

    print()

if __name__ == "__main__":
    main()
