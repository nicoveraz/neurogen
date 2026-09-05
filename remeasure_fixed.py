"""Re-measure every saved checkpoint under the fixed eval harness (exp 4).

Writes a NEW results file. New-harness and old-harness numbers are never
comparable (this repo's matched-null rule), so nothing here is compared against
a recorded val_bpb, and no existing result file is touched.

A full matched re-measurement additionally needs the A and B arms. Their saved
checkpoints are NOT the 20K runs used in the paired comparisons -- seed 42's are
the 100K run, seeds 137/256 are other runs entirely -- so every A/B arm must be
retrained before any comparison can move to the new harness. This script records
that gap explicitly rather than quietly reporting the arms it can reach.

Usage:
    uv run python remeasure_fixed.py            # all checkpoints
    uv run python remeasure_fixed.py --limit 3  # smoke test
"""
import argparse, glob, json, os, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from prepare import (load_data, evaluate_val_bpb, MAX_SEQ_LEN, VOCAB_SIZE,
                     FIXED_EVAL_TOKENS, FIXED_EVAL_SEED)
from train_r4 import (GPT, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS, DEVICE,
                      BATCH_SIZE, get_arch_cfg)

OUT = Path("gradient_results") / "fixed_eval_remeasure.json"
SEEDS = [42, 137, 256, 789, 1337]


def load_model(path):
    blob = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = blob.get("arch_cfg")
    if cfg is None:  # A/B checkpoints record the arch name, not the config
        cfg = {} if blob["arch"] == "baseline" else get_arch_cfg(blob["arch"])
    m = GPT(VOCAB_SIZE, MAX_SEQ_LEN, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS,
            arch_cfg=dict(cfg))
    m.load_state_dict(blob["model_state_dict"], strict=False)
    return m.to(DEVICE), blob


def committed_endpoint(name):
    """The value this checkpoint's own result file recorded, for identification.

    Recorded so a row can be traced back to its run. NOT for comparison: it was
    measured on the old harness.
    """
    if not name.startswith("model_F_switch_"):
        return None, None
    seed, rest = name[len("model_F_switch_"):-3].split("_", 1)
    p = Path("gradient_results") / f"exp7_curriculum_{rest}.json"
    if not p.exists():
        return None, None
    r = json.load(open(p))["runs"].get(f"F_switch_s{seed}")
    return (r["final_bpb"], p.name) if r else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    val = load_data("val")
    n_batches = FIXED_EVAL_TOKENS // (BATCH_SIZE * MAX_SEQ_LEN)
    print(f"fixed set: {FIXED_EVAL_TOKENS:,} tokens, {n_batches} batches, "
          f"seed {FIXED_EVAL_SEED}, {FIXED_EVAL_TOKENS/len(val)*100:.1f}% of val")

    paths = sorted(glob.glob("checkpoints/model_F_switch_*.pt")) + \
            sorted(glob.glob("checkpoints/model_baseline_*.pt")) + \
            sorted(glob.glob("checkpoints/model_window_power_4.0_*.pt"))
    if args.limit:
        paths = paths[:args.limit]

    rows, t0 = {}, time.time()
    for i, p in enumerate(paths, 1):
        name = os.path.basename(p)
        try:
            m, blob = load_model(p)
        except Exception as e:                      # a checkpoint we cannot score
            print(f"  [{i}/{len(paths)}] {name}: LOAD FAILED — {e}")
            rows[name] = {"error": str(e)}
            continue
        t = time.time()
        bpb = evaluate_val_bpb(m, val, BATCH_SIZE, MAX_SEQ_LEN, DEVICE, fixed=True)
        old, src = committed_endpoint(name)
        rows[name] = {
            "fixed_bpb": bpb,
            "arch": blob.get("arch"),
            "seed": blob.get("seed"),
            "switch_step": blob.get("switch_step"),
            "total_steps": blob.get("total_steps", blob.get("max_steps")),
            "old_harness_bpb": old,      # identification only — DO NOT compare
            "old_harness_source": src,
        }
        print(f"  [{i}/{len(paths)}] {name:<46} fixed={bpb:.6f}  ({time.time()-t:.0f}s)")
        del m

    # Which arms a matched comparison still needs.
    missing = []
    for arch in ("baseline", "window_power_4.0"):
        for s in SEEDS:
            missing.append(f"{arch}_s{s}")
    json.dump({
        "harness": {
            "fixed_eval_tokens": FIXED_EVAL_TOKENS, "batches": n_batches,
            "seed": FIXED_EVAL_SEED, "batch_size": BATCH_SIZE,
            "block_size": MAX_SEQ_LEN, "device": DEVICE,
        },
        "warning": ("Fixed-harness values. NOT comparable to any val_bpb recorded "
                    "elsewhere in this repo, which were measured by resampling. "
                    "old_harness_bpb is present for identification only."),
        "matched_comparison_blocked_by": {
            "arms": missing,
            "reason": ("No saved checkpoint is the 20K A or B run. The five that "
                       "exist are different runs: seed 42's are the 100K run "
                       "(0.807/0.799), seeds 137/256 are ~0.965-0.975, while the "
                       "20K arms are 0.904/0.893. All 10 A/B arms must be "
                       "retrained (~12 h MPS) before any paired comparison can "
                       "move to this harness."),
        },
        "runs": rows,
    }, open(OUT, "w"), indent=2)
    print(f"\n{len(rows)} checkpoints in {(time.time()-t0)/60:.1f} min -> {OUT}")
    print("A matched comparison is still blocked: no 20K A/B checkpoint exists.")


if __name__ == "__main__":
    main()
