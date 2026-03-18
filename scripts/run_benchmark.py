"""CLI script to run benchmarks for NeuroGen.

Usage:
    python scripts/run_benchmark.py --suite quick
    python scripts/run_benchmark.py --benchmark bm1
    python scripts/run_benchmark.py --benchmark bm2 --device cpu
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from neurogen.analysis.weight_analysis import aggregate_analysis, analyze_weight_dict
from neurogen.baselines.initializers import INITIALIZERS, initialize
from neurogen.config import GPTConfig, TrainConfig, get_device
from neurogen.data.shakespeare import ShakespeareDataset
from neurogen.model.gpt import GPT
from neurogen.training.trainer import train

SUITES: dict[str, list[str]] = {
    "quick": ["bm1", "bm2"],
    "standard": ["bm1", "bm2", "bm3"],
}


def _run_bm1(device: str, output_dir: Path, quick: bool = False) -> dict[str, Any]:
    """BM1: Initialization Quality -- weight properties across init strategies."""
    print("\n--- BM1: Initialization Quality ---")
    dataset = ShakespeareDataset()
    n_l, n_h, n_e, bs = (2, 2, 64, 32) if quick else (6, 6, 384, 256)
    cfg = GPTConfig(block_size=bs, vocab_size=dataset.vocab_size,
                    n_layer=n_l, n_head=n_h, n_embd=n_e, dropout=0.0)
    seeds = [42] if quick else [42, 137, 256]
    methods = ["xavier_normal", "kaiming_normal", "orthogonal"] if quick else list(INITIALIZERS)

    results: dict[str, Any] = {}
    for name in methods:
        analyses = []
        for s in seeds:
            torch.manual_seed(s)
            model = GPT(cfg).to(device)
            w = initialize(model, name)
            model.set_weight_tensors(w)
            analyses.append(aggregate_analysis(analyze_weight_dict(w)))

        keys = set().union(*(a.keys() for a in analyses))
        avg = {k: sum(a.get(k, 0) for a in analyses) / len(analyses) for k in keys}
        results[name] = avg
        print(f"  {name}: spectral_norm={avg.get('mean_spectral_norm', 0):.4f}")

    bm_dir = output_dir / "bm1"
    bm_dir.mkdir(parents=True, exist_ok=True)
    with open(bm_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def _run_bm2(device: str, output_dir: Path, quick: bool = False) -> dict[str, Any]:
    """BM2: Convergence Speed -- training convergence comparison."""
    print("\n--- BM2: Convergence Speed ---")
    dataset = ShakespeareDataset()
    if quick:
        cfg = GPTConfig(block_size=32, vocab_size=dataset.vocab_size,
                        n_layer=2, n_head=2, n_embd=64, dropout=0.0)
        steps, evl, bs_t, seeds = 200, 50, 32, [42]
        methods = ["xavier_normal", "kaiming_normal", "orthogonal"]
    else:
        cfg = GPTConfig(block_size=256, vocab_size=dataset.vocab_size,
                        n_layer=6, n_head=6, n_embd=384, dropout=0.2)
        steps, evl, bs_t, seeds = 5000, 250, 64, [42, 137, 256]
        methods = list(INITIALIZERS)

    results: dict[str, Any] = {}
    for name in methods:
        sr = []
        for s in seeds:
            torch.manual_seed(s)
            model = GPT(cfg)
            tc = TrainConfig(max_steps=steps, eval_interval=evl, batch_size=bs_t, lr=3e-4)
            fn = INITIALIZERS[name]
            m = train(model=model, dataset=dataset, config=tc,
                      init_fn=lambda mdl, f=fn: f(mdl), device=device)
            sr.append({"seed": s, "final_val_loss": m["final_val_loss"],
                       "best_val_loss": m["best_val_loss"], "time": m["total_time"]})
        n = len(sr)
        results[name] = {
            "final_val_loss_mean": sum(r["final_val_loss"] for r in sr) / n,
            "best_val_loss_mean": sum(r["best_val_loss"] for r in sr) / n,
            "n_seeds": n, "per_seed": sr,
        }
        print(f"  {name}: best_val={results[name]['best_val_loss_mean']:.4f}")

    bm_dir = output_dir / "bm2"
    bm_dir.mkdir(parents=True, exist_ok=True)
    with open(bm_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def _run_bm3(device: str, output_dir: Path, quick: bool = False) -> dict[str, Any]:
    """BM3: Compute Efficiency -- time and memory costs."""
    print("\n--- BM3: Compute Efficiency ---")
    dataset = ShakespeareDataset()
    if quick:
        cfg = GPTConfig(block_size=32, vocab_size=dataset.vocab_size,
                        n_layer=2, n_head=2, n_embd=64, dropout=0.0)
        steps, bs_t = 100, 32
        methods = ["xavier_normal", "kaiming_normal"]
    else:
        cfg = GPTConfig(block_size=256, vocab_size=dataset.vocab_size,
                        n_layer=6, n_head=6, n_embd=384, dropout=0.2)
        steps, bs_t = 1000, 64
        methods = list(INITIALIZERS)

    results: dict[str, Any] = {}
    for name in methods:
        torch.manual_seed(42)
        model = GPT(cfg)
        t0 = time.time()
        w = initialize(model, name)
        init_ms = (time.time() - t0) * 1000
        model.set_weight_tensors(w)
        tc = TrainConfig(max_steps=steps, eval_interval=steps, batch_size=bs_t, lr=3e-4)
        m = train(model=model, dataset=dataset, config=tc, device=device)
        results[name] = {
            "init_time_ms": init_ms, "train_time_s": m["total_time"],
            "steps_per_sec": m["steps_per_sec"],
            "final_val_loss": m["final_val_loss"],
        }
        print(f"  {name}: init={init_ms:.1f}ms, train={m['total_time']:.1f}s")

    bm_dir = output_dir / "bm3"
    bm_dir.mkdir(parents=True, exist_ok=True)
    with open(bm_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


BENCHMARKS = {"bm1": _run_bm1, "bm2": _run_bm2, "bm3": _run_bm3}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Run NeuroGen benchmarks.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--suite", choices=sorted(SUITES))
    g.add_argument("--benchmark", choices=sorted(BENCHMARKS))
    p.add_argument("--output-dir", default="outputs/benchmarks")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    """Run the specified benchmark(s)."""
    args = parse_args()
    device = args.device or get_device()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    if args.suite:
        quick = args.suite == "quick"
        t0 = time.time()
        for bm in SUITES[args.suite]:
            BENCHMARKS[bm](device, out, quick=quick)
        print(f"\nSuite '{args.suite}' done in {time.time() - t0:.1f}s")
    else:
        BENCHMARKS[args.benchmark](device, out, quick=False)


if __name__ == "__main__":
    main()
