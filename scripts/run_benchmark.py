"""CLI for running benchmarks (BM1-BM8)."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurogen.analysis.weight_analysis import weight_statistics
from neurogen.baselines.initializers import INITIALIZERS, available_initializers
from neurogen.ca.engine import CAWeightEngine, CA_VARIANTS
from neurogen.config import GPTConfig, TrainConfig, get_device
from neurogen.data.shakespeare import ShakespeareDataset
from neurogen.model.gpt import GPT
from neurogen.training.trainer import train


def run_bm1_init_quality(
    dataset: ShakespeareDataset,
    model_config: GPTConfig,
    device: str,
    n_seeds: int = 3,
    quick: bool = False,
) -> dict:
    """BM1: Initialization Quality Benchmark."""
    print("\n=== BM1: Initialization Quality ===")
    inits = available_initializers()
    if not quick:
        inits += list(CA_VARIANTS.keys())

    results = {}
    for init_name in inits:
        print(f"  {init_name}...", end=" ", flush=True)
        seed_stats = []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            model = GPT(model_config).to(device)

            if init_name in INITIALIZERS:
                weights = INITIALIZERS[init_name](model)
            else:
                engine = CAWeightEngine(init_name, device="cpu")
                weights = engine.develop_weights(model)
            model.set_weight_tensors(weights)

            # Collect stats per weight tensor
            all_stats = {}
            for name, w in weights.items():
                all_stats[name] = weight_statistics(w)

            # Initial loss
            x, y = dataset.get_batch("val", 32, model_config.block_size, device)
            _, loss = model(x, y)
            all_stats["initial_val_loss"] = loss.item()

            seed_stats.append(all_stats)

        results[init_name] = seed_stats
        print("done")

    return results


def run_bm2_convergence(
    dataset: ShakespeareDataset,
    model_config: GPTConfig,
    train_config: TrainConfig,
    device: str,
    inits: list[str] | None = None,
    n_seeds: int = 3,
) -> dict:
    """BM2: Convergence Speed Benchmark."""
    print("\n=== BM2: Convergence Speed ===")
    if inits is None:
        inits = ["xavier_normal", "kaiming_normal", "orthogonal"]

    results = {}
    for init_name in inits:
        results[init_name] = []
        for seed in range(n_seeds):
            print(f"  {init_name} seed={seed}...", flush=True)
            torch.manual_seed(seed)
            model = GPT(model_config)

            if init_name in INITIALIZERS:
                weights = INITIALIZERS[init_name](model)
                model.set_weight_tensors(weights)
            elif init_name in CA_VARIANTS:
                engine = CAWeightEngine(init_name, device="cpu")
                weights = engine.develop_weights(model)
                model.set_weight_tensors(weights)

            metrics = train(model, dataset, train_config)
            results[init_name].append(metrics)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NeuroGen benchmarks")
    parser.add_argument(
        "--benchmark", type=str, default="bm1",
        help="Benchmark to run (bm1, bm2, all)",
    )
    parser.add_argument("--suite", type=str, default="", help="Suite: quick, standard")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--output-dir", type=str, default="outputs/benchmarks")
    args = parser.parse_args()

    device = args.device or get_device()
    quick = args.quick or args.suite == "quick"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")

    dataset = ShakespeareDataset()
    n_seeds = 1 if quick else 3
    steps = 500 if quick else 5000

    model_config = GPTConfig(
        block_size=32 if quick else 256,
        vocab_size=dataset.vocab_size,
        n_layer=2, n_head=2, n_embd=64,
        dropout=0.0,
    )

    train_config = TrainConfig(
        max_steps=steps,
        eval_interval=50 if quick else 250,
        eval_steps=5,
        batch_size=32 if quick else 64,
        lr=3e-4,
        device=device,
        log_interval=50,
    )

    benchmarks_to_run = []
    if args.benchmark == "all" or args.suite:
        benchmarks_to_run = ["bm1", "bm2"]
    else:
        benchmarks_to_run = [args.benchmark]

    for bm in benchmarks_to_run:
        if bm == "bm1":
            results = run_bm1_init_quality(
                dataset, model_config, device, n_seeds=n_seeds, quick=quick
            )
            with open(output_dir / "bm1_init_quality.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"BM1 results saved to {output_dir / 'bm1_init_quality.json'}")

        elif bm == "bm2":
            inits = ["xavier_normal", "kaiming_normal"]
            if not quick:
                inits += ["orthogonal", "fixup"]
            results = run_bm2_convergence(
                dataset, model_config, train_config, device,
                inits=inits, n_seeds=n_seeds,
            )
            with open(output_dir / "bm2_convergence.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

            # Generate report
            from research.report import generate_phase_report
            generate_phase_report(
                results,
                "BM2: Convergence Speed Benchmark",
                output_dir / "bm2_convergence.md",
            )
            print(f"BM2 results saved to {output_dir}")

    print("\nBenchmarks complete.")


if __name__ == "__main__":
    main()
