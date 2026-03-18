"""CLI for running CA weight development standalone."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurogen.analysis.plotting import plot_weight_heatmap
from neurogen.analysis.weight_analysis import weight_statistics
from neurogen.ca.engine import CAWeightEngine
from neurogen.config import GPTConfig, get_device
from neurogen.model.gpt import GPT


def main() -> None:
    parser = argparse.ArgumentParser(description="Develop weights using CA")
    parser.add_argument(
        "--variant", type=str, default="grid_ca",
        choices=CAWeightEngine.available_variants(),
    )
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--visualize", action="store_true", help="Save weight heatmaps"
    )
    parser.add_argument("--output-dir", type=str, default="outputs/figures")
    args = parser.parse_args()

    device = args.device or get_device()
    print(f"Device: {device}, Variant: {args.variant}")

    # Create a model to get target shapes
    config = GPTConfig(
        block_size=32, vocab_size=256,
        n_layer=2, n_head=2, n_embd=64, dropout=0.0,
    )
    model = GPT(config)

    # Develop weights
    torch.manual_seed(args.seed)
    engine = CAWeightEngine(args.variant, device="cpu")
    weights = engine.develop_weights(model)

    print(f"\nGenome size: {engine.genome_size():,} parameters")
    print(f"Compression ratio: {engine.compression_ratio(model):.1f}x")

    print("\nWeight statistics:")
    for name, w in weights.items():
        stats = weight_statistics(w)
        print(
            f"  {name}: shape={tuple(w.shape)}, "
            f"mean={stats['mean']:.4f}, std={stats['std']:.4f}"
        )

    if args.visualize:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, w in weights.items():
            safe_name = name.replace(".", "_")
            path = output_dir / f"{args.variant}_{safe_name}.png"
            plot_weight_heatmap(w, title=f"{args.variant}: {name}", save_path=path)
            print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
