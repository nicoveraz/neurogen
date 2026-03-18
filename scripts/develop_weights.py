"""CLI script to develop weights using a CA variant and save them.

Usage:
    python scripts/develop_weights.py --variant grid_ca --steps 64 --save-path weights.pt
    python scripts/develop_weights.py --variant neural_ca --steps 128 --device mps
    python scripts/develop_weights.py --variant grid_ca --seed 42 --visualize
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from neurogen.analysis.weight_analysis import analyze_weight_dict, aggregate_analysis
from neurogen.ca.engine import CAWeightEngine
from neurogen.config import GPTConfig, get_device
from neurogen.data.shakespeare import ShakespeareDataset
from neurogen.model.gpt import GPT


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Develop weight matrices using a CA variant."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="grid_ca",
        help="CA variant name (default: grid_ca).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=64,
        help="Number of CA development steps (default: 64).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="weights.pt",
        help="Path to save the developed weights (default: weights.pt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cuda/mps/cpu). Auto-detects if not set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--seed-pattern",
        type=str,
        default="center",
        choices=["center", "random"],
        help="Seed initialization pattern (default: center).",
    )
    parser.add_argument(
        "--n-layer",
        type=int,
        default=6,
        help="Number of transformer layers (default: 6).",
    )
    parser.add_argument(
        "--n-head",
        type=int,
        default=6,
        help="Number of attention heads (default: 6).",
    )
    parser.add_argument(
        "--n-embd",
        type=int,
        default=384,
        help="Embedding dimension (default: 384).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Block size / context window (default: 256).",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Print weight analysis after development.",
    )
    return parser.parse_args()


def main() -> None:
    """Develop and save CA-generated weights."""
    args = parse_args()

    device = args.device or get_device()
    torch.manual_seed(args.seed)

    print(f"Device: {device}")
    print(f"CA variant: {args.variant}")
    print(f"Development steps: {args.steps}")
    print(f"Seed pattern: {args.seed_pattern}")

    # Check available variants
    available = CAWeightEngine.available_variants()
    if args.variant not in available:
        print(f"Error: Unknown variant '{args.variant}'.")
        print(f"Available: {', '.join(available)}")
        sys.exit(1)

    # Create model to get target shapes
    dataset = ShakespeareDataset()
    model_config = GPTConfig(
        block_size=args.block_size,
        vocab_size=dataset.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
    )
    model = GPT(model_config).to(device)

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Weight matrices: {len(model.get_weight_tensors())}")

    # Develop weights
    print(f"\nDeveloping weights with {args.variant}...")
    engine = CAWeightEngine(variant=args.variant, device=device)
    weights = engine.develop_weights(
        model,
        n_steps=args.steps,
        seed_pattern=args.seed_pattern,
    )

    print(f"Development time: {engine.last_develop_time_ms:.1f}ms")
    print(f"Genome size: {engine.genome_size():,} parameters")
    print(f"Compression ratio: {engine.compression_ratio(model):.1f}x")

    # Print per-weight summary
    print("\nDeveloped weights:")
    for name, w in weights.items():
        print(
            f"  {name}: shape={tuple(w.shape)}, "
            f"mean={w.mean().item():.4f}, std={w.std().item():.4f}, "
            f"min={w.min().item():.4f}, max={w.max().item():.4f}"
        )

    # Optional analysis
    if args.analyze:
        print("\nWeight Analysis:")
        analysis = analyze_weight_dict(weights)
        agg = aggregate_analysis(analysis)
        for metric, value in sorted(agg.items()):
            print(f"  {metric}: {value:.6f}")

    # Save weights
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Detach and move to CPU for portability
    cpu_weights = {k: v.detach().cpu() for k, v in weights.items()}
    torch.save(
        {
            "weights": cpu_weights,
            "variant": args.variant,
            "n_steps": args.steps,
            "seed": args.seed,
            "seed_pattern": args.seed_pattern,
            "model_config": {
                "block_size": args.block_size,
                "n_layer": args.n_layer,
                "n_head": args.n_head,
                "n_embd": args.n_embd,
            },
            "genome_size": engine.genome_size(),
        },
        save_path,
    )
    print(f"\nWeights saved to {save_path}")


if __name__ == "__main__":
    main()
