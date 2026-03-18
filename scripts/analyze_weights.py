"""CLI script to analyze saved weights or model parameters.

Usage:
    python scripts/analyze_weights.py --weights weights.pt
    python scripts/analyze_weights.py --model model_checkpoint.pt
    python scripts/analyze_weights.py --weights weights.pt --output analysis.json
    python scripts/analyze_weights.py --weights weights.pt --device cpu
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from neurogen.analysis.weight_analysis import (
    aggregate_analysis,
    analyze_weight_dict,
    condition_number,
    effective_rank,
    frobenius_norm,
    sparsity,
    spectral_norm,
    weight_statistics,
)
from neurogen.config import GPTConfig, get_device
from neurogen.model.gpt import GPT


def _load_weights_from_file(
    path: str, device: str
) -> dict[str, torch.Tensor]:
    """Load weight tensors from a .pt file.

    Handles two formats:
    1. Direct dict of tensors (from develop_weights.py).
    2. Model checkpoint with model_state_dict key.

    Args:
        path: Path to the .pt file.
        device: Device to load tensors onto.

    Returns:
        Dictionary mapping parameter names to weight tensors.
    """
    data = torch.load(path, map_location=device, weights_only=False)

    if isinstance(data, dict):
        if "weights" in data:
            # Format from develop_weights.py
            return {k: v.to(device) for k, v in data["weights"].items()}
        if "model_state_dict" in data:
            # Checkpoint format
            return {
                k: v.to(device)
                for k, v in data["model_state_dict"].items()
                if "weight" in k
                and "ln_" not in k
                and "pos_emb" not in k
                and "lm_head" not in k
            }
        # Assume it's a direct dict of tensors
        return {
            k: v.to(device)
            for k, v in data.items()
            if isinstance(v, torch.Tensor)
        }

    raise ValueError(
        f"Cannot interpret file format of {path}. "
        f"Expected dict with 'weights' or 'model_state_dict' key."
    )


def _load_weights_from_model(
    path: str, device: str
) -> dict[str, torch.Tensor]:
    """Load weight tensors from a model checkpoint.

    Creates a GPT model from the checkpoint config and loads state dict,
    then extracts weight tensors via get_weight_tensors().

    Args:
        path: Path to the checkpoint .pt file.
        device: Device to load tensors onto.

    Returns:
        Dictionary mapping parameter names to weight tensors.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "config" in checkpoint and "model_state_dict" in checkpoint:
        config = checkpoint["config"]
        if isinstance(config, GPTConfig):
            model = GPT(config).to(device)
        else:
            model = GPT(
                GPTConfig(
                    block_size=config.get("block_size", 256),
                    vocab_size=config.get("vocab_size", 65),
                    n_layer=config.get("n_layer", 6),
                    n_head=config.get("n_head", 6),
                    n_embd=config.get("n_embd", 384),
                )
            ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.get_weight_tensors()

    # Fallback: treat as raw weights
    return _load_weights_from_file(path, device)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Analyze weight tensors from saved files or model checkpoints."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--weights",
        type=str,
        help="Path to a weights .pt file (from develop_weights.py).",
    )
    group.add_argument(
        "--model",
        type=str,
        help="Path to a model checkpoint .pt file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save analysis results as JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cuda/mps/cpu). Auto-detects if not set.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-layer detailed analysis.",
    )
    return parser.parse_args()


def main() -> None:
    """Analyze weights and print/save results."""
    args = parse_args()
    device = args.device or get_device()

    # Load weights
    if args.weights:
        source_path = args.weights
        print(f"Loading weights from: {source_path}")
        weights = _load_weights_from_file(source_path, device)
    else:
        source_path = args.model
        print(f"Loading model from: {source_path}")
        weights = _load_weights_from_model(source_path, device)

    print(f"Device: {device}")
    print(f"Number of weight tensors: {len(weights)}")

    total_params = sum(w.numel() for w in weights.values())
    print(f"Total parameters in weights: {total_params:,}")
    print()

    # Run analysis
    per_layer = analyze_weight_dict(weights)
    agg = aggregate_analysis(per_layer)

    # Print aggregate summary
    print("=" * 60)
    print("AGGREGATE ANALYSIS")
    print("=" * 60)
    for metric, value in sorted(agg.items()):
        print(f"  {metric}: {value:.6f}")
    print()

    # Print per-layer if verbose
    if args.verbose:
        print("=" * 60)
        print("PER-LAYER ANALYSIS")
        print("=" * 60)
        for name, stats in per_layer.items():
            print(f"\n  {name}:")
            tensor = weights[name]
            print(f"    shape: {tuple(tensor.shape)}")
            for metric, value in sorted(stats.items()):
                if isinstance(value, float):
                    if value == float("inf"):
                        print(f"    {metric}: inf")
                    else:
                        print(f"    {metric}: {value:.6f}")
                else:
                    print(f"    {metric}: {value}")

    # Print summary table
    print()
    print("=" * 60)
    print("WEIGHT SUMMARY TABLE")
    print("=" * 60)
    header = f"{'Name':<40} {'Shape':<16} {'Mean':<10} {'Std':<10} {'Spec.Norm':<12}"
    print(header)
    print("-" * len(header))

    for name, stats in per_layer.items():
        shape_str = str(tuple(weights[name].shape))
        mean_str = f"{stats.get('mean', 0):.4f}"
        std_str = f"{stats.get('std', 0):.4f}"
        sn_str = f"{stats.get('spectral_norm', 0):.4f}"
        print(f"{name:<40} {shape_str:<16} {mean_str:<10} {std_str:<10} {sn_str:<12}")

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "source": source_path,
            "n_tensors": len(weights),
            "total_params": total_params,
            "aggregate": agg,
            "per_layer": {
                name: {k: v for k, v in stats.items()}
                for name, stats in per_layer.items()
            },
        }

        # Handle non-serializable values
        def _clean(obj: object) -> object:
            if isinstance(obj, float):
                if obj != obj:
                    return None
                if obj == float("inf") or obj == float("-inf"):
                    return str(obj)
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            return obj

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(_clean(output_data), f, indent=2)
        print(f"\nAnalysis saved to {output_path}")


if __name__ == "__main__":
    main()
