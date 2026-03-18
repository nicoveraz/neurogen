"""CLI for analyzing model weights."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurogen.analysis.plotting import plot_singular_values, plot_weight_heatmap
from neurogen.analysis.weight_analysis import (
    singular_value_spectrum,
    weight_statistics,
)
from neurogen.baselines.initializers import INITIALIZERS
from neurogen.ca.engine import CAWeightEngine, CA_VARIANTS
from neurogen.config import GPTConfig, get_device
from neurogen.model.gpt import GPT


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze weight matrices")
    parser.add_argument("--init", type=str, default="xavier_normal")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs/figures")
    args = parser.parse_args()

    device = args.device or get_device()

    config = GPTConfig(
        block_size=32, vocab_size=256,
        n_layer=2, n_head=2, n_embd=64, dropout=0.0,
    )
    model = GPT(config).to(device)

    # Get weights
    if args.init in INITIALIZERS:
        weights = INITIALIZERS[args.init](model)
    elif args.init in CA_VARIANTS:
        engine = CAWeightEngine(args.init, device="cpu")
        weights = engine.develop_weights(model)
    else:
        weights = model.get_weight_tensors()

    # Analyze
    print(f"\nAnalyzing {args.init} initialization:")
    all_stats = {}
    spectra = {}
    for name, w in weights.items():
        stats = weight_statistics(w)
        all_stats[name] = stats
        print(
            f"  {name}: "
            f"std={stats['std']:.4f}, "
            f"spectral_norm={stats.get('spectral_norm', 'N/A')}, "
            f"eff_rank={stats.get('effective_rank', 'N/A')}"
        )
        if w.dim() >= 2:
            spectra[name] = singular_value_spectrum(w)

    if args.visualize:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Weight heatmaps
        for name, w in weights.items():
            safe_name = name.replace(".", "_")
            plot_weight_heatmap(
                w, title=f"{args.init}: {name}",
                save_path=output_dir / f"heatmap_{args.init}_{safe_name}.png",
            )

        # Singular value spectra
        if spectra:
            plot_singular_values(
                spectra, title=f"SV Spectrum: {args.init}",
                save_path=output_dir / f"svd_{args.init}.png",
            )
        print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
