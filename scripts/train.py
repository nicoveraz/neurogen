"""CLI training script for NeuroGen.

Usage:
    python scripts/train.py --steps 1000 --device mps
    python scripts/train.py --init xavier_normal --seed 42 --save-checkpoint ckpt.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurogen.config import GPTConfig, TrainConfig, get_device
from neurogen.data.shakespeare import ShakespeareDataset
from neurogen.model.gpt import GPT
from neurogen.training.trainer import train, save_checkpoint


INIT_METHODS = {
    "default",
    "xavier_normal",
    "xavier_uniform",
    "kaiming_normal",
    "kaiming_uniform",
    "orthogonal",
    "zeros",
}


def apply_init(model: GPT, method: str) -> dict[str, torch.Tensor]:
    """Apply a baseline initialization method to the model.

    Args:
        model: The GPT model to initialize.
        method: Initialization method name.

    Returns:
        Dictionary of initialized weight tensors.
    """
    weights = model.get_weight_tensors()

    if method == "default":
        return weights

    for name, w in weights.items():
        if w.dim() < 2:
            continue
        if method == "xavier_normal":
            nn.init.xavier_normal_(w)
        elif method == "xavier_uniform":
            nn.init.xavier_uniform_(w)
        elif method == "kaiming_normal":
            nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")
        elif method == "kaiming_uniform":
            nn.init.kaiming_uniform_(w, mode="fan_out", nonlinearity="relu")
        elif method == "orthogonal":
            nn.init.orthogonal_(w)
        elif method == "zeros":
            nn.init.zeros_(w)

    return weights


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a MicroGPT model on Shakespeare."
    )
    parser.add_argument(
        "--init",
        type=str,
        default="xavier_normal",
        choices=sorted(INIT_METHODS),
        help="Weight initialization method (default: xavier_normal).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Number of training steps (default: 5000).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cuda/mps/cpu). Auto-detects if not set.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Peak learning rate (default: 3e-4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64).",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=250,
        help="Steps between evaluations (default: 250).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--save-checkpoint",
        type=str,
        default=None,
        help="Path to save a checkpoint after training.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full training pipeline."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Device setup
    device = args.device if args.device else get_device()
    print(f"Using device: {device}")

    # Load dataset
    print("Loading Shakespeare dataset...")
    dataset = ShakespeareDataset()
    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Train tokens: {len(dataset.train_data):,}")
    print(f"Val tokens: {len(dataset.val_data):,}")

    # Create model
    model_config = GPTConfig(vocab_size=dataset.vocab_size)
    model = GPT(model_config)
    print(f"Model parameters: {model.count_parameters():,}")

    # Training config
    train_config = TrainConfig(
        max_steps=args.steps,
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    # Initialization function
    init_method = args.init
    init_fn = None
    if init_method != "default":
        def init_fn(m: GPT) -> dict[str, torch.Tensor]:
            return apply_init(m, init_method)

    print(f"Initialization: {init_method}")
    print(f"Training for {args.steps} steps...")
    print("-" * 60)

    # Train
    metrics = train(
        model=model,
        dataset=dataset,
        config=train_config,
        init_fn=init_fn,
        device=device,
    )

    # Print results
    print("-" * 60)
    print(f"Training complete in {metrics['total_time']:.1f}s")
    print(f"  Steps/sec:       {metrics['steps_per_sec']:.1f}")
    print(f"  Final train loss: {metrics['final_train_loss']:.4f}")
    print(f"  Final val loss:   {metrics['final_val_loss']:.4f}")
    print(f"  Best val loss:    {metrics['best_val_loss']:.4f}")

    # Save checkpoint if requested
    if args.save_checkpoint:
        # Recreate optimizer for checkpoint (lightweight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        save_checkpoint(model, optimizer, args.steps, metrics, args.save_checkpoint)

    # Generate sample text
    print("\n--- Sample Generation ---")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)
    print(dataset.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
