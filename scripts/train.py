"""CLI for training MicroGPT with configurable initialization."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurogen.config import GPTConfig, TrainConfig, get_device
from neurogen.data.shakespeare import ShakespeareDataset
from neurogen.model.gpt import GPT
from neurogen.training.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MicroGPT on Shakespeare")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-layer", type=int, default=6, help="Number of layers")
    parser.add_argument("--n-head", type=int, default=6, help="Number of heads")
    parser.add_argument("--n-embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--block-size", type=int, default=256, help="Context length")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--device", type=str, default="", help="Device override")
    parser.add_argument(
        "--init", type=str, default="default", help="Initialization method"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=250, help="Eval every N steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--generate", type=int, default=200, help="Chars to generate after training"
    )
    parser.add_argument(
        "--save-metrics", type=str, default="", help="Path to save metrics JSON"
    )
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Device
    device = args.device if args.device else get_device()
    print(f"Using device: {device}")

    # Load dataset
    dataset = ShakespeareDataset()
    print(
        f"Dataset loaded: {len(dataset.train_data)} train chars, "
        f"{len(dataset.val_data)} val chars, vocab_size={dataset.vocab_size}"
    )

    # Create model
    model_config = GPTConfig(
        block_size=args.block_size,
        vocab_size=dataset.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(model_config)
    print(f"Model created: {model.count_parameters():,} parameters")

    # Apply initialization if requested
    if args.init != "default":
        try:
            from neurogen.baselines.initializers import get_initializer

            initializer = get_initializer(args.init)
            weights = initializer(model)
            model.set_weight_tensors(weights)
            print(f"Applied initialization: {args.init}")
        except ImportError:
            print(f"Warning: initializer '{args.init}' not available yet")

    # Train
    train_config = TrainConfig(
        max_steps=args.steps,
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
    metrics = train(model, dataset, train_config)

    # Generate sample text
    if args.generate > 0:
        print(f"\n--- Generated text ({args.generate} chars) ---")
        prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = model.generate(prompt, max_new_tokens=args.generate, temperature=0.8)
        print(dataset.decode(generated[0]))
        print("--- End ---")

    # Save metrics
    if args.save_metrics:
        metrics_path = Path(args.save_metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

    print(f"\nTraining complete in {metrics['total_train_time_s']:.1f}s")
    if metrics["best_val_loss"] is not None:
        print(f"Best val_loss: {metrics['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
