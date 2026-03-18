"""Quick evaluation utilities for model assessment."""

from typing import Any, Callable

import torch

from neurogen.config import TrainConfig, get_device
from neurogen.data.shakespeare import ShakespeareDataset
from neurogen.model.gpt import GPT
from neurogen.training.trainer import evaluate, train


def quick_evaluate(
    model: GPT,
    dataset: ShakespeareDataset,
    config: TrainConfig,
    device: str,
    n_batches: int = 10,
) -> float:
    """Run a quick evaluation on the validation set.

    A lightweight wrapper around evaluate() with a smaller default batch count,
    useful for rapid checks during development.

    Args:
        model: The GPT model to evaluate.
        dataset: The dataset providing validation batches.
        config: Training config (for batch_size).
        device: Device string.
        n_batches: Number of batches to average over.

    Returns:
        Average validation loss.
    """
    return evaluate(model, dataset, config, device, n_batches=n_batches)


def train_and_evaluate(
    model: GPT,
    dataset: ShakespeareDataset,
    train_config: TrainConfig,
    device: str,
    init_fn: Callable[[GPT], dict[str, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    """Train a model and return comprehensive evaluation metrics.

    Runs the full training loop with optional custom initialization, then
    computes final validation loss and gathers all metrics.

    Args:
        model: The GPT model to train and evaluate.
        dataset: The dataset providing batches.
        train_config: Training configuration.
        device: Device string.
        init_fn: Optional initialization function.

    Returns:
        Dictionary containing:
            - train_losses: List of per-step training losses.
            - val_losses: List of validation losses at eval intervals.
            - best_val_loss: Best validation loss seen during training.
            - final_train_loss: Loss on the last training step.
            - final_val_loss: Validation loss after training completes.
            - total_time: Wall-clock training time in seconds.
            - steps_per_sec: Training throughput.
            - post_train_val_loss: Fresh validation evaluation after training.
    """
    metrics = train(
        model=model,
        dataset=dataset,
        config=train_config,
        init_fn=init_fn,
        device=device,
    )

    # Run a fresh evaluation after training
    post_val_loss = evaluate(
        model=model,
        dataset=dataset,
        config=train_config,
        device=device,
        n_batches=20,
    )
    metrics["post_train_val_loss"] = post_val_loss

    return metrics
