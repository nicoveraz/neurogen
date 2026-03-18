"""Evaluation and metrics collection utilities."""

import torch

from neurogen.model.gpt import GPT
from neurogen.data.shakespeare import ShakespeareDataset


@torch.no_grad()
def quick_evaluate(
    model: GPT,
    dataset: ShakespeareDataset,
    n_batches: int = 10,
    batch_size: int = 32,
    device: str = "cpu",
) -> float:
    """Quick evaluation: average val loss over n_batches.

    Args:
        model: The model to evaluate.
        dataset: The dataset.
        n_batches: Number of batches to average.
        batch_size: Batch size.
        device: Device.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    for _ in range(n_batches):
        x, y = dataset.get_batch("val", batch_size, model.config.block_size, device)
        _, loss = model(x, y)
        total_loss += loss.item()
    model.train()
    return total_loss / n_batches


def train_and_evaluate(
    model: GPT,
    dataset: ShakespeareDataset,
    steps: int = 500,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str = "cpu",
) -> float:
    """Train for N steps and return final val loss.

    Lightweight version for meta-learning inner loops.

    Args:
        model: The model.
        dataset: The dataset.
        steps: Training steps.
        batch_size: Batch size.
        lr: Learning rate.
        device: Device.

    Returns:
        Final validation loss.
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(steps):
        x, y = dataset.get_batch("train", batch_size, model.config.block_size, device)
        _, loss = model(x, y)

        if torch.isnan(loss) or torch.isinf(loss):
            return 100.0  # penalty for diverged training

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return quick_evaluate(model, dataset, n_batches=5, batch_size=batch_size, device=device)
