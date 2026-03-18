"""Training loop, learning rate schedule, and checkpoint utilities."""

import math
import time
from pathlib import Path
from typing import Any, Callable

import torch

from neurogen.config import GPTConfig, TrainConfig, get_device
from neurogen.data.shakespeare import ShakespeareDataset
from neurogen.model.gpt import GPT


def get_lr(step: int, config: TrainConfig) -> float:
    """Compute learning rate with linear warmup and cosine decay.

    Args:
        step: Current training step (0-indexed).
        config: Training configuration with warmup_steps, lr, min_lr, max_steps.

    Returns:
        Learning rate for the given step.
    """
    # Linear warmup
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps

    # After max_steps, return min_lr
    if step >= config.max_steps:
        return config.min_lr

    # Cosine decay between warmup_steps and max_steps
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.lr - config.min_lr)


def evaluate(
    model: GPT,
    dataset: ShakespeareDataset,
    config: TrainConfig,
    device: str,
    n_batches: int = 20,
) -> float:
    """Evaluate model on the validation set.

    Args:
        model: The GPT model to evaluate.
        dataset: The dataset providing validation batches.
        config: Training config (for batch_size).
        device: Device string.
        n_batches: Number of batches to average over.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = dataset.get_batch(
                "val", config.batch_size, model.config.block_size, device
            )
            _, loss = model(x, y)
            total_loss += loss.item()
    model.train()
    return total_loss / n_batches


def train(
    model: GPT,
    dataset: ShakespeareDataset,
    config: TrainConfig,
    init_fn: Callable[[GPT], dict[str, torch.Tensor]] | None = None,
    device: str | None = None,
    callback: Callable[[int, float, float | None], None] | None = None,
) -> dict[str, Any]:
    """Run the training loop.

    Args:
        model: The GPT model to train.
        dataset: The dataset providing batches.
        config: Training configuration.
        init_fn: Optional initialization function that takes the model and
            returns a weight dict. Applied before training starts.
        device: Device string override. If None, uses config.device or auto.
        callback: Optional function called each step with (step, train_loss, val_loss).
            val_loss is None except at eval intervals.

    Returns:
        Dictionary of training metrics including train_losses, val_losses,
        best_val_loss, total_time, and steps_per_sec.
    """
    if device is None:
        device = config.device if config.device != "auto" else get_device()

    model = model.to(device)

    # Apply custom initialization if provided
    if init_fn is not None:
        weights = init_fn(model)
        model.set_weight_tensors(weights)

    # Set up optimizer with weight decay separation
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=config.lr, betas=(0.9, 0.95))

    # Training state
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    start_time = time.time()

    model.train()
    for step in range(config.max_steps):
        # Update learning rate
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward and backward
        x, y = dataset.get_batch(
            "train", config.batch_size, model.config.block_size, device
        )
        _, loss = model(x, y)
        loss.backward()

        # Gradient clipping
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        train_loss = loss.item()
        train_losses.append(train_loss)

        # Evaluation
        val_loss = None
        if (step + 1) % config.eval_interval == 0 or step == config.max_steps - 1:
            val_loss = evaluate(model, dataset, config, device)
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            print(
                f"step {step + 1:5d} | train_loss {train_loss:.4f} | "
                f"val_loss {val_loss:.4f} | lr {lr:.2e}"
            )

        if callback is not None:
            callback(step, train_loss, val_loss)

    total_time = time.time() - start_time
    steps_per_sec = config.max_steps / total_time if total_time > 0 else 0.0

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1] if train_losses else float("inf"),
        "final_val_loss": val_losses[-1] if val_losses else float("inf"),
        "total_time": total_time,
        "steps_per_sec": steps_per_sec,
    }


def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict[str, Any],
    path: str | Path,
) -> None:
    """Save a training checkpoint.

    Args:
        model: The GPT model.
        optimizer: The optimizer.
        step: Current training step.
        metrics: Dictionary of training metrics.
        path: File path to save the checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "metrics": metrics,
        "config": model.config,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str | Path,
    model: GPT,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: File path to load the checkpoint from.
        model: The GPT model to load weights into.
        optimizer: Optional optimizer to load state into.
        device: Device to map tensors to.

    Returns:
        Dictionary with step and metrics from the checkpoint.
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "step": checkpoint.get("step", 0),
        "metrics": checkpoint.get("metrics", {}),
    }
