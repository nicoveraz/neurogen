"""Training loop for MicroGPT with metrics collection."""

import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from neurogen.config import GPTConfig, TrainConfig
from neurogen.data.shakespeare import ShakespeareDataset
from neurogen.model.gpt import GPT


def get_lr(step: int, config: TrainConfig) -> float:
    """Compute learning rate with linear warmup + cosine decay.

    Args:
        step: Current training step.
        config: Training configuration.

    Returns:
        Learning rate for this step.
    """
    # Linear warmup
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    # Cosine decay
    decay_ratio = (step - config.warmup_steps) / max(
        1, config.max_steps - config.warmup_steps
    )
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.lr - config.min_lr)


@torch.no_grad()
def evaluate(
    model: GPT,
    dataset: ShakespeareDataset,
    config: TrainConfig,
) -> dict[str, float]:
    """Evaluate model on train and val splits.

    Args:
        model: The model to evaluate.
        dataset: The dataset.
        config: Training configuration.

    Returns:
        Dict with 'train_loss' and 'val_loss'.
    """
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(config.eval_steps):
            x, y = dataset.get_batch(
                split, config.batch_size, model.config.block_size, config.device
            )
            _, loss = model(x, y)
            losses.append(loss.item())
        out[f"{split}_loss"] = sum(losses) / len(losses)
    model.train()
    return out


def train(
    model: GPT,
    dataset: ShakespeareDataset,
    config: TrainConfig,
) -> dict:
    """Train the model and collect metrics.

    Args:
        model: The GPT model to train.
        dataset: The Shakespeare dataset.
        config: Training configuration.

    Returns:
        Dict of collected metrics (loss curves, gradient norms, etc.).
    """
    model.to(config.device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    metrics: dict = {
        "train_loss": [],
        "val_loss": [],
        "gradient_norm": [],
        "learning_rate": [],
        "step_times": [],
    }

    start_time = time.time()

    for step in range(config.max_steps):
        step_start = time.time()

        # Set learning rate
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get batch and forward pass
        x, y = dataset.get_batch(
            "train", config.batch_size, model.config.block_size, config.device
        )
        _, loss = model(x, y)

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf loss detected at step {step}!")
            metrics["nan_detected_at_step"] = step
            break

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        if config.grad_clip > 0:
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

        optimizer.step()

        step_time = time.time() - step_start

        # Log training metrics
        if step % config.log_interval == 0:
            metrics["train_loss"].append({"step": step, "loss": loss.item()})
            metrics["gradient_norm"].append(
                {"step": step, "grad_norm": grad_norm.item()}
            )
            metrics["learning_rate"].append({"step": step, "lr": lr})
            metrics["step_times"].append({"step": step, "time": step_time})

        # Evaluate
        if step % config.eval_interval == 0 or step == config.max_steps - 1:
            eval_metrics = evaluate(model, dataset, config)
            metrics["val_loss"].append(
                {"step": step, "val_loss": eval_metrics["val_loss"]}
            )
            print(
                f"step {step:5d} | "
                f"train_loss {loss.item():.4f} | "
                f"val_loss {eval_metrics['val_loss']:.4f} | "
                f"lr {lr:.2e} | "
                f"grad_norm {grad_norm.item():.4f}"
            )

    total_time = time.time() - start_time
    metrics["total_train_time_s"] = total_time

    # Final metrics
    if metrics["val_loss"]:
        metrics["final_val_loss"] = metrics["val_loss"][-1]["val_loss"]
        metrics["best_val_loss"] = min(v["val_loss"] for v in metrics["val_loss"])
    else:
        metrics["final_val_loss"] = None
        metrics["best_val_loss"] = None

    return metrics


def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict,
    path: str | Path,
) -> None:
    """Save a training checkpoint.

    Args:
        model: The model.
        optimizer: The optimizer.
        step: Current training step.
        metrics: Collected metrics.
        path: Path to save the checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": model.config,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> dict:
    """Load a training checkpoint.

    Args:
        path: Path to the checkpoint.
        device: Device to load tensors onto.

    Returns:
        Checkpoint dict with model_state_dict, optimizer_state_dict, etc.
    """
    return torch.load(path, map_location=device, weights_only=False)
