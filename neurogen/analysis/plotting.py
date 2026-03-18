"""Plotting functions for weight analysis and training curves."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_loss_curves(
    curves: dict[str, list[dict]],
    title: str = "Training Loss Curves",
    save_path: str | Path | None = None,
) -> None:
    """Plot loss curves for multiple runs.

    Args:
        curves: Dict mapping labels to list of {"step": int, "val_loss": float}.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, data in curves.items():
        steps = [d["step"] for d in data]
        losses = [d["val_loss"] for d in data]
        ax.plot(steps, losses, label=label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_weight_heatmap(
    weight: torch.Tensor,
    title: str = "Weight Matrix",
    save_path: str | Path | None = None,
) -> None:
    """Plot a weight matrix as a heatmap.

    Args:
        weight: 2D weight tensor.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    w = weight.detach().cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = max(abs(w.min()), abs(w.max()))
    im = ax.imshow(w, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Input dimension")
    ax.set_ylabel("Output dimension")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_singular_values(
    spectra: dict[str, list[float]],
    title: str = "Singular Value Spectrum",
    save_path: str | Path | None = None,
) -> None:
    """Plot singular value spectra for multiple weight matrices.

    Args:
        spectra: Dict mapping labels to lists of singular values.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, svs in spectra.items():
        ax.semilogy(range(len(svs)), svs, label=label, alpha=0.8)
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular Value (log scale)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_comparison_bars(
    data: dict[str, float],
    title: str = "Initialization Comparison",
    ylabel: str = "Value",
    save_path: str | Path | None = None,
) -> None:
    """Plot a bar chart comparing metrics across initializers.

    Args:
        data: Dict mapping labels to values.
        title: Plot title.
        ylabel: Y-axis label.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(data.keys())
    values = list(data.values())
    bars = ax.bar(range(len(names)), values)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_gradient_flow(
    grad_norms: dict[str, list[dict]],
    title: str = "Gradient Norms Over Training",
    save_path: str | Path | None = None,
) -> None:
    """Plot gradient norm evolution during training.

    Args:
        grad_norms: Dict mapping labels to list of {"step": int, "grad_norm": float}.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, data in grad_norms.items():
        steps = [d["step"] for d in data]
        norms = [d["grad_norm"] for d in data]
        ax.plot(steps, norms, label=label, alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
