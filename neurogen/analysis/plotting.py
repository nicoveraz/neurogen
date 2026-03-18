"""Plotting utilities for NeuroGen experiments.

All functions save figures to disk (non-interactive Agg backend).
Uses seaborn for styling and matplotlib for rendering.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402

from neurogen.analysis.weight_analysis import singular_value_spectrum  # noqa: E402

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

_SAVE_KW = {"dpi": 150, "bbox_inches": "tight"}


def plot_loss_curves(
    metrics_list: list[dict[str, list[float]]],
    labels: list[str],
    save_path: str,
) -> None:
    """Overlay train/val loss curves from multiple runs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for key, ax, title in [
        ("train_loss", axes[0], "Training Loss"),
        ("val_loss", axes[1], "Validation Loss"),
    ]:
        for metrics, label in zip(metrics_list, labels):
            if key in metrics:
                ax.plot(range(len(metrics[key])), metrics[key], label=label, alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, **_SAVE_KW)
    plt.close(fig)


def plot_weight_heatmap(
    weight_tensor: torch.Tensor, title: str, save_path: str
) -> None:
    """Visualize a weight matrix as a heatmap (downsampled if large)."""
    w = weight_tensor.detach().cpu().float().numpy()
    if w.ndim == 1:
        w = w.reshape(1, -1)
    max_d = 128
    if w.ndim == 2:
        if w.shape[0] > max_d:
            w = w[:: w.shape[0] // max_d, :]
        if w.shape[1] > max_d:
            w = w[:, :: w.shape[1] // max_d]
    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = max(abs(float(w.min())), abs(float(w.max())), 1e-10)
    sns.heatmap(
        w, ax=ax, center=0, vmin=-vmax, vmax=vmax,
        cmap="RdBu_r", xticklabels=False, yticklabels=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, **_SAVE_KW)
    plt.close(fig)


def plot_singular_values(
    weight_tensor: torch.Tensor, title: str, save_path: str
) -> None:
    """Bar chart of the singular value spectrum."""
    svs = singular_value_spectrum(weight_tensor)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(svs)), svs, color=sns.color_palette()[0], alpha=0.8)
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular Value")
    ax.set_title(title)
    if len(svs) >= 2 and svs[-1] > 0 and svs[0] / max(svs[-1], 1e-10) > 100:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(save_path, **_SAVE_KW)
    plt.close(fig)


def plot_weight_comparison(
    analysis_a: dict[str, dict[str, float]],
    analysis_b: dict[str, dict[str, float]],
    labels: list[str],
    save_path: str,
) -> None:
    """Side-by-side bar charts comparing weight stats from two inits."""
    keys_a = {k for k in analysis_a if k != "aggregate"}
    keys_b = {k for k in analysis_b if k != "aggregate"}
    common = sorted(keys_a & keys_b)
    if not common:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No common layers to compare", ha="center", va="center")
        fig.savefig(save_path, **_SAVE_KW)
        plt.close(fig)
        return

    short = [k.split(".")[-1] if "." in k else k for k in common]
    metrics = [
        ("spectral_norm", "Spectral Norm"),
        ("effective_rank", "Effective Rank"),
        ("std", "Std Dev"),
        ("sparsity", "Sparsity"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (mkey, mlabel) in enumerate(metrics):
        ax = axes.flat[idx]
        va = [analysis_a[k].get(mkey, 0.0) for k in common]
        vb = [analysis_b[k].get(mkey, 0.0) for k in common]
        x = np.arange(len(common))
        w = 0.35
        ax.bar(x - w / 2, va, w, label=labels[0], alpha=0.8)
        ax.bar(x + w / 2, vb, w, label=labels[1], alpha=0.8)
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel)
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=8)
    fig.suptitle("Weight Initialization Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, **_SAVE_KW)
    plt.close(fig)


def plot_training_dynamics(
    metrics: dict[str, list[float]], save_path: str
) -> None:
    """Multi-panel plot of loss, grad norm, lr, and weight stats."""
    panels: list[tuple[str, list[tuple[str, str]]]] = [
        ("Loss", [("train_loss", "Train"), ("val_loss", "Val")]),
        ("Gradient Norm", [("grad_norm", "Grad Norm")]),
        ("Learning Rate", [("lr", "LR")]),
        ("Weight Statistics", [
            ("weight_std", "Weight Std"),
            ("weight_spectral_norm", "Spectral Norm"),
        ]),
    ]
    active = [
        (t, [(k, l) for k, l in s if k in metrics])
        for t, s in panels
    ]
    active = [(t, s) for t, s in active if s]
    if not active:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center")
        fig.savefig(save_path, **_SAVE_KW)
        plt.close(fig)
        return

    n = len(active)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), squeeze=False)
    for idx, (title, series) in enumerate(active):
        ax = axes.flat[idx]
        for key, label in series:
            ax.plot(range(len(metrics[key])), metrics[key], label=label, alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_title(title)
        ax.legend(fontsize=8)
    for idx in range(len(active), axes.size):
        axes.flat[idx].set_visible(False)
    fig.suptitle("Training Dynamics", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, **_SAVE_KW)
    plt.close(fig)


def plot_ca_gradient_alignment(alignments: list[float], save_path: str) -> None:
    """Plot CA-gradient cosine similarity over training steps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    steps = list(range(len(alignments)))
    pal = sns.color_palette()

    # Time series
    ax = axes[0]
    ax.plot(steps, alignments, alpha=0.4, color=pal[0], linewidth=0.5)
    if len(alignments) > 10:
        w = max(1, len(alignments) // 50)
        sm = _moving_average(alignments, w)
        ax.plot(range(len(sm)), sm, color=pal[1], linewidth=2, label=f"Smoothed (w={w})")
        ax.legend(fontsize=8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("CA-Gradient Alignment Over Training")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Histogram
    ax = axes[1]
    mean_val = float(np.mean(alignments))
    ax.hist(alignments, bins=50, alpha=0.7, color=pal[0])
    ax.axvline(x=mean_val, color=pal[1], linestyle="--", label=f"Mean = {mean_val:.4f}")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Alignments")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, **_SAVE_KW)
    plt.close(fig)


def _moving_average(data: list[float], window: int) -> list[float]:
    """Simple moving average."""
    if window <= 1:
        return data[:]
    cs = np.cumsum([0.0] + data)
    return [float((cs[i + window] - cs[i]) / window) for i in range(len(data) - window + 1)]
