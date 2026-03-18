"""Weight analysis utilities: spectral norms, effective rank, sparsity, etc.

All analysis functions move tensors to CPU before computing to ensure
MPS compatibility. Results are always plain Python floats or dicts.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def spectral_norm(weight: Tensor) -> float:
    """Compute spectral norm (largest singular value), MPS-safe.

    Args:
        weight: 2D weight tensor.

    Returns:
        Largest singular value as a float.
    """
    w = weight.detach().cpu().float()
    if w.dim() < 2 or min(w.shape) == 0:
        return 0.0
    s = torch.linalg.svdvals(w)
    return s[0].item()


def effective_rank(weight: Tensor) -> float:
    """Effective rank via Shannon entropy of normalized singular values."""
    w = weight.detach().cpu().float()
    if w.dim() < 2 or min(w.shape) == 0:
        return 0.0

    s = torch.linalg.svdvals(w)
    s = s[s > 1e-10]

    if len(s) == 0:
        return 0.0

    p = s / s.sum()
    entropy = -(p * p.log()).sum().item()
    return math.exp(entropy)


def sparsity(weight: Tensor, threshold: float = 1e-4) -> float:
    """Compute sparsity as fraction of near-zero entries.

    Args:
        weight: Weight tensor of any shape.
        threshold: Absolute value below which an entry is considered zero.

    Returns:
        Fraction of entries with |value| < threshold.
    """
    w = weight.detach().cpu().float()
    if w.numel() == 0:
        return 0.0
    n_zero = (w.abs() < threshold).sum().item()
    return n_zero / w.numel()


def frobenius_norm(weight: Tensor) -> float:
    """Compute Frobenius norm of a weight tensor, MPS-safe.

    Args:
        weight: Weight tensor of any shape.

    Returns:
        Frobenius norm as a float.
    """
    w = weight.detach().cpu().float()
    return torch.norm(w, p="fro").item()


def singular_value_spectrum(weight: Tensor) -> list[float]:
    """Compute the full singular value spectrum in descending order.

    Args:
        weight: 2D weight tensor.

    Returns:
        List of singular values in descending order.
    """
    w = weight.detach().cpu().float()
    if w.dim() < 2 or min(w.shape) == 0:
        return [float(w.abs().max())] if w.numel() > 0 else []
    s = torch.linalg.svdvals(w)
    return [v.item() for v in s]


def weight_statistics(weight: Tensor) -> dict[str, float]:
    """Compute summary statistics for a weight tensor.

    Args:
        weight: Weight tensor of any shape.

    Returns:
        Dictionary with keys: mean, std, min, max, spectral_norm,
        effective_rank, sparsity, frobenius.
    """
    w = weight.detach().cpu().float()
    stats: dict[str, float] = {
        "mean": w.mean().item(),
        "std": w.std().item(),
        "min": w.min().item(),
        "max": w.max().item(),
        "frobenius": frobenius_norm(weight),
        "sparsity": sparsity(weight),
    }
    if w.dim() >= 2:
        stats["spectral_norm"] = spectral_norm(weight)
        stats["effective_rank"] = effective_rank(weight)
    else:
        stats["spectral_norm"] = float(w.abs().max()) if w.numel() > 0 else 0.0
        stats["effective_rank"] = 1.0 if w.numel() > 0 else 0.0
    return stats


def condition_number(weight: Tensor) -> float:
    """Compute condition number of a weight matrix, MPS-safe.

    Args:
        weight: 2D weight tensor.

    Returns:
        Condition number (ratio of largest to smallest singular value).
    """
    w = weight.detach().cpu().float()
    if w.dim() < 2 or min(w.shape) == 0:
        return float("inf")

    s = torch.linalg.svdvals(w)
    if s[-1].item() < 1e-10:
        return float("inf")
    return (s[0] / s[-1]).item()


def analyze_weight_dict(
    weights: dict[str, Tensor],
) -> dict[str, dict[str, float]]:
    """Analyze all weight tensors in a dictionary.

    Computes spectral norm, effective rank, sparsity, Frobenius norm,
    and summary statistics for each weight tensor.

    Args:
        weights: Dictionary mapping parameter names to weight tensors.

    Returns:
        Nested dictionary: {param_name: {metric_name: value}}.
    """
    results: dict[str, dict[str, float]] = {}
    for name, w in weights.items():
        stats = weight_statistics(w)
        analysis: dict[str, float] = {
            **stats,
            "spectral_norm": spectral_norm(w),
            "frobenius_norm": frobenius_norm(w),
            "sparsity": sparsity(w),
        }
        if w.dim() >= 2:
            analysis["effective_rank"] = effective_rank(w)
            analysis["condition_number"] = condition_number(w)

        results[name] = analysis

    return results


def aggregate_analysis(
    per_layer: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Aggregate per-layer analysis into summary statistics.

    Computes mean across all layers for each metric.

    Args:
        per_layer: Per-layer analysis from analyze_weight_dict().

    Returns:
        Dictionary with mean values for each metric across layers.
    """
    if not per_layer:
        return {}

    all_metrics: dict[str, list[float]] = {}
    for layer_stats in per_layer.values():
        for metric_name, value in layer_stats.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            if value != float("inf") and value == value:
                all_metrics[metric_name].append(value)

    aggregated: dict[str, float] = {}
    for metric_name, values in all_metrics.items():
        if values:
            aggregated[f"mean_{metric_name}"] = sum(values) / len(values)
        else:
            aggregated[f"mean_{metric_name}"] = 0.0

    return aggregated


def compare_weight_sets(
    weights_a: dict[str, Tensor],
    weights_b: dict[str, Tensor],
) -> dict[str, dict[str, float]]:
    """Compare two weight dictionaries layer by layer.

    Args:
        weights_a: First weight dictionary (name -> tensor).
        weights_b: Second weight dictionary (name -> tensor).

    Returns:
        Per-layer comparison with l2_distance, cosine_similarity,
        spectral norms, effective ranks, plus an "aggregate" entry.
    """
    common = sorted(set(weights_a) & set(weights_b))
    result: dict[str, dict[str, float]] = {}
    total_l2_sq = 0.0
    cos_num = cos_da = cos_db = 0.0

    for key in common:
        a = weights_a[key].detach().cpu().float()
        b = weights_b[key].detach().cpu().float()
        l2 = torch.norm(a - b, p="fro").item()
        af, bf = a.flatten(), b.flatten()
        dot = torch.dot(af, bf).item()
        na, nb = torch.norm(af).item(), torch.norm(bf).item()
        cos = dot / (na * nb) if na > 1e-10 and nb > 1e-10 else 0.0
        result[key] = {
            "l2_distance": l2,
            "cosine_similarity": cos,
            "spectral_norm_a": spectral_norm(weights_a[key]),
            "spectral_norm_b": spectral_norm(weights_b[key]),
            "effective_rank_a": effective_rank(weights_a[key]),
            "effective_rank_b": effective_rank(weights_b[key]),
        }
        total_l2_sq += l2 ** 2
        cos_num += dot
        cos_da += na ** 2
        cos_db += nb ** 2

    agg_cos = (
        cos_num / (math.sqrt(cos_da) * math.sqrt(cos_db))
        if cos_da > 1e-10 and cos_db > 1e-10
        else 0.0
    )
    result["aggregate"] = {
        "l2_distance": math.sqrt(total_l2_sq),
        "cosine_similarity": agg_cos,
        "n_layers_compared": float(len(common)),
    }
    return result


def analyze_model_weights(
    model: object,
) -> dict[str, dict[str, float]]:
    """Full analysis of all weight tensors in a model.

    Args:
        model: GPT model with get_weight_tensors() method.

    Returns:
        Per-layer weight_statistics plus an "aggregate" summary.
    """
    weights: dict[str, Tensor] = model.get_weight_tensors()  # type: ignore[union-attr]
    result: dict[str, dict[str, float]] = {}
    all_sn: list[float] = []
    all_er: list[float] = []
    all_sp: list[float] = []
    all_std: list[float] = []

    for name in sorted(weights):
        stats = weight_statistics(weights[name])
        result[name] = stats
        all_sn.append(stats["spectral_norm"])
        all_er.append(stats["effective_rank"])
        all_sp.append(stats["sparsity"])
        all_std.append(stats["std"])

    n = len(weights)
    if n > 0:
        result["aggregate"] = {
            "n_tensors": float(n),
            "mean_of_stds": sum(all_std) / n,
            "mean_spectral_norm": sum(all_sn) / n,
            "max_spectral_norm": max(all_sn),
            "mean_effective_rank": sum(all_er) / n,
            "mean_sparsity": sum(all_sp) / n,
        }
    else:
        result["aggregate"] = {"n_tensors": 0.0}
    return result
