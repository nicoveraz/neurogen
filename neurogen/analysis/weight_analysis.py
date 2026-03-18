"""Weight analysis utilities: spectral norms, effective rank, sparsity, etc.

All analysis functions move tensors to CPU before computing to ensure
MPS compatibility (MPS doesn't support all linalg ops).
"""

import torch
import numpy as np


def spectral_norm(weight: torch.Tensor) -> float:
    """Compute spectral norm (largest singular value), MPS-safe.

    Args:
        weight: Weight tensor (2D).

    Returns:
        Spectral norm as a float.
    """
    w = weight.detach().cpu().float()
    if w.dim() < 2:
        return w.abs().max().item()
    return torch.linalg.svdvals(w)[0].item()


def effective_rank(weight: torch.Tensor) -> float:
    """Compute effective rank via Shannon entropy of normalized singular values.

    Args:
        weight: Weight tensor (2D).

    Returns:
        Effective rank (continuous value between 1 and min(m, n)).
    """
    w = weight.detach().cpu().float()
    if w.dim() < 2:
        return 1.0
    sv = torch.linalg.svdvals(w)
    sv = sv[sv > 1e-10]  # remove near-zero values
    if len(sv) == 0:
        return 0.0
    # Normalize to probability distribution
    p = sv / sv.sum()
    # Shannon entropy
    entropy = -(p * torch.log(p)).sum().item()
    return np.exp(entropy)


def sparsity(weight: torch.Tensor, threshold: float = 1e-4) -> float:
    """Compute sparsity (fraction of near-zero values).

    Args:
        weight: Weight tensor.
        threshold: Values below this threshold count as zero.

    Returns:
        Sparsity ratio in [0, 1].
    """
    w = weight.detach().cpu().float()
    return (w.abs() < threshold).float().mean().item()


def frobenius_norm(weight: torch.Tensor) -> float:
    """Compute Frobenius norm.

    Args:
        weight: Weight tensor.

    Returns:
        Frobenius norm as a float.
    """
    w = weight.detach().cpu().float()
    return torch.norm(w, p="fro").item()


def condition_number(weight: torch.Tensor) -> float:
    """Compute condition number (ratio of largest to smallest singular value).

    Args:
        weight: Weight tensor (2D).

    Returns:
        Condition number. Returns inf for rank-deficient matrices.
    """
    w = weight.detach().cpu().float()
    if w.dim() < 2:
        return 1.0
    sv = torch.linalg.svdvals(w)
    if sv[-1].item() < 1e-10:
        return float("inf")
    return (sv[0] / sv[-1]).item()


def singular_value_spectrum(weight: torch.Tensor) -> list[float]:
    """Get the full singular value spectrum.

    Args:
        weight: Weight tensor (2D).

    Returns:
        List of singular values in descending order.
    """
    w = weight.detach().cpu().float()
    if w.dim() < 2:
        return [w.abs().max().item()]
    return torch.linalg.svdvals(w).tolist()


def weight_statistics(weight: torch.Tensor) -> dict[str, float]:
    """Compute comprehensive statistics for a weight tensor.

    Args:
        weight: Weight tensor.

    Returns:
        Dict with mean, std, min, max, spectral_norm, effective_rank,
        sparsity, frobenius_norm, condition_number.
    """
    w = weight.detach().cpu().float()
    stats = {
        "mean": w.mean().item(),
        "std": w.std().item(),
        "min": w.min().item(),
        "max": w.max().item(),
        "frobenius_norm": frobenius_norm(weight),
        "sparsity": sparsity(weight),
    }
    if w.dim() >= 2:
        stats["spectral_norm"] = spectral_norm(weight)
        stats["effective_rank"] = effective_rank(weight)
        stats["condition_number"] = condition_number(weight)
    return stats


def compare_weight_sets(
    weights_a: dict[str, torch.Tensor],
    weights_b: dict[str, torch.Tensor],
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """Compare two sets of weights with comprehensive statistics.

    Args:
        weights_a: First set of weight tensors.
        weights_b: Second set of weight tensors.
        label_a: Label for first set.
        label_b: Label for second set.

    Returns:
        Dict with per-layer and aggregate comparison metrics.
    """
    results = {"per_layer": {}, "aggregate": {}}

    all_stats_a = []
    all_stats_b = []

    for name in weights_a:
        if name not in weights_b:
            continue
        stats_a = weight_statistics(weights_a[name])
        stats_b = weight_statistics(weights_b[name])
        results["per_layer"][name] = {
            label_a: stats_a,
            label_b: stats_b,
        }
        all_stats_a.append(stats_a)
        all_stats_b.append(stats_b)

    # Aggregate stats
    if all_stats_a:
        for key in ["mean", "std", "spectral_norm", "frobenius_norm"]:
            vals_a = [s[key] for s in all_stats_a if key in s]
            vals_b = [s[key] for s in all_stats_b if key in s]
            if vals_a and vals_b:
                results["aggregate"][key] = {
                    label_a: {
                        "mean": np.mean(vals_a),
                        "std": np.std(vals_a),
                    },
                    label_b: {
                        "mean": np.mean(vals_b),
                        "std": np.std(vals_b),
                    },
                }

    return results
