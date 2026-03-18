"""Gradient-aware dynamic pruning CA rule."""

from __future__ import annotations

import torch
from torch import Tensor

from neurogen.ca.live.base import LiveCA


class PruningCA(LiveCA):
    """Gradient-aware dynamic pruning via local importance estimation.

    Computes per-cell importance as magnitude times gradient utility.
    Weights with low local importance are decayed toward zero (pruned),
    while important weights receive a small reinforcement boost.

    REQUIRES grad_W -- returns zeros if no gradient is provided.

    Biological analog: synaptic pruning during development, where the
    brain starts overconnected and selectively removes connections
    based on local activity patterns and utility signals.

    Args:
        neighborhood_size: Size of the local neighborhood kernel.
    """

    def __init__(self, neighborhood_size: int = 3) -> None:
        super().__init__(neighborhood_size=neighborhood_size)

    def step(self, W: Tensor, grad_W: Tensor | None = None) -> Tensor:
        """Compute pruning delta based on gradient-weighted importance.

        Low-importance weights (small magnitude and small gradient) are
        decayed toward zero. High-importance weights get a small boost
        in the direction of their sign.

        Args:
            W: Weight matrix of shape (H, W_dim).
            grad_W: Gradient at this weight, same shape as W. Required
                for this rule to produce non-zero output.

        Returns:
            delta: Pruning/growth correction to add to weights, same shape as W.
                Returns zeros if grad_W is None.
        """
        if grad_W is None:
            return torch.zeros_like(W)

        # Importance = magnitude * gradient utility
        importance = W.abs() * grad_W.abs()

        # Get local importance via neighborhood mean
        pad = self.k // 2
        imp_4d = importance.unsqueeze(0).unsqueeze(0)
        padded_4d = torch.nn.functional.pad(
            imp_4d, [pad, pad, pad, pad], mode="circular"
        )
        padded_imp = padded_4d.squeeze(0).squeeze(0)
        neighborhoods = padded_imp.unfold(0, self.k, 1).unfold(1, self.k, 1)
        local_importance = neighborhoods.mean(dim=(-2, -1))

        # Threshold at local median
        threshold = local_importance.median()

        # Low-importance: decay toward zero
        decay_mask = (importance < threshold).float()
        # High-importance: small boost
        growth_mask = (importance >= threshold).float()

        delta = -0.01 * W * decay_mask
        delta = delta + 0.001 * W.sign() * growth_mask

        return delta
