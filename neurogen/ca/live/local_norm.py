"""Homeostatic local weight normalization CA rule."""

from __future__ import annotations

import torch
from torch import Tensor

from neurogen.ca.live.base import LiveCA


class LocalNormCA(LiveCA):
    """Homeostatic weight normalization via local statistics.

    Each cell is pushed toward its local neighborhood statistics. This
    prevents gradient explosion/vanishing at the local level by maintaining
    healthy weight scale throughout the matrix.

    The rule applies two corrections:
    - Mean correction: reduces outliers by pulling toward local mean.
    - Std correction: normalizes local scale toward a target std.

    Does NOT require gradients to operate.

    Biological analog: synaptic scaling -- neurons adjust all their
    synapses to maintain a target firing rate.

    Args:
        neighborhood_size: Size of the local neighborhood kernel.
        target_std: Target standard deviation for local neighborhoods.
    """

    def __init__(
        self,
        neighborhood_size: int = 3,
        target_std: float = 0.02,
    ) -> None:
        super().__init__(neighborhood_size=neighborhood_size)
        self.target_std = target_std

    def step(self, W: Tensor, grad_W: Tensor | None = None) -> Tensor:
        """Compute homeostatic normalization delta.

        Pushes each weight toward local statistics: reduces outliers and
        normalizes local variance toward target_std.

        Args:
            W: Weight matrix of shape (H, W_dim).
            grad_W: Not used by this rule, accepted for interface consistency.

        Returns:
            delta: Homeostatic correction to add to weights, same shape as W.
        """
        stats = self._get_neighborhood_stats(W)
        local_mean = stats["local_mean"]
        local_std = stats["local_std"]

        # Push toward local mean (reduce outliers)
        mean_correction = -0.1 * (W - local_mean)

        # Normalize local scale toward target std
        std_correction = (self.target_std / (local_std + 1e-8) - 1.0) * W * 0.01

        delta = mean_correction + std_correction
        return delta
