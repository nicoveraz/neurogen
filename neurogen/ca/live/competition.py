"""Lateral inhibition / winner-take-all CA rule."""

from __future__ import annotations

import torch
from torch import Tensor

from neurogen.ca.live.base import LiveCA


class CompetitionCA(LiveCA):
    """Lateral inhibition where nearby weights compete.

    The strongest weights in a local neighborhood suppress their
    neighbors, producing sparse, winner-take-all connectivity patterns.
    Uses a larger neighborhood (k=5) than default to allow for broader
    competitive dynamics.

    Does NOT require gradients to operate.

    Biological analog: lateral inhibition in cortical columns, where
    strongly active neurons suppress their neighbors to sharpen
    representations.

    Args:
        neighborhood_size: Size of the local neighborhood kernel.
            Defaults to 5 for broader competition.
    """

    def __init__(self, neighborhood_size: int = 5) -> None:
        super().__init__(neighborhood_size=neighborhood_size)

    def step(self, W: Tensor, grad_W: Tensor | None = None) -> Tensor:
        """Compute competitive delta via lateral inhibition.

        Identifies local winners (cells whose absolute value is near
        the local maximum) and strengthens them while suppressing losers.

        Args:
            W: Weight matrix of shape (H, W_dim).
            grad_W: Not used by this rule, accepted for interface consistency.

        Returns:
            delta: Competition update to add to weights, same shape as W.
        """
        stats = self._get_neighborhood_stats(W.abs())
        local_max = stats["local_max"]

        # Winner threshold: |W| >= 95% of local max
        is_winner = (W.abs() >= local_max * 0.95).float()
        is_loser = 1.0 - is_winner

        # Winners get strengthened, losers get suppressed
        delta = 0.001 * W * is_winner + (-0.005) * W * is_loser

        return delta
