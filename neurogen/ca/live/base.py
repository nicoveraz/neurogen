"""Base class for live CA rules operating during training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LiveCA(nn.Module):
    """Base class for live CA rules operating during training.

    Treats each weight matrix as a 2D cellular automaton grid.
    Each cell's state is the weight value, and it can see its neighborhood.
    The CA produces a delta that gets blended with the gradient update
    according to the alpha schedule.

    Args:
        neighborhood_size: Size of the local neighborhood kernel (must be odd).
    """

    def __init__(self, neighborhood_size: int = 3) -> None:
        super().__init__()
        self.k = neighborhood_size

    def _get_neighborhood_stats(self, W: Tensor) -> dict[str, Tensor]:
        """Extract local neighborhood statistics for each cell.

        Uses circular padding and unfold for efficient neighborhood extraction.
        Each cell gets statistics computed over its k x k neighborhood.

        Args:
            W: Weight matrix of shape (H, W_dim).

        Returns:
            Dictionary with keys:
                - local_mean: Mean of each cell's neighborhood, shape (H, W_dim).
                - local_std: Std of each cell's neighborhood, shape (H, W_dim).
                - local_max: Max of each cell's neighborhood, shape (H, W_dim).
                - local_min: Min of each cell's neighborhood, shape (H, W_dim).
        """
        pad = self.k // 2
        # F.pad circular mode requires 4D input for 4-element padding.
        # Unsqueeze to (1, 1, H, W_dim), pad, then squeeze back.
        W_4d = W.unsqueeze(0).unsqueeze(0)
        padded_4d = F.pad(W_4d, [pad, pad, pad, pad], mode="circular")
        padded = padded_4d.squeeze(0).squeeze(0)
        # Unfold into neighborhoods: (H, W_dim, k, k)
        neighborhoods = padded.unfold(0, self.k, 1).unfold(1, self.k, 1)

        local_mean = neighborhoods.mean(dim=(-2, -1))
        local_std = neighborhoods.std(dim=(-2, -1))
        local_max = neighborhoods.amax(dim=(-2, -1))
        local_min = neighborhoods.amin(dim=(-2, -1))

        return {
            "local_mean": local_mean,
            "local_std": local_std,
            "local_max": local_max,
            "local_min": local_min,
        }

    def step(self, W: Tensor, grad_W: Tensor | None = None) -> Tensor:
        """Compute CA update delta for weight matrix W.

        Args:
            W: Weight matrix of shape (H, W_dim).
            grad_W: Optional gradient at this weight, same shape as W.

        Returns:
            delta: Update to add to weights, same shape as W.

        Raises:
            NotImplementedError: Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement step()")
