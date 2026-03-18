"""Learned CA rule where the update function is a small neural network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neurogen.ca.live.base import LiveCA


class LearnedCA(LiveCA):
    """Learned CA rule where the update is parameterized by a small MLP.

    The CA rule itself is a small neural network (the "genome"). Each cell
    in the weight matrix receives a feature vector describing its local
    context, and the rule_net maps this to a per-cell update delta.

    The rule_net uses weight sharing across all cells (same MLP applied
    everywhere), making it position-invariant like a biological developmental
    program.

    Features without gradients (n_features=5):
        center_val, local_mean, local_std, local_max, local_min

    Features with gradients (n_features=7):
        center_val, local_mean, local_std, local_max, local_min,
        grad_mean, grad_magnitude

    Supports gradient flow through rule_net for meta-learning.

    Args:
        neighborhood_size: Size of the local neighborhood kernel.
        n_features: Number of input features per cell (5 without grad, 7 with).
        hidden: Hidden dimension of the rule network.
    """

    def __init__(
        self,
        neighborhood_size: int = 3,
        n_features: int = 5,
        hidden: int = 64,
    ) -> None:
        super().__init__(neighborhood_size=neighborhood_size)
        self.n_features = n_features
        self.rule_net = nn.Sequential(
            nn.Linear(n_features, hidden, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(hidden, hidden, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(hidden, 1, dtype=torch.float32),
            nn.Tanh(),
        )

    def _get_grad_neighborhood_stats(
        self, grad_W: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute gradient neighborhood statistics.

        Args:
            grad_W: Gradient tensor of shape (H, W_dim).

        Returns:
            Tuple of (grad_mean, grad_magnitude) each of shape (H, W_dim).
        """
        pad = self.k // 2
        grad_4d = grad_W.unsqueeze(0).unsqueeze(0)
        grad_padded_4d = F.pad(grad_4d, [pad, pad, pad, pad], mode="circular")
        grad_padded = grad_padded_4d.squeeze(0).squeeze(0)
        grad_neigh = grad_padded.unfold(0, self.k, 1).unfold(1, self.k, 1)
        grad_mean = grad_neigh.mean(dim=(-2, -1))
        grad_magnitude = grad_W.abs()
        return grad_mean, grad_magnitude

    def step(self, W: Tensor, grad_W: Tensor | None = None) -> Tensor:
        """Compute learned CA delta by applying rule_net to cell features.

        Builds a per-cell feature vector from neighborhood statistics and
        optionally gradient information, then applies the rule_net MLP to
        produce a bounded update delta.

        Args:
            W: Weight matrix of shape (H, W_dim).
            grad_W: Optional gradient at this weight. If provided and
                n_features >= 7, gradient features are included.

        Returns:
            delta: Learned update to add to weights, same shape as W.
                Scaled by 0.01 for stability.
        """
        stats = self._get_neighborhood_stats(W)

        # Build feature stack: (H, W_dim, n_features)
        features: list[Tensor] = [
            W,                    # center_val
            stats["local_mean"],
            stats["local_std"],
            stats["local_max"],
            stats["local_min"],
        ]

        if grad_W is not None and self.n_features >= 7:
            grad_mean, grad_magnitude = self._get_grad_neighborhood_stats(
                grad_W
            )
            features.append(grad_mean)
            features.append(grad_magnitude)

        feature_stack = torch.stack(features[:self.n_features], dim=-1)

        # Apply rule_net to every cell (weight sharing)
        delta = self.rule_net(feature_stack).squeeze(-1)

        # Scale output for stability
        return delta * 0.01
