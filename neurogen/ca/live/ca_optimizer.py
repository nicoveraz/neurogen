"""CA as a learned optimizer that sees both weights and gradients."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from neurogen.ca.live.base import LiveCA


class CAOptimizer(LiveCA):
    """CA-based learned optimizer that processes weights and gradients.

    Unlike other live CA rules that produce a blended delta, CAOptimizer
    produces the full weight update. It sees both the current weight values
    and the current gradients as its state, enabling it to learn optimizer
    behaviors (momentum, adaptive learning rates, weight decay) as emergent
    properties of the local rule.

    The rule_net takes per-cell features derived from both W and grad_W
    and outputs the weight update directly.

    Features per cell (12 total):
        From weights: center_val, local_mean, local_std, local_max, local_min
        From gradients: grad_center, grad_local_mean, grad_local_std,
                       grad_local_max, grad_local_min
        Combined: weight_grad_product, abs_weight

    Biological analog: neuromodulation, where dopamine and serotonin
    modulate synaptic plasticity rules based on both synapse state and
    error signals.

    Args:
        neighborhood_size: Size of the local neighborhood kernel.
        hidden: Hidden dimension of the rule network.
    """

    N_FEATURES: int = 12

    def __init__(
        self,
        neighborhood_size: int = 3,
        hidden: int = 64,
    ) -> None:
        super().__init__(neighborhood_size=neighborhood_size)
        self.rule_net = nn.Sequential(
            nn.Linear(self.N_FEATURES, hidden, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(hidden, hidden, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(hidden, 1, dtype=torch.float32),
            nn.Tanh(),
        )

    def step(self, W: Tensor, grad_W: Tensor | None = None) -> Tensor:
        """Compute the full weight update from weights and gradients.

        Builds a combined feature vector from weight and gradient
        neighborhood statistics, then applies rule_net to produce
        a per-cell weight update.

        Args:
            W: Weight matrix of shape (H, W_dim).
            grad_W: Gradient at this weight, same shape as W. If None,
                gradient features are filled with zeros.

        Returns:
            delta: The full weight update (not blended), same shape as W.
                Scaled by 0.01 for stability.
        """
        # Weight neighborhood statistics
        w_stats = self._get_neighborhood_stats(W)

        if grad_W is None:
            grad_W = torch.zeros_like(W)

        # Gradient neighborhood statistics
        g_stats = self._get_neighborhood_stats(grad_W)

        # Build feature stack: (H, W_dim, N_FEATURES)
        features = torch.stack(
            [
                W,                          # center_val
                w_stats["local_mean"],
                w_stats["local_std"],
                w_stats["local_max"],
                w_stats["local_min"],
                grad_W,                     # grad_center
                g_stats["local_mean"],      # grad_local_mean
                g_stats["local_std"],       # grad_local_std
                g_stats["local_max"],       # grad_local_max
                g_stats["local_min"],       # grad_local_min
                W * grad_W,                 # weight_grad_product
                W.abs(),                    # abs_weight
            ],
            dim=-1,
        )

        # Apply rule_net to every cell
        delta = self.rule_net(features).squeeze(-1)

        # Scale for stability
        return delta * 0.01
