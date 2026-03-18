"""Modularity-enforcing CA rule that encourages block-diagonal structure."""

from __future__ import annotations

import torch
from torch import Tensor

from neurogen.ca.live.base import LiveCA


class ModularityCA(LiveCA):
    """Encourages block-diagonal structure in weight matrices.

    Divides the weight matrix into an n_blocks x n_blocks grid. Weights
    within diagonal blocks are reinforced (strengthened), while weights
    in off-diagonal blocks are decayed (weakened). This promotes modular
    connectivity where groups of neurons form tight clusters with
    weaker inter-cluster connections.

    Does NOT require gradients to operate.

    Biological analog: cortical columns and modular organization in the
    brain, where nearby neurons form tightly connected modules.

    Args:
        neighborhood_size: Size of the local neighborhood kernel (used
            by base class but not directly by this rule's step logic).
        n_blocks: Number of blocks along each dimension.
    """

    def __init__(
        self,
        neighborhood_size: int = 3,
        n_blocks: int = 6,
    ) -> None:
        super().__init__(neighborhood_size=neighborhood_size)
        self.n_blocks = n_blocks

    def step(self, W: Tensor, grad_W: Tensor | None = None) -> Tensor:
        """Compute modularity-enforcing delta.

        Reinforces on-diagonal blocks and decays off-diagonal blocks
        to encourage block-diagonal weight structure.

        Args:
            W: Weight matrix of shape (H, W_dim).
            grad_W: Not used by this rule, accepted for interface consistency.

        Returns:
            delta: Modularity correction to add to weights, same shape as W.
        """
        H, W_dim = W.shape
        bh = H // self.n_blocks
        bw = W_dim // self.n_blocks
        delta = torch.zeros_like(W)

        for i in range(self.n_blocks):
            for j in range(self.n_blocks):
                row_start = i * bh
                row_end = (i + 1) * bh if i < self.n_blocks - 1 else H
                col_start = j * bw
                col_end = (j + 1) * bw if j < self.n_blocks - 1 else W_dim

                block = W[row_start:row_end, col_start:col_end]

                if i == j:
                    # On-diagonal blocks: reinforce (strengthen)
                    delta[row_start:row_end, col_start:col_end] = block * 0.001
                else:
                    # Off-diagonal blocks: decay (weaken)
                    delta[row_start:row_end, col_start:col_end] = -block * 0.002

        return delta
