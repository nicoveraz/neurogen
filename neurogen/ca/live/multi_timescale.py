"""Multi-timescale CA that combines multiple rules at different frequencies."""

from __future__ import annotations

import torch
from torch import Tensor

from neurogen.ca.live.base import LiveCA


class MultiTimescaleCA(LiveCA):
    """Combines multiple CA rules operating at different frequencies.

    Mimics the brain's multi-process dynamics where different structural
    plasticity mechanisms operate at different timescales:
    - Fast CA: small corrections every step (e.g., homeostatic scaling)
    - Medium CA: structural adjustments periodically (e.g., modularity)
    - Slow CA: large-scale reorganization rarely (e.g., pruning)

    Each CA is applied only when the current training step is a multiple
    of its interval, and scaled by its own alpha coefficient.

    Biological analog: the full stack of synaptic scaling (fast),
    structural plasticity (medium), and myelination/reorganization (slow).

    Args:
        fast_ca: CA rule applied at high frequency.
        medium_ca: CA rule applied at medium frequency.
        slow_ca: CA rule applied at low frequency.
        fast_interval: Apply fast_ca every N steps.
        medium_interval: Apply medium_ca every N steps.
        slow_interval: Apply slow_ca every N steps.
        alpha_fast: Scaling factor for fast CA deltas.
        alpha_medium: Scaling factor for medium CA deltas.
        alpha_slow: Scaling factor for slow CA deltas.
    """

    def __init__(
        self,
        fast_ca: LiveCA,
        medium_ca: LiveCA,
        slow_ca: LiveCA,
        fast_interval: int = 1,
        medium_interval: int = 100,
        slow_interval: int = 1000,
        alpha_fast: float = 1.0,
        alpha_medium: float = 1.0,
        alpha_slow: float = 1.0,
    ) -> None:
        # Use neighborhood_size from fast_ca as the "default"
        super().__init__(neighborhood_size=fast_ca.k)
        self.fast_ca = fast_ca
        self.medium_ca = medium_ca
        self.slow_ca = slow_ca
        self.fast_interval = fast_interval
        self.medium_interval = medium_interval
        self.slow_interval = slow_interval
        self.alpha_fast = alpha_fast
        self.alpha_medium = alpha_medium
        self.alpha_slow = alpha_slow

    def step(
        self,
        W: Tensor,
        grad_W: Tensor | None = None,
        step_number: int = 0,
    ) -> Tensor:
        """Compute combined multi-timescale CA delta.

        Checks which CAs should fire at the current step_number and
        accumulates their scaled deltas.

        Args:
            W: Weight matrix of shape (H, W_dim).
            grad_W: Optional gradient at this weight.
            step_number: Current training step, used to determine which
                CAs fire.

        Returns:
            delta: Combined update from all active CAs, same shape as W.
        """
        delta = torch.zeros_like(W)

        # Fast CA: high-frequency corrections
        if step_number % self.fast_interval == 0:
            delta = delta + self.alpha_fast * self.fast_ca.step(W, grad_W)

        # Medium CA: periodic structural adjustments
        if step_number % self.medium_interval == 0:
            delta = delta + self.alpha_medium * self.medium_ca.step(W, grad_W)

        # Slow CA: rare large-scale reorganization
        if step_number % self.slow_interval == 0:
            delta = delta + self.alpha_slow * self.slow_ca.step(W, grad_W)

        return delta
