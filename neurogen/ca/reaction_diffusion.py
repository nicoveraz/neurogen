"""Variant E: Reaction-Diffusion — Turing pattern formation for weight matrices.

Uses Gray-Scott or FitzHugh-Nagumo reaction-diffusion systems to produce
naturally modular, periodic structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurogen.ca.genome import CAGenome


class ReactionDiffusionGenome(CAGenome):
    """Reaction-Diffusion genome for weight development.

    Simulates a two-field (activator-inhibitor) reaction-diffusion system.
    Parameters (feed rate, kill rate, diffusion rates) form the genome.

    Args:
        model: RD model type ("gray_scott", "fitzhugh_nagumo", "brusselator").
        dt: Time step for integration.
    """

    def __init__(
        self,
        model: str = "gray_scott",
        dt: float = 1.0,
    ) -> None:
        super().__init__()
        self.model_type = model
        self.dt = dt

        if model == "gray_scott":
            # Gray-Scott parameters (learnable)
            self.feed_rate = nn.Parameter(torch.tensor(0.037))
            self.kill_rate = nn.Parameter(torch.tensor(0.06))
            self.Du = nn.Parameter(torch.tensor(0.16))  # diffusion of U
            self.Dv = nn.Parameter(torch.tensor(0.08))  # diffusion of V
        elif model == "fitzhugh_nagumo":
            self.a = nn.Parameter(torch.tensor(0.7))
            self.b = nn.Parameter(torch.tensor(0.8))
            self.tau = nn.Parameter(torch.tensor(12.5))
            self.Du = nn.Parameter(torch.tensor(1.0))
            self.Dv = nn.Parameter(torch.tensor(100.0))
        elif model == "brusselator":
            self.a_param = nn.Parameter(torch.tensor(4.5))
            self.b_param = nn.Parameter(torch.tensor(7.5))
            self.Du = nn.Parameter(torch.tensor(2.0))
            self.Dv = nn.Parameter(torch.tensor(16.0))

        # Output mixing weights
        self.mix_u = nn.Parameter(torch.tensor(1.0))
        self.mix_v = nn.Parameter(torch.tensor(0.0))

    def _laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """Compute discrete Laplacian with periodic boundary."""
        f = field.unsqueeze(0).unsqueeze(0)
        kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=field.dtype,
            device=field.device,
        ).reshape(1, 1, 3, 3)
        padded = F.pad(f, (1, 1, 1, 1), mode="circular")
        return F.conv2d(padded, kernel)[0, 0]

    def _step_gray_scott(
        self, u: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One step of Gray-Scott reaction-diffusion."""
        f = self.feed_rate
        k = self.kill_rate
        lap_u = self._laplacian(u)
        lap_v = self._laplacian(v)
        uvv = u * v * v
        du = self.Du * lap_u - uvv + f * (1 - u)
        dv = self.Dv * lap_v + uvv - (f + k) * v
        return u + du * self.dt, v + dv * self.dt

    def _step_fitzhugh_nagumo(
        self, u: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One step of FitzHugh-Nagumo."""
        lap_u = self._laplacian(u)
        lap_v = self._laplacian(v)
        du = self.Du * lap_u + u - u**3 - v
        dv = self.Dv * lap_v + (u - self.a - self.b * v) / self.tau
        return u + du * self.dt, v + dv * self.dt

    def _step_brusselator(
        self, u: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One step of Brusselator."""
        a = self.a_param
        b = self.b_param
        lap_u = self._laplacian(u)
        lap_v = self._laplacian(v)
        du = self.Du * lap_u + a - (b + 1) * u + u**2 * v
        dv = self.Dv * lap_v + b * u - u**2 * v
        return u + du * self.dt, v + dv * self.dt

    def develop(
        self,
        seed: torch.Tensor | None = None,
        target_shape: tuple[int, int] = (64, 64),
        n_steps: int = 200,
    ) -> torch.Tensor:
        """Run reaction-diffusion to develop a weight matrix.

        Args:
            seed: Optional initial state. If None, uses perturbed uniform state.
            target_shape: Desired output shape.
            n_steps: Number of simulation steps.

        Returns:
            Developed weight matrix.
        """
        device = next(self.parameters()).device
        H, W = target_shape

        # Initial conditions
        if self.model_type == "gray_scott":
            u = torch.ones(H, W, device=device)
            v = torch.zeros(H, W, device=device)
            # Seed: small square of V in center
            ch, cw = H // 2, W // 2
            size = max(1, min(H, W) // 8)
            v[ch - size : ch + size, cw - size : cw + size] = 0.25
            u[ch - size : ch + size, cw - size : cw + size] = 0.5
            # Add small noise
            u = u + torch.randn_like(u) * 0.01
            v = v + torch.randn_like(v) * 0.01
        else:
            u = torch.randn(H, W, device=device) * 0.1
            v = torch.randn(H, W, device=device) * 0.1

        # Select step function
        step_fn = {
            "gray_scott": self._step_gray_scott,
            "fitzhugh_nagumo": self._step_fitzhugh_nagumo,
            "brusselator": self._step_brusselator,
        }[self.model_type]

        for _ in range(n_steps):
            u, v = step_fn(u, v)
            # Clamp for stability
            u = u.clamp(-10, 10)
            v = v.clamp(-10, 10)

        # Mix fields to produce weight matrix
        weight = self.mix_u * u + self.mix_v * v

        # Center and scale
        weight = weight - weight.mean()
        if weight.std() > 1e-8:
            weight = weight / weight.std() * 0.02

        return weight
