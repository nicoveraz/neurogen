"""Reaction-Diffusion genome for weight development.

Implements Gray-Scott, FitzHugh-Nagumo, and Brusselator models.
Two coupled fields (activator/inhibitor) produce Turing patterns
that serve as structured weight initializations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neurogen.ca.genome import CAGenome


class ReactionDiffusionGenome(CAGenome):
    """Reaction-Diffusion genome using coupled PDE systems.

    Learnable parameters control diffusion rates and reaction kinetics.
    The final activator field is used as the weight matrix.
    """

    SUPPORTED_MODELS: tuple[str, ...] = (
        "gray_scott", "fitzhugh_nagumo", "brusselator"
    )

    def __init__(
        self,
        model_type: str = "gray_scott",
        dt: float = 0.5,
        seed_pattern: str = "center",
        device: str = "cpu",
    ) -> None:
        """Initialize with model_type in ("gray_scott", "fitzhugh_nagumo", "brusselator")."""
        super().__init__(device=device)
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Supported: {self.SUPPORTED_MODELS}"
            )
        self.model_type = model_type
        self.dt = dt
        self.seed_pattern = seed_pattern

        # Learnable reaction-diffusion parameters
        if model_type == "gray_scott":
            self._init_gray_scott_params()
        elif model_type == "fitzhugh_nagumo":
            self._init_fitzhugh_nagumo_params()
        elif model_type == "brusselator":
            self._init_brusselator_params()

        # Learnable output scaling
        self.output_scale = nn.Parameter(
            torch.tensor(0.02, dtype=torch.float32)
        )

        # 3x3 Laplacian kernel (fixed, not learnable)
        laplacian = torch.tensor(
            [[0.05, 0.2, 0.05],
             [0.2, -1.0, 0.2],
             [0.05, 0.2, 0.05]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", laplacian)

        self.to(device)

    def _init_gray_scott_params(self) -> None:
        """Initialize learnable parameters for Gray-Scott model."""
        # Diffusion rates
        self.Du = nn.Parameter(torch.tensor(0.16, dtype=torch.float32))
        self.Dv = nn.Parameter(torch.tensor(0.08, dtype=torch.float32))
        # Feed rate
        self.feed = nn.Parameter(torch.tensor(0.04, dtype=torch.float32))
        # Kill rate
        self.kill = nn.Parameter(torch.tensor(0.06, dtype=torch.float32))

    def _init_fitzhugh_nagumo_params(self) -> None:
        """Initialize learnable parameters for FitzHugh-Nagumo model."""
        self.Du = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.Dv = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        # Excitability parameters
        self.a = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.epsilon = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

    def _init_brusselator_params(self) -> None:
        """Initialize learnable parameters for Brusselator model."""
        self.Du = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.Dv = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        # Reaction parameters
        self.a_param = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.b_param = nn.Parameter(torch.tensor(3.0, dtype=torch.float32))

    def _laplacian(self, field: Tensor) -> Tensor:
        """Compute discrete Laplacian with circular padding."""
        x = field.unsqueeze(0).unsqueeze(0)
        x = F.pad(x, (1, 1, 1, 1), mode="circular")
        lap = F.conv2d(x, self.laplacian_kernel)
        return lap.squeeze(0).squeeze(0)

    def _step_gray_scott(self, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Gray-Scott: du/dt = Du*lap(u) - u*v^2 + f*(1-u)."""
        Du = self.Du.abs()
        Dv = self.Dv.abs()
        feed = self.feed.abs()
        kill = self.kill.abs()

        lap_u = self._laplacian(u)
        lap_v = self._laplacian(v)

        uvv = u * v * v

        du = Du * lap_u - uvv + feed * (1.0 - u)
        dv = Dv * lap_v + uvv - (feed + kill) * v

        u_new = u + du * self.dt
        v_new = v + dv * self.dt

        return u_new.clamp(0, 1), v_new.clamp(0, 1)

    def _step_fitzhugh_nagumo(
        self, u: Tensor, v: Tensor
    ) -> tuple[Tensor, Tensor]:
        """FitzHugh-Nagumo: du/dt = Du*lap(u) + u - u^3/3 - v."""
        Du = self.Du.abs()
        Dv = self.Dv.abs()

        lap_u = self._laplacian(u)
        lap_v = self._laplacian(v)

        du = Du * lap_u + u - u.pow(3) / 3.0 - v
        dv = Dv * lap_v + self.epsilon.abs() * (u + self.a - self.b * v)

        u_new = u + du * self.dt
        v_new = v + dv * self.dt

        return u_new.clamp(-2, 2), v_new.clamp(-2, 2)

    def _step_brusselator(
        self, u: Tensor, v: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Brusselator: du/dt = Du*lap(u) + a - (b+1)*u + u^2*v."""
        Du = self.Du.abs()
        Dv = self.Dv.abs()
        a = self.a_param.abs()
        b = self.b_param.abs()

        lap_u = self._laplacian(u)
        lap_v = self._laplacian(v)

        u2v = u * u * v

        du = Du * lap_u + a - (b + 1) * u + u2v
        dv = Dv * lap_v + b * u - u2v

        u_new = u + du * self.dt
        v_new = v + dv * self.dt

        return u_new.clamp(-5, 5), v_new.clamp(-5, 5)

    def _init_fields(
        self, h: int, w: int, noise_scale: float = 0.001
    ) -> tuple[Tensor, Tensor]:
        """Initialize activator/inhibitor fields with model-dependent seeding."""
        dev = self._device_str

        if self.model_type == "gray_scott":
            # u starts at 1 everywhere, v starts at 0 with a seeded region
            u = torch.ones(h, w, dtype=torch.float32, device=dev)
            v = torch.zeros(h, w, dtype=torch.float32, device=dev)
            # Seed a region in the center with v=0.25
            ch, cw = h // 2, w // 2
            rh = max(1, h // 8)
            rw = max(1, w // 8)
            v[ch - rh : ch + rh + 1, cw - rw : cw + rw + 1] = 0.25
            u[ch - rh : ch + rh + 1, cw - rw : cw + rw + 1] = 0.5

        elif self.model_type == "fitzhugh_nagumo":
            u = torch.zeros(h, w, dtype=torch.float32, device=dev)
            v = torch.zeros(h, w, dtype=torch.float32, device=dev)
            # Seed a small perturbation
            ch, cw = h // 2, w // 2
            rh = max(1, h // 8)
            rw = max(1, w // 8)
            u[ch - rh : ch + rh + 1, cw - rw : cw + rw + 1] = 1.0

        elif self.model_type == "brusselator":
            a = self.a_param.abs().item()
            b = self.b_param.abs().item()
            # Steady state: u=a, v=b/a
            u_ss = a
            v_ss = b / max(a, 1e-6)
            u = torch.full((h, w), u_ss, dtype=torch.float32, device=dev)
            v = torch.full((h, w), v_ss, dtype=torch.float32, device=dev)

        else:
            u = torch.zeros(h, w, dtype=torch.float32, device=dev)
            v = torch.zeros(h, w, dtype=torch.float32, device=dev)

        # Add noise for seed-dependent variation
        u = u + torch.randn_like(u) * noise_scale
        v = v + torch.randn_like(v) * noise_scale

        return u, v

    def develop(
        self, seed: Tensor, target_shape: tuple[int, ...], n_steps: int = 64
    ) -> Tensor:
        """Run the RD simulation for n_steps and return the weight matrix."""
        assert len(target_shape) == 2, (
            f"ReactionDiffusion expects 2D target shape, got {target_shape}"
        )
        h, w = target_shape

        # Initialize fields
        u, v = self._init_fields(h, w, noise_scale=0.001)

        # Select step function
        step_fn = {
            "gray_scott": self._step_gray_scott,
            "fitzhugh_nagumo": self._step_fitzhugh_nagumo,
            "brusselator": self._step_brusselator,
        }[self.model_type]

        # Run simulation
        for _ in range(n_steps):
            u, v = step_fn(u, v)

        # Use activator field as weight matrix
        weights = u

        # Center and scale to reasonable magnitude
        weights = weights - weights.mean()
        w_std = weights.std().clamp(min=1e-8)
        weights = weights * (self.output_scale.abs() / w_std)

        return weights

    def forward(
        self, target_shape: tuple[int, ...], n_steps: int = 64
    ) -> Tensor:
        """Create seed and develop weight matrix."""
        seed = self.create_seed(
            target_shape, self.seed_pattern, noise_scale=0.001
        )
        return self.develop(seed, target_shape, n_steps)
