"""Neural Cellular Automata genome for weight development.

Based on Growing Neural Cellular Automata (Mordvintsev et al., 2020).
Each cell has a multi-channel hidden state. Sobel-like perception filters
sense neighboring states. A small MLP processes the perceived state and
produces a state delta. A stochastic update mask ensures not all cells
update every step. A learned projection maps the hidden state to a
scalar weight value.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neurogen.ca.genome import CAGenome


class NeuralCAGenome(CAGenome):
    """Neural CA genome based on Growing Neural Cellular Automata.

    The genome consists of:
    - Sobel-like perception filters for sensing neighbors (fixed)
    - A small MLP that processes perceived state and outputs state delta
    - A linear projection from hidden channels to scalar weight value
    - A stochastic update mask (cells update with probability p_update)

    Args:
        n_channels: Number of hidden state channels per cell.
        hidden_dim: Hidden dimension of the update MLP.
        seed_pattern: Seed initialization pattern.
        p_update: Probability that each cell updates per step.
        device: Device string.
    """

    def __init__(
        self,
        n_channels: int = 16,
        hidden_dim: int = 64,
        seed_pattern: str = "center",
        p_update: float = 0.5,
        device: str = "cpu",
    ) -> None:
        """Initialize the NeuralCAGenome.

        Args:
            n_channels: Number of hidden channels per cell.
            hidden_dim: Hidden dimension for the update MLP.
            seed_pattern: Seed initialization pattern ("center" or "random").
            p_update: Cell update probability per step.
            device: Device string ("cpu", "cuda", or "mps").
        """
        super().__init__(device=device)
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.seed_pattern = seed_pattern
        self.p_update = p_update

        # Sobel-like perception filters (3x3 kernels for x/y gradients
        # and identity). Each channel gets 3 perception outputs.
        # Total perceived channels = n_channels * 3
        self._build_perception_filters()

        perceived_channels = n_channels * 3

        # Update MLP: perceived state -> state delta
        self.update_mlp = nn.Sequential(
            nn.Linear(perceived_channels, hidden_dim, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_channels, dtype=torch.float32),
        )

        # Initialize last layer with zeros for stable initial dynamics
        nn.init.zeros_(self.update_mlp[-1].weight)
        nn.init.zeros_(self.update_mlp[-1].bias)

        # Projection from hidden state to scalar weight value
        self.weight_proj = nn.Linear(
            n_channels, 1, bias=True, dtype=torch.float32
        )
        nn.init.xavier_uniform_(self.weight_proj.weight, gain=0.1)
        nn.init.zeros_(self.weight_proj.bias)

        self.to(device)

    def _build_perception_filters(self) -> None:
        """Build Sobel-like perception filters as fixed buffers.

        Creates three 3x3 convolution kernels per channel:
        - Sobel-x: horizontal gradient filter
        - Sobel-y: vertical gradient filter
        - Identity: pass-through center value
        """
        # Sobel filters for horizontal and vertical gradients
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ) / 8.0

        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        ) / 8.0

        identity = torch.tensor(
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            dtype=torch.float32,
        )

        # Stack all three filters: (3, 1, 3, 3)
        filters = torch.stack([sobel_x, sobel_y, identity]).unsqueeze(1)
        self.register_buffer("perception_filters", filters)

    def _perceive(self, state: Tensor) -> Tensor:
        """Apply perception filters to the hidden state.

        Applies Sobel-x, Sobel-y, and identity filters to each channel
        independently using depthwise convolution with circular padding.

        Args:
            state: Hidden state of shape (1, C, H, W).

        Returns:
            Perceived state of shape (1, C*3, H, W).
        """
        # Circular padding for boundary handling
        padded = F.pad(state, (1, 1, 1, 1), mode="circular")

        # Apply each filter to each channel (depthwise)
        n_ch = state.shape[1]
        perceived_parts: list[Tensor] = []

        for f_idx in range(3):
            # Expand filter to all channels: (C, 1, 3, 3)
            filt = self.perception_filters[f_idx : f_idx + 1].expand(
                n_ch, 1, 3, 3
            )
            # Depthwise convolution (groups=n_ch)
            conv_out = F.conv2d(padded, filt, groups=n_ch)
            perceived_parts.append(conv_out)

        # Concatenate along channel dim: (1, C*3, H, W)
        return torch.cat(perceived_parts, dim=1)

    def develop(
        self, seed: Tensor, target_shape: tuple[int, ...], n_steps: int = 64
    ) -> Tensor:
        """Run the neural CA for n_steps to develop a weight matrix.

        Starting from a seeded multi-channel state, iteratively applies
        perception and update rules with stochastic masking. The final
        hidden state is projected to a scalar weight matrix.

        Args:
            seed: Seed tensor. If 2D (H, W), it is expanded to multi-channel.
                If 3D (C, H, W), used directly.
            target_shape: Shape of the output weight matrix (H, W).
            n_steps: Number of CA iteration steps.

        Returns:
            Developed weight matrix of shape (H, W).
        """
        assert len(target_shape) == 2, (
            f"NeuralCA expects 2D target shape, got {target_shape}"
        )
        h, w = target_shape

        # Initialize multi-channel state
        if seed.dim() == 2:
            # Expand 2D seed to multi-channel: first channel gets seed,
            # rest are noise
            state = torch.randn(
                1, self.n_channels, h, w,
                dtype=torch.float32, device=seed.device,
            ) * 0.001
            state[0, 0] = seed
        elif seed.dim() == 3:
            state = seed.unsqueeze(0)  # (1, C, H, W)
        else:
            state = seed.reshape(1, self.n_channels, h, w)

        for step in range(n_steps):
            # Perception: gather neighborhood information
            perceived = self._perceive(state)  # (1, C*3, H, W)

            # Reshape for MLP: (H*W, C*3)
            perceived_flat = perceived.squeeze(0).permute(1, 2, 0)
            perceived_flat = perceived_flat.reshape(h * w, -1)

            # Update MLP: (H*W, C)
            delta = self.update_mlp(perceived_flat)
            delta = delta.reshape(1, self.n_channels, h, w)

            # Stochastic update mask: not all cells update each step
            if self.training or self.p_update < 1.0:
                mask = (
                    torch.rand(1, 1, h, w, device=state.device)
                    < self.p_update
                ).float()
                delta = delta * mask

            # Apply delta
            state = state + delta * 0.1

        # Project multi-channel state to scalar weight values
        # state: (1, C, H, W) -> (H, W, C) -> (H*W, C) -> (H*W, 1)
        state_flat = state.squeeze(0).permute(1, 2, 0).reshape(h * w, -1)
        weights = self.weight_proj(state_flat)  # (H*W, 1)
        weights = weights.reshape(h, w)

        # Scale to reasonable weight magnitude
        w_std = weights.std().clamp(min=1e-8)
        weights = weights * (0.02 / w_std)

        return weights

    def forward(
        self, target_shape: tuple[int, ...], n_steps: int = 64
    ) -> Tensor:
        """Convenience forward pass that creates seed and develops.

        Args:
            target_shape: Shape of the output weight matrix (H, W).
            n_steps: Number of CA development steps.

        Returns:
            Developed weight matrix.
        """
        seed = self.create_seed(
            target_shape, self.seed_pattern, noise_scale=0.001
        )
        return self.develop(seed, target_shape, n_steps)
