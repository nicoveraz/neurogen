"""Variant B: Neural Cellular Automata — hidden state per cell with perception filters.

Based on Mordvintsev et al. "Growing Neural Cellular Automata" (2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurogen.ca.genome import CAGenome


class NeuralCAGenome(CAGenome):
    """Neural CA genome with multi-channel hidden state per cell.

    Each cell has a hidden state vector. Perception is done via
    Sobel-like filters. Update is via a small MLP with stochastic mask.

    Args:
        n_channels: Number of hidden state channels per cell.
        hidden_dim: Width of the update MLP.
        perception: Type of perception filter ("sobel", "laplacian", "learned_3x3").
        stochastic_rate: Fraction of cells updated per step.
    """

    def __init__(
        self,
        n_channels: int = 16,
        hidden_dim: int = 128,
        perception: str = "sobel",
        stochastic_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.perception_type = perception
        self.stochastic_rate = stochastic_rate

        # Perception: produces 3*n_channels features (identity + 2 gradient directions)
        if perception == "learned_3x3":
            self.perception_filters = nn.Parameter(
                torch.randn(3 * n_channels, n_channels, 3, 3) * 0.01
            )
        else:
            # Fixed Sobel-like filters (not learned)
            self.register_buffer(
                "_sobel_x",
                torch.tensor(
                    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
                )
                / 8.0,
            )
            self.register_buffer(
                "_sobel_y",
                torch.tensor(
                    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
                )
                / 8.0,
            )

        perception_channels = 3 * n_channels

        # Update MLP
        self.update_net = nn.Sequential(
            nn.Linear(perception_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_channels),
        )

        # Output projection: from hidden state to scalar weight value
        self.output_proj = nn.Linear(n_channels, 1, bias=False)

        # Initialize small for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _perceive(self, state: torch.Tensor) -> torch.Tensor:
        """Apply perception filters to sense neighborhood.

        Args:
            state: Cell states, shape (1, n_channels, H, W).

        Returns:
            Perceived features, shape (1, 3*n_channels, H, W).
        """
        if self.perception_type == "learned_3x3":
            return F.conv2d(
                F.pad(state, (1, 1, 1, 1), mode="circular"),
                self.perception_filters,
            )

        # Fixed Sobel perception
        # Identity channel
        identity = state

        # Sobel-X for each channel
        sx = self._sobel_x.reshape(1, 1, 3, 3).expand(
            self.n_channels, 1, 3, 3
        )
        sobel_x = F.conv2d(
            F.pad(state, (1, 1, 1, 1), mode="circular"),
            sx,
            groups=self.n_channels,
        )

        # Sobel-Y for each channel
        sy = self._sobel_y.reshape(1, 1, 3, 3).expand(
            self.n_channels, 1, 3, 3
        )
        sobel_y = F.conv2d(
            F.pad(state, (1, 1, 1, 1), mode="circular"),
            sy,
            groups=self.n_channels,
        )

        return torch.cat([identity, sobel_x, sobel_y], dim=1)

    def develop(
        self,
        seed: torch.Tensor | None = None,
        target_shape: tuple[int, int] = (64, 64),
        n_steps: int = 64,
    ) -> torch.Tensor:
        """Run Neural CA development.

        Args:
            seed: Optional initial seed (ignored, uses internal seed).
            target_shape: Desired output shape (H, W).
            n_steps: Number of development steps.

        Returns:
            Developed weight matrix of target_shape.
        """
        device = next(self.parameters()).device
        H, W = target_shape

        # Initialize state: small values in center, zero elsewhere
        state = torch.zeros(1, self.n_channels, H, W, device=device)
        ch, cw = H // 2, W // 2
        size = max(1, min(H, W) // 8)
        state[:, 0, ch - size : ch + size, cw - size : cw + size] = 1.0

        for step in range(n_steps):
            # Perceive neighborhood
            perceived = self._perceive(state)  # (1, 3*C, H, W)

            # Reshape for MLP: (H*W, 3*C)
            perceived_flat = perceived[0].permute(1, 2, 0).reshape(-1, 3 * self.n_channels)

            # Compute update
            delta = self.update_net(perceived_flat)  # (H*W, C)
            delta = delta.reshape(H, W, self.n_channels).permute(2, 0, 1).unsqueeze(0)

            # Stochastic update mask
            if self.training or self.stochastic_rate < 1.0:
                mask = (
                    torch.rand(1, 1, H, W, device=device) < self.stochastic_rate
                ).float()
                delta = delta * mask

            # Residual update
            state = state + delta * 0.1

        # Project to scalar weight values
        output = state[0].permute(1, 2, 0)  # (H, W, C)
        weight = self.output_proj(output).squeeze(-1)  # (H, W)

        # Scale to reasonable initialization range
        if weight.std() > 1e-8:
            weight = weight / weight.std() * 0.02

        return weight
