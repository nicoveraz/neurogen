"""Variant A: Classic Grid CA — treats weight matrix as 2D grid of cells.

Each cell is updated based on its neighborhood statistics using a small MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurogen.ca.genome import CAGenome


class GridCAGenome(CAGenome):
    """Grid-based cellular automaton genome.

    Update rule: w[i,j] = f(neighborhood_mean, neighborhood_std, current_value, step_frac)
    where f is a small MLP shared across all cells.

    Args:
        hidden_dim: Width of the update MLP hidden layers.
        n_layers: Number of hidden layers in the update MLP.
        neighborhood: Type of neighborhood ("moore_3x3", "moore_5x5", "von_neumann").
        boundary: Boundary condition ("periodic", "zero_pad").
        seed_pattern: Initial seed pattern type.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        neighborhood: str = "moore_3x3",
        boundary: str = "periodic",
        seed_pattern: str = "center_block",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.neighborhood = neighborhood
        self.boundary = boundary
        self.seed_pattern = seed_pattern

        # Input: (neighborhood_mean, neighborhood_std, current_value, step_fraction)
        input_dim = 4
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Tanh())  # bound output
        self.update_mlp = nn.Sequential(*layers)

        # Initialize with small weights for stable development
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _get_kernel_size(self) -> int:
        if self.neighborhood == "moore_3x3":
            return 3
        elif self.neighborhood == "moore_5x5":
            return 5
        elif self.neighborhood == "von_neumann":
            return 3
        else:
            return 3

    def _get_neighborhood_stats(
        self, grid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute neighborhood mean and std for each cell.

        Args:
            grid: 2D grid of cell values, shape (H, W).

        Returns:
            Tuple of (neighborhood_mean, neighborhood_std), each shape (H, W).
        """
        k = self._get_kernel_size()
        pad = k // 2

        # Add batch and channel dims for conv2d
        g = grid.unsqueeze(0).unsqueeze(0)

        if self.boundary == "periodic":
            g = F.pad(g, (pad, pad, pad, pad), mode="circular")
        else:
            g = F.pad(g, (pad, pad, pad, pad), mode="constant", value=0)

        # Create averaging kernel
        if self.neighborhood == "von_neumann":
            kernel = torch.zeros(1, 1, k, k, device=grid.device)
            kernel[0, 0, 1, 0] = 1  # up
            kernel[0, 0, 1, 2] = 1  # down
            kernel[0, 0, 0, 1] = 1  # left
            kernel[0, 0, 2, 1] = 1  # right
            kernel[0, 0, 1, 1] = 1  # center
            n_neighbors = 5.0
        else:
            kernel = torch.ones(1, 1, k, k, device=grid.device)
            n_neighbors = float(k * k)

        # Mean
        mean = F.conv2d(g, kernel)[0, 0] / n_neighbors

        # Variance (E[X^2] - E[X]^2)
        sq_mean = F.conv2d(g**2, kernel)[0, 0] / n_neighbors
        var = (sq_mean - mean**2).clamp(min=0)
        std = var.sqrt()

        return mean, std

    def _create_seed(
        self, target_shape: tuple[int, int], device: torch.device
    ) -> torch.Tensor:
        """Create initial seed tensor.

        Args:
            target_shape: Shape of the target weight matrix.
            device: Device to create tensor on.

        Returns:
            Seed tensor of target_shape.
        """
        H, W = target_shape
        # Add small random noise to make seeds depend on torch random state
        grid = torch.randn(H, W, device=device) * 0.001

        if self.seed_pattern == "center_block":
            ch, cw = H // 2, W // 2
            size = max(1, min(H, W) // 8)
            grid[ch - size : ch + size, cw - size : cw + size] += 0.1
        elif self.seed_pattern == "diagonal":
            min_dim = min(H, W)
            for i in range(min_dim):
                grid[i, i] += 0.1
        elif self.seed_pattern == "random_sparse":
            mask = torch.rand(H, W, device=device) < 0.05
            grid[mask] += torch.randn(mask.sum().item(), device=device) * 0.1
        elif self.seed_pattern == "identity_like":
            min_dim = min(H, W)
            for i in range(min_dim):
                grid[i, i] += 1.0 / min_dim
        else:
            ch, cw = H // 2, W // 2
            grid[ch, cw] += 0.1

        return grid

    def develop(
        self,
        seed: torch.Tensor | None = None,
        target_shape: tuple[int, int] = (64, 64),
        n_steps: int = 64,
    ) -> torch.Tensor:
        """Run the grid CA for n_steps to develop a weight matrix.

        Args:
            seed: Optional initial seed tensor. If None, creates one from seed_pattern.
            target_shape: Desired output shape.
            n_steps: Number of development steps.

        Returns:
            Developed weight matrix.
        """
        device = next(self.parameters()).device

        if seed is None:
            grid = self._create_seed(target_shape, device)
        else:
            grid = seed.to(device)
            if grid.shape != target_shape:
                grid = F.interpolate(
                    grid.unsqueeze(0).unsqueeze(0),
                    size=target_shape,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]

        for step in range(n_steps):
            step_frac = torch.tensor(
                step / max(1, n_steps - 1), device=device, dtype=torch.float32
            )

            mean, std = self._get_neighborhood_stats(grid)

            # Stack features: (H, W, 4)
            features = torch.stack(
                [
                    mean,
                    std,
                    grid,
                    step_frac.expand_as(grid),
                ],
                dim=-1,
            )

            # Apply update MLP to all cells
            delta = self.update_mlp(features).squeeze(-1)

            # Residual update with scaling for stability
            grid = grid + delta * 0.1

        # Scale output to reasonable initialization range
        if grid.std() > 1e-8:
            grid = grid / grid.std() * 0.02

        return grid
