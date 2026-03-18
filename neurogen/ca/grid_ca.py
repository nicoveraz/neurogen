"""Grid-based Cellular Automaton genome for weight development.

Classic 2D grid CA where each cell is updated based on its Moore neighborhood
using a small shared MLP (the genome). The MLP processes per-cell features
(center value, neighborhood statistics, step fraction) and outputs a delta
that is added to the cell value.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neurogen.ca.genome import CAGenome


class GridCAGenome(CAGenome):
    """Grid CA genome with a small MLP update rule.

    The update rule (genome) is a shared MLP that processes local features
    for each cell and produces a per-cell delta. Features include the center
    value, neighborhood mean/std/max/min, and the current step fraction.

    Uses Moore neighborhood (3x3) with circular boundary padding and
    unfold-based neighborhood extraction for efficiency.

    Args:
        hidden_dim: Hidden dimension of the update MLP.
        seed_pattern: How to seed the initial grid ("center" or "random").
        device: Device string.
    """

    # Number of input features per cell for the MLP
    N_FEATURES: int = 6

    def __init__(
        self,
        hidden_dim: int = 64,
        seed_pattern: str = "center",
        device: str = "cpu",
    ) -> None:
        """Initialize the GridCAGenome.

        Args:
            hidden_dim: Hidden dimension for the update MLP.
            seed_pattern: Seed initialization pattern ("center" or "random").
            device: Device string ("cpu", "cuda", or "mps").
        """
        super().__init__(device=device)
        self.hidden_dim = hidden_dim
        self.seed_pattern = seed_pattern

        # Small MLP: 6 input features -> hidden -> hidden -> 1 delta
        self.update_mlp = nn.Sequential(
            nn.Linear(self.N_FEATURES, hidden_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, dtype=torch.float32),
        )

        # Initialize with small weights for stable CA dynamics
        for layer in self.update_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

        self.to(device)

    def _extract_neighborhood_features(
        self, grid: Tensor, step_fraction: float
    ) -> Tensor:
        """Extract per-cell features from the grid using Moore neighborhood.

        Uses circular padding and unfold to efficiently gather 3x3
        neighborhood values for all cells simultaneously.

        Args:
            grid: 2D grid of shape (H, W).
            step_fraction: Current step / total steps, in [0, 1].

        Returns:
            Feature tensor of shape (H*W, N_FEATURES).
        """
        h, w = grid.shape

        # Circular padding for boundary handling: pad 1 on each side
        padded = F.pad(
            grid.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
            (1, 1, 1, 1),
            mode="circular",
        )  # (1, 1, H+2, W+2)

        # Extract 3x3 patches using unfold
        # unfold(dim, size, step) -> patches along that dim
        patches = padded.squeeze(0).squeeze(0)  # (H+2, W+2)
        patches = patches.unfold(0, 3, 1).unfold(1, 3, 1)  # (H, W, 3, 3)
        patches = patches.reshape(h * w, 9)  # (H*W, 9)

        # Center value is index 4 in a 3x3 flattened patch
        center = patches[:, 4:5]  # (H*W, 1)

        # Neighborhood = all 9 cells (including center for statistics)
        neigh_mean = patches.mean(dim=1, keepdim=True)
        neigh_std = patches.std(dim=1, keepdim=True).clamp(min=1e-8)
        neigh_max = patches.max(dim=1, keepdim=True).values
        neigh_min = patches.min(dim=1, keepdim=True).values

        # Step fraction broadcast to all cells
        step_feat = torch.full(
            (h * w, 1), step_fraction,
            dtype=torch.float32, device=grid.device
        )

        # Concatenate features: (H*W, 6)
        features = torch.cat(
            [center, neigh_mean, neigh_std, neigh_max, neigh_min, step_feat],
            dim=1,
        )
        return features

    def develop(
        self, seed: Tensor, target_shape: tuple[int, ...], n_steps: int = 64
    ) -> Tensor:
        """Run the grid CA for n_steps to develop a weight matrix.

        Creates a seed grid, then iteratively applies the update MLP to
        compute per-cell deltas and evolve the grid. Returns the final
        grid as the developed weight matrix.

        Args:
            seed: Seed tensor of shape target_shape (H, W).
            target_shape: Shape of the output weight matrix (H, W).
            n_steps: Number of CA iteration steps.

        Returns:
            Developed weight matrix of shape (H, W).
        """
        assert len(target_shape) == 2, (
            f"GridCA expects 2D target shape, got {target_shape}"
        )
        h, w = target_shape

        # Ensure seed matches target shape
        if seed.shape != target_shape:
            seed = self.create_seed(
                target_shape, self.seed_pattern, noise_scale=0.001
            )

        grid = seed.clone()

        for step in range(n_steps):
            step_fraction = step / max(n_steps - 1, 1)

            # Extract per-cell features
            features = self._extract_neighborhood_features(
                grid, step_fraction
            )

            # Apply update MLP to get per-cell deltas
            delta = self.update_mlp(features)  # (H*W, 1)
            delta = delta.reshape(h, w)

            # Apply delta with a small step size for stability
            grid = grid + delta * 0.1

        # Scale output to reasonable weight magnitude (~std 0.02)
        grid_std = grid.std().clamp(min=1e-8)
        grid = grid * (0.02 / grid_std)

        return grid

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
