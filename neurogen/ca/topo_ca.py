"""Variant D: Topological CA — CA rules grow edge weights on a connectivity graph.

The graph structure encodes priors; CA rules develop the weights on edges.
"""

import torch
import torch.nn as nn

from neurogen.ca.genome import CAGenome


class TopologicalCAGenome(CAGenome):
    """Topological CA genome — develops weights as edge values on a graph.

    Args:
        hidden_dim: Width of edge update MLP.
        graph_type: Type of initial graph topology.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        graph_type: str = "grid",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.graph_type = graph_type

        # Edge update rule MLP
        # Input: (edge_weight, src_degree, dst_degree, distance, step_frac) = 5
        self.edge_update = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _create_topology(
        self, H: int, W: int, device: torch.device
    ) -> torch.Tensor:
        """Create initial connectivity pattern as adjacency matrix.

        Args:
            H: Number of output neurons (rows).
            W: Number of input neurons (cols).
            device: Device.

        Returns:
            Initial adjacency/weight matrix of shape (H, W).
        """
        if self.graph_type == "grid":
            # Grid-like: each output neuron connects to nearby input neurons
            weights = torch.zeros(H, W, device=device)
            for i in range(H):
                center = int(i * W / H)
                width = max(1, W // H)
                lo = max(0, center - width)
                hi = min(W, center + width + 1)
                weights[i, lo:hi] = 0.1
        elif self.graph_type == "modular":
            # Block-diagonal structure
            n_blocks = min(4, H, W)
            block_h = H // n_blocks
            block_w = W // n_blocks
            weights = torch.zeros(H, W, device=device)
            for b in range(n_blocks):
                h_start = b * block_h
                w_start = b * block_w
                h_end = min(H, h_start + block_h)
                w_end = min(W, w_start + block_w)
                weights[h_start:h_end, w_start:w_end] = 0.1
        elif self.graph_type == "small_world":
            # Local connections + random long-range
            weights = torch.zeros(H, W, device=device)
            for i in range(H):
                center = int(i * W / H)
                width = max(1, W // (2 * H))
                lo = max(0, center - width)
                hi = min(W, center + width + 1)
                weights[i, lo:hi] = 0.1
            # Add random long-range connections
            mask = torch.rand(H, W, device=device) < 0.02
            weights[mask] = 0.05
        else:
            # Default: sparse random
            weights = (torch.rand(H, W, device=device) < 0.1).float() * 0.1

        return weights

    def develop(
        self,
        seed: torch.Tensor | None = None,
        target_shape: tuple[int, int] = (64, 64),
        n_steps: int = 32,
    ) -> torch.Tensor:
        """Develop weights by iterating edge update rules on a graph.

        Args:
            seed: Ignored (uses graph topology).
            target_shape: Desired output shape (H, W).
            n_steps: Number of development steps.

        Returns:
            Developed weight matrix.
        """
        device = next(self.parameters()).device
        H, W = target_shape

        weights = self._create_topology(H, W, device)
        # Add seed-dependent noise
        weights = weights + torch.randn(H, W, device=device) * 0.001

        # Precompute degree-like features
        row_sums = weights.abs().sum(dim=1, keepdim=True).expand(H, W)  # src degree
        col_sums = weights.abs().sum(dim=0, keepdim=True).expand(H, W)  # dst degree

        # Distance from diagonal (normalized)
        row_idx = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1)
        col_idx = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0)
        distance = ((row_idx / H) - (col_idx / W)).abs()

        for step in range(n_steps):
            step_frac = torch.tensor(
                step / max(1, n_steps - 1), device=device, dtype=torch.float32
            )

            # Recompute degrees
            row_sums = weights.abs().sum(dim=1, keepdim=True).expand(H, W)
            col_sums = weights.abs().sum(dim=0, keepdim=True).expand(H, W)

            # Normalize degrees
            max_deg = max(row_sums.max().item(), col_sums.max().item(), 1e-8)
            row_norm = row_sums / max_deg
            col_norm = col_sums / max_deg

            # Stack features: (H, W, 5)
            features = torch.stack(
                [weights, row_norm, col_norm, distance, step_frac.expand(H, W)],
                dim=-1,
            )

            # Update all edges
            delta = self.edge_update(features).squeeze(-1)
            weights = weights + delta * 0.1

        # Scale to initialization range
        weights = weights - weights.mean()
        if weights.std() > 1e-8:
            weights = weights / weights.std() * 0.02

        return weights
