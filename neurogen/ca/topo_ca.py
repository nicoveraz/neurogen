"""Topological Cellular Automata genome for weight development.

Graph-based weight development where a connectivity graph defines the
structure and CA rules grow edge weights. The adjacency structure encodes
priors (local connectivity, hierarchical modules) and maps to the final
weight matrix.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor

from neurogen.ca.genome import CAGenome


class TopologicalCAGenome(CAGenome):
    """Topological CA genome using graph-based weight development.

    The genome defines:
    - An initial graph topology (adjacency pattern)
    - An MLP update rule that evolves edge weights based on local
      node features (degree, edge weight statistics, step fraction)
    - A mapping from the adjacency matrix to the weight matrix

    The topology is initialized with a structured pattern (small-world,
    block-diagonal, or random) plus noise for seed-dependent variation.

    Args:
        hidden_dim: Hidden dimension of the edge update MLP.
        n_nodes: Number of graph nodes (if None, derived from target shape).
        topology: Initial topology type ("small_world", "block", "random").
        seed_pattern: Seed initialization pattern.
        device: Device string.
    """

    # Input features per edge: edge_weight, src_degree, dst_degree,
    # neighbor_edge_mean, neighbor_edge_std, step_fraction
    N_EDGE_FEATURES: int = 6

    def __init__(
        self,
        hidden_dim: int = 64,
        n_nodes: int | None = None,
        topology: str = "small_world",
        seed_pattern: str = "center",
        device: str = "cpu",
    ) -> None:
        """Initialize the TopologicalCAGenome.

        Args:
            hidden_dim: Hidden dimension for the edge update MLP.
            n_nodes: Number of graph nodes.
            topology: Initial topology type.
            seed_pattern: Seed initialization pattern.
            device: Device string ("cpu", "cuda", or "mps").
        """
        super().__init__(device=device)
        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes
        self.topology = topology
        self.seed_pattern = seed_pattern

        # Edge update MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(
                self.N_EDGE_FEATURES, hidden_dim, dtype=torch.float32
            ),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, dtype=torch.float32),
        )

        # Initialize with small weights
        for layer in self.edge_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

        self.to(device)

    def _create_initial_adjacency(
        self, h: int, w: int, noise_scale: float = 0.001
    ) -> Tensor:
        """Create an initial adjacency/weight matrix with structure.

        Args:
            h: Number of rows (source nodes).
            w: Number of columns (destination nodes).
            noise_scale: Scale of random noise for seed-dependent variation.

        Returns:
            Initial adjacency matrix of shape (h, w).
        """
        adj = torch.zeros(h, w, dtype=torch.float32, device=self._device_str)

        if self.topology == "small_world":
            # Start with local connectivity, then add long-range connections
            for i in range(h):
                for j in range(w):
                    # Local connections (diagonal band)
                    dist = abs(i - j * h / max(w, 1))
                    if dist < max(h, w) * 0.1:
                        adj[i, j] = 0.1 * math.exp(-dist * 0.5)
                    # Random long-range connections (sparse)
                    elif (i * 7 + j * 13) % 37 == 0:
                        adj[i, j] = 0.05

        elif self.topology == "block":
            # Block-diagonal structure
            n_blocks = min(4, min(h, w))
            bh = h // n_blocks
            bw = w // n_blocks
            for b in range(n_blocks):
                r_start = b * bh
                r_end = min((b + 1) * bh, h)
                c_start = b * bw
                c_end = min((b + 1) * bw, w)
                adj[r_start:r_end, c_start:c_end] = 0.1

        elif self.topology == "random":
            adj = torch.randn(
                h, w, dtype=torch.float32, device=self._device_str
            ) * 0.05

        else:
            # Default: sparse random
            adj = torch.randn(
                h, w, dtype=torch.float32, device=self._device_str
            ) * 0.02

        # Add noise so different seeds produce different outputs
        noise = torch.randn_like(adj) * noise_scale
        adj = adj + noise

        return adj

    def _compute_edge_features(
        self, adj: Tensor, step_fraction: float
    ) -> Tensor:
        """Compute per-edge features from the adjacency matrix.

        Args:
            adj: Adjacency matrix of shape (H, W).
            step_fraction: Current step fraction in [0, 1].

        Returns:
            Edge feature tensor of shape (H*W, N_EDGE_FEATURES).
        """
        h, w = adj.shape

        # Edge weights flattened
        edge_weights = adj.reshape(-1, 1)

        # Source node degree (sum of outgoing edge weights per row)
        src_degree = adj.abs().sum(dim=1, keepdim=True)  # (H, 1)
        src_degree = src_degree.expand(h, w).reshape(-1, 1)

        # Destination node degree (sum of incoming edge weights per col)
        dst_degree = adj.abs().sum(dim=0, keepdim=True)  # (1, W)
        dst_degree = dst_degree.expand(h, w).reshape(-1, 1)

        # Neighbor edge statistics: for each edge (i,j), consider
        # edges from row i (same source) as neighbors
        row_means = adj.mean(dim=1, keepdim=True).expand(h, w).reshape(-1, 1)
        row_stds = (
            adj.std(dim=1, keepdim=True).clamp(min=1e-8)
            .expand(h, w).reshape(-1, 1)
        )

        # Step fraction
        step_feat = torch.full(
            (h * w, 1), step_fraction,
            dtype=torch.float32, device=adj.device
        )

        return torch.cat(
            [edge_weights, src_degree, dst_degree,
             row_means, row_stds, step_feat],
            dim=1,
        )

    def develop(
        self, seed: Tensor, target_shape: tuple[int, ...], n_steps: int = 64
    ) -> Tensor:
        """Run the topological CA to develop a weight matrix.

        Initializes a structured adjacency matrix, then iteratively
        updates edge weights using the edge MLP. The final adjacency
        matrix IS the weight matrix.

        Args:
            seed: Seed tensor (used for noise seeding).
            target_shape: Shape of the output weight matrix (H, W).
            n_steps: Number of CA iteration steps.

        Returns:
            Developed weight matrix of shape (H, W).
        """
        assert len(target_shape) == 2, (
            f"TopoCA expects 2D target shape, got {target_shape}"
        )
        h, w = target_shape

        # Initialize adjacency matrix with structured topology
        adj = self._create_initial_adjacency(h, w, noise_scale=0.001)

        for step in range(n_steps):
            step_fraction = step / max(n_steps - 1, 1)

            # Compute per-edge features
            features = self._compute_edge_features(adj, step_fraction)

            # Apply edge MLP to get deltas
            delta = self.edge_mlp(features)  # (H*W, 1)
            delta = delta.reshape(h, w)

            # Apply delta
            adj = adj + delta * 0.1

        # Scale to reasonable magnitude
        adj_std = adj.std().clamp(min=1e-8)
        adj = adj * (0.02 / adj_std)

        return adj

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
