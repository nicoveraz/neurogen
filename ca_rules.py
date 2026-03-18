"""
CA rule library for NeuroGen.
Import functions from here into train.py.

Organized by biological principle (see NEUROGEN.md):
- Principles 1-4: CA Initialization (pre-training structure)
- Principles 5-7: Live CA Rules (during-training dynamics)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Utilities
# ============================================================================

def neighborhood_mean(W: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Mean of k×k neighborhood around each element."""
    W2d = W.unsqueeze(0).unsqueeze(0).float()
    kernel = torch.ones(1, 1, k, k, device=W.device) / (k * k)
    return F.conv2d(W2d, kernel, padding=k // 2).squeeze()


def neighborhood_std(W: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Std of k×k neighborhood around each element."""
    mean = neighborhood_mean(W, k)
    mean_sq = neighborhood_mean(W ** 2, k)
    var = (mean_sq - mean ** 2).clamp(min=0)
    return var.sqrt()


def rescale(W: torch.Tensor, target_std: float = 0.02) -> torch.Tensor:
    """Rescale tensor to have target standard deviation."""
    std = W.std()
    if std > 1e-8:
        W = W * (target_std / std)
    return W

# ============================================================================
# CA Development Engine
# ============================================================================

def grid_ca_develop(
    shape: tuple[int, int],
    n_steps: int = 64,
    seed: str = "center",
    neighborhood: int = 3,
    target_std: float = 0.02,
) -> torch.Tensor:
    """Grow a weight matrix using a grid cellular automaton.

    Args:
        shape: Target (rows, cols) shape.
        n_steps: Number of CA development steps.
        seed: Seed pattern ("center", "random", "diagonal", "distributed").
        neighborhood: Neighborhood size for local updates.
        target_std: Target standard deviation for output.

    Returns:
        Developed weight matrix.
    """
    h, w = shape
    grid = torch.zeros(h, w)

    # Seed the grid
    if seed == "center":
        ch, cw = h // 2, w // 2
        r = max(1, min(h, w) // 8)
        grid[ch - r : ch + r, cw - r : cw + r] = torch.randn(2 * r, 2 * r) * 0.1
    elif seed == "random":
        grid = torch.randn(h, w) * 0.01
    elif seed == "diagonal":
        for i in range(min(h, w)):
            grid[i, i] = torch.randn(1).item() * 0.1
    elif seed == "distributed":
        n_seeds = max(4, min(h, w) // 8)
        for _ in range(n_seeds):
            r, c = torch.randint(h, (1,)).item(), torch.randint(w, (1,)).item()
            grid[r, c] = torch.randn(1).item() * 0.5
    elif seed == "diagonal_band":
        band = max(1, min(h, w) // 6)
        for i in range(min(h, w)):
            lo = max(0, i - band)
            hi = min(w, i + band + 1)
            grid[i, lo:hi] = torch.randn(hi - lo) * 0.1
    elif seed == "off_diagonal":
        offset = min(h, w) // 3
        for i in range(min(h, w)):
            j = (i + offset) % w
            grid[i, j] = torch.randn(1).item() * 0.3
    elif seed == "gradient":
        grid = torch.linspace(-0.1, 0.1, h).unsqueeze(1) * torch.linspace(-0.1, 0.1, w).unsqueeze(0)

    # Run CA
    k = neighborhood
    for _ in range(n_steps):
        mean = neighborhood_mean(grid, k)
        std = neighborhood_std(grid, k)
        # Update rule: move toward local mean + noise scaled by local variation
        delta = 0.1 * (mean - grid) + 0.05 * std * torch.randn_like(grid)
        # Growth from non-zero regions
        alive = (grid.abs() > 1e-4).float()
        alive_neighbors = neighborhood_mean(alive, k)
        grow = (alive_neighbors > 0.1).float() * (1 - alive) * torch.randn_like(grid) * 0.02
        grid = grid + delta + grow

    return rescale(grid, target_std)

# ============================================================================
# Principle 1: Functional Specialization (different seeds per head)
# ============================================================================

def specialized_heads_init(
    n_heads: int, head_dim: int, n_steps: int = 32
) -> list[torch.Tensor]:
    """Each head gets a different CA seed for distinct functional bias.

    Returns list of (head_dim, head_dim) weight matrices.
    """
    seeds = ["diagonal_band", "distributed", "off_diagonal", "gradient"]
    heads = []
    for h in range(n_heads):
        seed = seeds[h % len(seeds)]
        w = grid_ca_develop((head_dim, head_dim), n_steps=n_steps, seed=seed)
        heads.append(w)
    return heads

# ============================================================================
# Principle 2: Hierarchical Processing (depth-dependent CA)
# ============================================================================

def hierarchical_init_for_layer(
    shape: tuple[int, int],
    layer_idx: int,
    n_layers: int,
    n_steps: int = 48,
) -> torch.Tensor:
    """Depth-dependent CA: local structure early, distributed structure deep.

    Args:
        shape: Target weight matrix shape.
        layer_idx: Current layer index.
        n_layers: Total number of layers.
        n_steps: Base number of CA steps.
    """
    locality = 1.0 - layer_idx / max(n_layers - 1, 1)

    if locality > 0.66:
        seed = "diagonal_band"  # local, sequential
        k = 3
    elif locality > 0.33:
        seed = "center"  # transitional
        k = 5
    else:
        seed = "distributed"  # global, abstract
        k = 5

    steps = int(n_steps * (1 + (1 - locality)))  # more steps for deeper layers
    return grid_ca_develop(shape, n_steps=steps, seed=seed, neighborhood=k)

# ============================================================================
# Principle 3: Long-Range Connectivity (reaction-diffusion)
# ============================================================================

def reaction_diffusion_init(
    shape: tuple[int, int],
    feed: float = 0.04,
    kill: float = 0.06,
    n_steps: int = 200,
    target_std: float = 0.02,
) -> torch.Tensor:
    """Gray-Scott reaction-diffusion for Turing patterns.

    Different parameter regimes produce different structures:
    - feed=0.04, kill=0.06 → stripes (connectivity highways)
    - feed=0.03, kill=0.06 → spots (modular columns)
    - feed=0.025, kill=0.06 → branching (dendritic)
    """
    h, w = shape
    # Two coupled fields: U (activator), V (inhibitor)
    U = torch.ones(h, w)
    V = torch.zeros(h, w)

    # Seed: small perturbation in center
    ch, cw = h // 2, w // 2
    r = max(2, min(h, w) // 8)
    U[ch - r : ch + r, cw - r : cw + r] = 0.5 + torch.rand(2 * r, 2 * r) * 0.1
    V[ch - r : ch + r, cw - r : cw + r] = 0.25 + torch.rand(2 * r, 2 * r) * 0.1

    du, dv = 0.16, 0.08  # diffusion rates
    dt = 1.0

    for _ in range(n_steps):
        # Laplacian via convolution
        lap_kernel = torch.tensor([[0.05, 0.2, 0.05],
                                   [0.2, -1.0, 0.2],
                                   [0.05, 0.2, 0.05]])
        lap_kernel = lap_kernel.view(1, 1, 3, 3)
        Lu = F.conv2d(U.unsqueeze(0).unsqueeze(0), lap_kernel, padding=1).squeeze()
        Lv = F.conv2d(V.unsqueeze(0).unsqueeze(0), lap_kernel, padding=1).squeeze()

        # Gray-Scott dynamics
        uvv = U * V * V
        U = U + dt * (du * Lu - uvv + feed * (1 - U))
        V = V + dt * (dv * Lv + uvv - (feed + kill) * V)
        U = U.clamp(0, 1)
        V = V.clamp(0, 1)

    # Convert pattern to weight matrix (U - V gives structure)
    W = U - V
    return rescale(W, target_std)

# ============================================================================
# Principle 4: Modular Organization (cortical columns)
# ============================================================================

def modular_init(
    shape: tuple[int, int],
    n_modules: int = 4,
    n_steps: int = 48,
    target_std: float = 0.02,
) -> torch.Tensor:
    """Multi-seed CA producing block-diagonal structure.

    Each module grows independently, creating natural modularity.
    Light cross-module connections are added for communication.
    """
    h, w = shape
    grid = torch.zeros(h, w)
    block_h = h // n_modules
    block_w = w // n_modules

    for m in range(n_modules):
        rh = min(block_h, h - m * block_h)
        rw = min(block_w, w - m * block_w)
        if rh <= 0 or rw <= 0:
            break
        block = grid_ca_develop((rh, rw), n_steps=n_steps, seed="random")
        grid[m * block_h : m * block_h + rh, m * block_w : m * block_w + rw] = block

    # Light cross-module connections
    grid = grid + torch.randn(h, w) * 0.001

    return rescale(grid, target_std)

# ============================================================================
# Handcrafted structural priors (baselines for comparison)
# ============================================================================

def block_diagonal_init(
    shape: tuple[int, int], n_blocks: int = 4, target_std: float = 0.02
) -> torch.Tensor:
    """Direct block-diagonal structure (no CA, for comparison)."""
    h, w = shape
    W = torch.zeros(h, w)
    bh, bw = h // n_blocks, w // n_blocks
    for b in range(n_blocks):
        rh = min(bh, h - b * bh)
        rw = min(bw, w - b * bw)
        W[b * bh : b * bh + rh, b * bw : b * bw + rw] = torch.randn(rh, rw)
    return rescale(W, target_std)


def orthogonal_init(shape: tuple[int, int]) -> torch.Tensor:
    """Orthogonal initialization (computed on CPU for MPS safety)."""
    W = torch.empty(shape, dtype=torch.float32)
    nn.init.orthogonal_(W)
    return W

# ============================================================================
# Principle 5: Competition / Lateral Inhibition (live CA)
# ============================================================================

def competition_step(W: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Live CA: strongest weights suppress neighbors.

    Returns delta to apply: W += alpha * delta
    """
    W_abs = W.abs()
    if W.dim() != 2 or W.shape[0] < k or W.shape[1] < k:
        return torch.zeros_like(W)
    local_max = F.max_pool2d(
        W_abs.unsqueeze(0).unsqueeze(0), k, stride=1, padding=k // 2
    ).squeeze()
    is_winner = (W_abs >= local_max * 0.95).float()
    # Winners get reinforced, losers get suppressed
    delta = 0.001 * W * is_winner + (-0.003) * W * (1 - is_winner)
    return delta

# ============================================================================
# Principle 6: Homeostatic Regulation (live CA)
# ============================================================================

def homeostatic_step(W: torch.Tensor, target_std: float = 0.02) -> torch.Tensor:
    """Live CA: synaptic scaling to maintain healthy weight statistics.

    Returns delta to apply: W += alpha * delta
    """
    if W.dim() != 2 or W.shape[0] < 3 or W.shape[1] < 3:
        return torch.zeros_like(W)
    local_mean = neighborhood_mean(W, k=3)
    local_std = neighborhood_std(W, k=3)
    # Pull outliers toward local mean
    mean_correction = -0.1 * (W - local_mean)
    # Push local variance toward target
    std_ratio = target_std / (local_std + 1e-8)
    std_correction = (std_ratio - 1).clamp(-0.5, 0.5) * W * 0.01
    return mean_correction + std_correction


def modularity_step(W: torch.Tensor, n_blocks: int = 4) -> torch.Tensor:
    """Live CA: reinforce block-diagonal structure, decay cross-block.

    Returns delta to apply: W += alpha * delta
    """
    if W.dim() != 2:
        return torch.zeros_like(W)
    h, w = W.shape
    bh, bw = h // n_blocks, w // n_blocks
    mask = torch.zeros_like(W)
    for b in range(n_blocks):
        r0, r1 = b * bh, min((b + 1) * bh, h)
        c0, c1 = b * bw, min((b + 1) * bw, w)
        mask[r0:r1, c0:c1] = 1.0
    # Reinforce within-block, decay cross-block
    delta = 0.001 * W * mask + (-0.002) * W * (1 - mask)
    return delta

# ============================================================================
# Principle 5b: Gradient-Aware Pruning (live CA)
# ============================================================================

def pruning_step(W: torch.Tensor, grad_W: torch.Tensor | None = None) -> torch.Tensor:
    """Live CA: eliminate low-utility connections.

    If gradients available, prune weights with low gradient magnitude.
    Otherwise, prune by absolute weight magnitude.

    Returns delta to apply: W += alpha * delta
    """
    if grad_W is not None:
        utility = W.abs() * grad_W.abs()
    else:
        utility = W.abs()
    threshold = utility.quantile(0.1)  # bottom 10%
    prune_mask = (utility < threshold).float()
    # Decay pruned weights toward zero
    delta = -0.01 * W * prune_mask
    return delta

# ============================================================================
# Principle 7: Critical Period Alpha Schedules
# ============================================================================

def critical_period_alpha(
    step: int, total_steps: int, alpha_0: float = 0.01
) -> float:
    """Strong early influence, rapid closing at 20% of training."""
    critical_end = total_steps * 0.2
    if step < critical_end:
        return alpha_0 * (1.0 - step / critical_end)
    return alpha_0 * 0.01


def layerwise_critical_period(
    step: int, layer_idx: int, n_layers: int,
    total_steps: int, alpha_0: float = 0.01,
) -> float:
    """Earlier layers close their critical period first."""
    close_frac = 0.1 + 0.2 * (layer_idx / max(n_layers - 1, 1))
    critical_end = total_steps * close_frac
    if step < critical_end:
        return alpha_0 * (1.0 - step / critical_end)
    return alpha_0 * 0.01


def exponential_decay_alpha(
    step: int, total_steps: int, alpha_0: float = 0.01, decay: float = 5.0,
) -> float:
    """Simple exponential decay."""
    return alpha_0 * math.exp(-decay * step / total_steps)


def adaptive_alpha(
    step: int, alpha_0: float = 0.01, loss_history: list[float] | None = None,
) -> float:
    """Increase when loss stagnates, decrease when improving."""
    if loss_history is None or len(loss_history) < 10:
        return alpha_0
    recent = loss_history[-5:]
    older = loss_history[-10:-5]
    improvement = (sum(older) / len(older)) - (sum(recent) / len(recent))
    if improvement < 0.001:
        return alpha_0 * 2.0  # stagnating: increase CA influence
    return alpha_0 * 0.5  # improving: reduce CA influence


def cyclic_alpha(
    step: int, total_steps: int, alpha_0: float = 0.01, period: int = 500,
) -> float:
    """Periodic bursts of CA activity (sleep/wake consolidation analog)."""
    phase = (step % period) / period
    return alpha_0 * (0.5 + 0.5 * math.cos(2 * math.pi * phase))

# ============================================================================
# Learned CA Rule (genome = small MLP)
# ============================================================================

class LearnedCAGenome(nn.Module):
    """Small MLP that acts as a CA update rule.

    Input: local neighborhood statistics (mean, std, center value, step_frac)
    Output: weight delta for center cell

    Genome size is deliberately tiny (~200-500 params).
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, 1, bias=False),
        )

    def forward(self, W: torch.Tensor, step_frac: float = 0.0) -> torch.Tensor:
        """Compute weight delta using learned rule.

        Args:
            W: Weight matrix (2D).
            step_frac: Fraction of training completed (0 to 1).

        Returns:
            Delta tensor same shape as W.
        """
        if W.dim() != 2 or W.shape[0] < 3 or W.shape[1] < 3:
            return torch.zeros_like(W)
        mean = neighborhood_mean(W, k=3)
        std = neighborhood_std(W, k=3)
        step_t = torch.full_like(W, step_frac)
        features = torch.stack([W, mean, std, step_t], dim=-1)  # (H, W, 4)
        delta = self.net(features).squeeze(-1)  # (H, W)
        return delta * 0.001  # small scale for stability
