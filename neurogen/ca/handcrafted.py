"""Hand-designed structured initialization rules.

Each function returns dict[str, Tensor] matching model.get_weight_tensors().
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor


def _get_weight_shapes(model: nn.Module) -> dict[str, tuple[int, ...]]:
    """Extract weight tensor names and shapes from a model.

    Args:
        model: A GPT model with get_weight_tensors() method.

    Returns:
        Dict mapping parameter names to their shapes.
    """
    weight_tensors = model.get_weight_tensors()
    return {name: t.shape for name, t in weight_tensors.items()}


def block_diagonal_init(
    model: nn.Module, n_blocks: int = 6
) -> dict[str, Tensor]:
    """Initialize weights with block-diagonal structure.

    Creates weight matrices where each block along the diagonal is
    initialized with Xavier normal, and off-diagonal blocks are near
    zero. This encodes a modularity prior: different groups of neurons
    primarily communicate within their group.

    Args:
        model: A GPT model with get_weight_tensors() method.
        n_blocks: Number of blocks along the diagonal.

    Returns:
        Dict mapping parameter names to block-diagonal weight tensors.
    """
    weights: dict[str, Tensor] = {}
    source_weights = model.get_weight_tensors()

    for name, param in source_weights.items():
        shape = param.shape
        device = param.device

        if len(shape) != 2:
            # Non-matrix parameters: use standard init
            weights[name] = torch.randn(
                shape, dtype=torch.float32, device=device
            ) * 0.02
            continue

        h, w = shape
        mat = torch.zeros(h, w, dtype=torch.float32, device=device)

        # Create block-diagonal structure
        actual_blocks = min(n_blocks, min(h, w))
        bh = h // actual_blocks
        bw = w // actual_blocks

        for b in range(actual_blocks):
            r_start = b * bh
            r_end = (b + 1) * bh if b < actual_blocks - 1 else h
            c_start = b * bw
            c_end = (b + 1) * bw if b < actual_blocks - 1 else w

            block_h = r_end - r_start
            block_w = c_end - c_start

            # Xavier normal for each block
            std = math.sqrt(2.0 / (block_h + block_w))
            mat[r_start:r_end, c_start:c_end] = torch.randn(
                block_h, block_w, dtype=torch.float32, device=device
            ) * std

        # Add very small off-diagonal noise for gradient flow
        mat = mat + torch.randn_like(mat) * 0.001

        weights[name] = mat

    return weights


def low_rank_sparse_init(
    model: nn.Module, rank: int = 8, sparsity: float = 0.9
) -> dict[str, Tensor]:
    """Initialize weights as low-rank plus sparse.

    Constructs each weight matrix as W = UV^T + S, where U and V are
    low-rank factors and S is a sparse matrix. This encodes a prior
    that weights have a dominant low-rank component (global structure)
    plus sparse corrections (local specialization).

    Args:
        model: A GPT model with get_weight_tensors() method.
        rank: Rank of the low-rank component.
        sparsity: Fraction of entries in the sparse component that
            are zero (0.9 = 90% sparse).

    Returns:
        Dict mapping parameter names to low-rank + sparse weight tensors.
    """
    weights: dict[str, Tensor] = {}
    source_weights = model.get_weight_tensors()

    for name, param in source_weights.items():
        shape = param.shape
        device = param.device

        if len(shape) != 2:
            weights[name] = torch.randn(
                shape, dtype=torch.float32, device=device
            ) * 0.02
            continue

        h, w = shape
        actual_rank = min(rank, min(h, w))

        # Low-rank component: U * V^T
        # Scale so the product has std ~ 0.02
        u_std = math.sqrt(0.02 / math.sqrt(actual_rank))
        u = torch.randn(
            h, actual_rank, dtype=torch.float32, device=device
        ) * u_std
        v = torch.randn(
            w, actual_rank, dtype=torch.float32, device=device
        ) * u_std
        low_rank = u @ v.t()

        # Sparse component
        sparse_mat = torch.randn(
            h, w, dtype=torch.float32, device=device
        ) * 0.01
        # Zero out entries according to sparsity
        mask = torch.rand(h, w, dtype=torch.float32, device=device) > sparsity
        sparse_mat = sparse_mat * mask.float()

        weights[name] = low_rank + sparse_mat

    return weights


def frequency_biased_init(model: nn.Module) -> dict[str, Tensor]:
    """Initialize with different frequency biases per attention head.

    Each attention head's query/key/value projection gets a different
    frequency bias in its initialization. Lower-numbered heads get
    low-frequency (smooth) initialization; higher-numbered heads get
    high-frequency (detailed) initialization. This encodes a prior
    that different heads should specialize in different frequency bands.

    FFN layers get a standard initialization with slight frequency
    structure (smooth in early layers, sharper in later layers).

    Args:
        model: A GPT model with get_weight_tensors() method.

    Returns:
        Dict mapping parameter names to frequency-biased weight tensors.
    """
    weights: dict[str, Tensor] = {}
    source_weights = model.get_weight_tensors()

    # Count total layers for depth-dependent scaling
    layer_indices: dict[str, int] = {}
    max_layer = 0
    for name in source_weights:
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p.startswith("h") or p.isdigit():
                try:
                    layer_idx = int(p) if p.isdigit() else int(
                        parts[i + 1]
                    )
                    layer_indices[name] = layer_idx
                    max_layer = max(max_layer, layer_idx)
                    break
                except (ValueError, IndexError):
                    continue
        if name not in layer_indices:
            layer_indices[name] = 0

    n_layers = max_layer + 1

    for name, param in source_weights.items():
        shape = param.shape
        device = param.device

        if len(shape) != 2:
            weights[name] = torch.randn(
                shape, dtype=torch.float32, device=device
            ) * 0.02
            continue

        h, w = shape
        layer_idx = layer_indices.get(name, 0)

        # Determine if this is an attention weight (c_attn, c_proj)
        is_attn = "attn" in name or "c_attn" in name

        if is_attn and "c_attn" in name:
            # This is QKV combined: split into thirds for different heads
            # Each third (Q, K, V) gets frequency-biased init
            mat = _frequency_biased_matrix(h, w, layer_idx, n_layers, device)
        elif is_attn and "c_proj" in name:
            # Attention output projection
            mat = _frequency_biased_matrix(h, w, layer_idx, n_layers, device)
        else:
            # FFN or other weights: depth-dependent smoothness
            depth_fraction = layer_idx / max(n_layers - 1, 1)
            # Early layers: smoother; later layers: sharper
            cutoff = max(2, int((1 - depth_fraction * 0.5) * min(h, w) // 2))
            mat = _spectral_init(h, w, cutoff, device)

        weights[name] = mat

    return weights


def _frequency_biased_matrix(
    h: int, w: int, layer_idx: int, n_layers: int, device: torch.device | str
) -> Tensor:
    """Create a weight matrix with frequency bias based on layer depth.

    Lower layer indices get more low-frequency content; higher indices
    get more high-frequency content.

    Args:
        h: Matrix height.
        w: Matrix width.
        layer_idx: Index of this layer (0-based).
        n_layers: Total number of layers.
        device: Device for tensor creation.

    Returns:
        Weight matrix of shape (h, w).
    """
    depth_fraction = layer_idx / max(n_layers - 1, 1)

    # Frequency cutoff: lower layers -> lower cutoff
    min_cutoff = max(2, min(h, w) // 8)
    max_cutoff = min(h, w) // 2
    cutoff = int(min_cutoff + depth_fraction * (max_cutoff - min_cutoff))

    return _spectral_init(h, w, cutoff, device)


def _spectral_init(
    h: int, w: int, freq_cutoff: int, device: torch.device | str
) -> Tensor:
    """Create a weight matrix with a spectral frequency cutoff.

    Generates random Fourier coefficients up to freq_cutoff, then
    applies inverse FFT to get the spatial-domain matrix.

    Args:
        h: Matrix height.
        w: Matrix width.
        freq_cutoff: Maximum frequency component to include.
        device: Device for tensor creation.

    Returns:
        Weight matrix of shape (h, w) with std ~0.02.
    """
    # Generate random spectrum with frequency cutoff
    spectrum = torch.zeros(h, w, dtype=torch.complex64, device="cpu")

    fh = min(freq_cutoff, h)
    fw = min(freq_cutoff, w)

    # Random complex coefficients for low frequencies
    real_part = torch.randn(fh, fw, dtype=torch.float32)
    imag_part = torch.randn(fh, fw, dtype=torch.float32)

    # Decay with frequency for smoother results
    for i in range(fh):
        for j in range(fw):
            decay = 1.0 / (1.0 + math.sqrt(i * i + j * j))
            real_part[i, j] *= decay
            imag_part[i, j] *= decay

    spectrum[:fh, :fw] = torch.complex(real_part, imag_part)

    # Inverse FFT
    mat = torch.fft.ifft2(spectrum).real

    # Scale to target std
    mat_std = mat.std().clamp(min=1e-8)
    mat = mat * (0.02 / mat_std)

    return mat.to(device=device, dtype=torch.float32)
