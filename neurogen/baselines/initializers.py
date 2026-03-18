"""Baseline initialization strategies for transformer weight matrices.

Each initializer conforms to the interface:
    def initialize(model: GPT, **kwargs) -> dict[str, torch.Tensor]
"""

import math

import torch
import torch.nn as nn

from neurogen.model.gpt import GPT


def _get_fan(tensor: torch.Tensor) -> tuple[int, int]:
    """Compute fan_in and fan_out for a weight tensor."""
    if tensor.dim() < 2:
        raise ValueError(f"Cannot compute fan for {tensor.dim()}D tensor")
    fan_in = tensor.shape[1]
    fan_out = tensor.shape[0]
    return fan_in, fan_out


def xavier_uniform(model: GPT, **kwargs) -> dict[str, torch.Tensor]:
    """Glorot uniform initialization."""
    weights = {}
    for name, param in model.get_weight_tensors().items():
        fan_in, fan_out = _get_fan(param)
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        weights[name] = torch.empty_like(param).uniform_(-bound, bound)
    return weights


def xavier_normal(model: GPT, **kwargs) -> dict[str, torch.Tensor]:
    """Glorot normal initialization."""
    weights = {}
    for name, param in model.get_weight_tensors().items():
        fan_in, fan_out = _get_fan(param)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        weights[name] = torch.empty_like(param).normal_(0, std)
    return weights


def kaiming_uniform(model: GPT, **kwargs) -> dict[str, torch.Tensor]:
    """He uniform initialization (ReLU-aware)."""
    weights = {}
    for name, param in model.get_weight_tensors().items():
        fan_in, _ = _get_fan(param)
        bound = math.sqrt(6.0 / fan_in)
        weights[name] = torch.empty_like(param).uniform_(-bound, bound)
    return weights


def kaiming_normal(model: GPT, **kwargs) -> dict[str, torch.Tensor]:
    """He normal initialization."""
    weights = {}
    for name, param in model.get_weight_tensors().items():
        fan_in, _ = _get_fan(param)
        std = math.sqrt(2.0 / fan_in)
        weights[name] = torch.empty_like(param).normal_(0, std)
    return weights


def orthogonal(model: GPT, **kwargs) -> dict[str, torch.Tensor]:
    """Orthogonal initialization (uses CPU for QR decomposition, MPS-safe)."""
    weights = {}
    for name, param in model.get_weight_tensors().items():
        # QR decomposition not supported on MPS, compute on CPU
        w = torch.empty(param.shape, dtype=param.dtype, device="cpu")
        nn.init.orthogonal_(w)
        weights[name] = w.to(param.device)
    return weights


def sparse_init(model: GPT, sparsity: float = 0.9, **kwargs) -> dict[str, torch.Tensor]:
    """Sparse initialization."""
    weights = {}
    for name, param in model.get_weight_tensors().items():
        fan_in, fan_out = _get_fan(param)
        std = 1.0 / math.sqrt(fan_in)
        w = torch.empty_like(param).normal_(0, std)
        mask = torch.rand_like(w) > sparsity
        weights[name] = w * mask
    return weights


def fixup(model: GPT, **kwargs) -> dict[str, torch.Tensor]:
    """Fixup initialization (residual-aware scaling).

    Scales residual branch weights by 1/sqrt(2*n_layer) to prevent
    signal explosion in deep residual networks.
    """
    n_layer = model.config.n_layer
    weights = {}
    for name, param in model.get_weight_tensors().items():
        fan_in, fan_out = _get_fan(param)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        w = torch.empty_like(param).normal_(0, std)
        # Scale output projections
        if "c_proj" in name:
            w *= 1.0 / math.sqrt(2 * n_layer)
        weights[name] = w
    return weights


def mimetic(model: GPT, **kwargs) -> dict[str, torch.Tensor]:
    """Mimetic initialization (identity-preserving for attention).

    Initializes attention weights to approximate identity mapping and
    FFN weights with scaled normal initialization.
    """
    n_layer = model.config.n_layer
    n_embd = model.config.n_embd
    weights = {}
    for name, param in model.get_weight_tensors().items():
        fan_in, fan_out = _get_fan(param)
        if "c_attn" in name:
            # Initialize Q, K, V projections with small values
            # so attention starts near uniform
            std = 0.02 / math.sqrt(n_layer)
            w = torch.empty_like(param).normal_(0, std)
        elif "c_proj" in name and "attn" in name:
            # Attention output projection: near-identity for residual
            w = torch.zeros_like(param)
            min_dim = min(fan_in, fan_out)
            w[:min_dim, :min_dim] = torch.eye(min_dim) * (1.0 / n_layer)
        elif "c_fc" in name:
            # FFN up-projection
            std = math.sqrt(2.0 / fan_in)
            w = torch.empty_like(param).normal_(0, std)
        elif "c_proj" in name and "mlp" in name:
            # FFN down-projection: scaled for residual
            std = 0.02 / math.sqrt(2 * n_layer)
            w = torch.empty_like(param).normal_(0, std)
        else:
            std = 0.02
            w = torch.empty_like(param).normal_(0, std)
        weights[name] = w
    return weights


def spectral_delta(model: GPT, **kwargs) -> dict[str, torch.Tensor]:
    """Identity-like initialization with spectral scaling.

    Creates near-identity matrices for square weights, and scaled
    random matrices for non-square weights, with controlled spectral norm.
    All computation done on CPU for MPS compatibility, then moved to device.
    """
    weights = {}
    for name, param in model.get_weight_tensors().items():
        fan_in, fan_out = _get_fan(param)
        if fan_in == fan_out:
            w = torch.eye(fan_in, dtype=param.dtype) + torch.empty(
                param.shape, dtype=param.dtype
            ).normal_(0, 0.01)
        else:
            w = torch.empty(param.shape, dtype=param.dtype).normal_(
                0, 1.0 / math.sqrt(max(fan_in, fan_out))
            )
        # Normalize spectral norm to ~1
        with torch.no_grad():
            s = torch.linalg.svdvals(w.float())[0].item()
            if s > 0:
                w = w / s
        weights[name] = w.to(param.device)
    return weights


# Registry of all initializers
INITIALIZERS: dict[str, callable] = {
    "xavier_uniform": xavier_uniform,
    "xavier_normal": xavier_normal,
    "kaiming_uniform": kaiming_uniform,
    "kaiming_normal": kaiming_normal,
    "orthogonal": orthogonal,
    "sparse": sparse_init,
    "fixup": fixup,
    "mimetic": mimetic,
    "spectral_delta": spectral_delta,
}


def get_initializer(name: str) -> callable:
    """Get an initialization function by name.

    Args:
        name: Name of the initialization strategy.

    Returns:
        The initializer function.

    Raises:
        KeyError: If the name is not recognized.
    """
    if name not in INITIALIZERS:
        available = ", ".join(sorted(INITIALIZERS.keys()))
        raise KeyError(f"Unknown initializer '{name}'. Available: {available}")
    return INITIALIZERS[name]


def available_initializers() -> list[str]:
    """Return list of available initializer names."""
    return sorted(INITIALIZERS.keys())
