"""Baseline weight initialization strategies for MicroGPT.

Provides 9 initialization methods conforming to the standard interface:
    def initialize(model, config=None) -> dict[str, torch.Tensor]

Each initializer generates weight tensors matching the keys from
model.get_weight_tensors() and returns them on the same device as
the model parameters.
"""

import math
from typing import Callable

import torch
import torch.nn as nn


def _get_model_info(model: nn.Module) -> tuple[dict[str, torch.Tensor], int, str]:
    """Extract weight tensors, layer count, and device from a model."""
    weights = model.get_weight_tensors()
    n_layers = model.config.n_layer
    device = next(model.parameters()).device
    return weights, n_layers, str(device)


def _simple_init(
    model: nn.Module, init_fn: Callable, **kwargs: object
) -> dict[str, torch.Tensor]:
    """Apply a torch.nn.init function to all weight tensors."""
    weights, _, _ = _get_model_info(model)
    result: dict[str, torch.Tensor] = {}
    for name, param in weights.items():
        new_w = torch.empty_like(param)
        init_fn(new_w, **kwargs)
        result[name] = new_w
    return result


def xavier_uniform_init(
    model: nn.Module, config: str | None = None
) -> dict[str, torch.Tensor]:
    """Glorot uniform: U(-a, a), a = gain * sqrt(6 / (fan_in + fan_out))."""
    return _simple_init(model, nn.init.xavier_uniform_)


def xavier_normal_init(
    model: nn.Module, config: str | None = None
) -> dict[str, torch.Tensor]:
    """Glorot normal: N(0, gain * sqrt(2 / (fan_in + fan_out)))."""
    return _simple_init(model, nn.init.xavier_normal_)


def kaiming_uniform_init(
    model: nn.Module, config: str | None = None
) -> dict[str, torch.Tensor]:
    """He uniform for ReLU/GELU layers."""
    return _simple_init(model, nn.init.kaiming_uniform_, a=math.sqrt(5))


def kaiming_normal_init(
    model: nn.Module, config: str | None = None
) -> dict[str, torch.Tensor]:
    """He normal for ReLU/GELU layers."""
    return _simple_init(
        model, nn.init.kaiming_normal_, a=0, mode="fan_in", nonlinearity="relu"
    )


def orthogonal_init(
    model: nn.Module, config: str | None = None
) -> dict[str, torch.Tensor]:
    """Orthogonal matrices via QR decomposition (computed on CPU for MPS)."""
    weights, _, device = _get_model_info(model)
    result: dict[str, torch.Tensor] = {}
    for name, param in weights.items():
        cpu_w = torch.empty(param.shape, dtype=torch.float32, device="cpu")
        nn.init.orthogonal_(cpu_w)
        result[name] = cpu_w.to(device)
    return result


def sparse_init(
    model: nn.Module, config: str | None = None
) -> dict[str, torch.Tensor]:
    """Sparse init: 90% zeros, non-zero entries ~ N(0, 0.01)."""
    weights, _, device = _get_model_info(model)
    result: dict[str, torch.Tensor] = {}
    for name, param in weights.items():
        new_w = torch.randn(param.shape, dtype=torch.float32, device=device) * 0.01
        mask = (torch.rand(param.shape, device=device) > 0.9).float()
        result[name] = new_w * mask
    return result


def fixup_init(
    model: nn.Module, config: str | None = None
) -> dict[str, torch.Tensor]:
    """Fixup: Xavier normal, with c_proj layers scaled by 1/sqrt(n_layers)."""
    weights, n_layers, _ = _get_model_info(model)
    scale = 1.0 / math.sqrt(n_layers)
    result: dict[str, torch.Tensor] = {}
    for name, param in weights.items():
        new_w = torch.empty_like(param)
        nn.init.xavier_normal_(new_w)
        if "c_proj" in name:
            new_w.mul_(scale)
        result[name] = new_w
    return result


def mimetic_init(
    model: nn.Module, config: str | None = None
) -> dict[str, torch.Tensor]:
    """Identity-like init for attention, small random for FFN.

    c_attn gets Q/K/V identity blocks + noise, c_proj near-zero,
    c_fc and embeddings get small random values.
    """
    weights, _, device = _get_model_info(model)
    n_embd = model.config.n_embd
    result: dict[str, torch.Tensor] = {}
    for name, param in weights.items():
        shape = param.shape
        if "c_attn" in name:
            new_w = torch.zeros(shape, dtype=torch.float32, device=device)
            eye = torch.eye(n_embd, dtype=torch.float32, device=device)
            rows = min(shape[0], n_embd)
            cols = min(shape[1], n_embd)
            for i in range(3):  # Q, K, V blocks
                off = i * n_embd
                if off + rows <= shape[0]:
                    new_w[off : off + rows, :cols] = eye[:rows, :cols]
            new_w += torch.randn_like(new_w) * 0.01
        elif "c_proj" in name:
            new_w = torch.randn(shape, dtype=torch.float32, device=device) * 0.001
        else:
            new_w = torch.randn(shape, dtype=torch.float32, device=device) * 0.02
        result[name] = new_w
    return result


def spectral_delta_init(
    model: nn.Module, config: str | None = None
) -> dict[str, torch.Tensor]:
    """Identity + small perturbation, scaled by spectral norm (CPU for MPS)."""
    weights, _, device = _get_model_info(model)
    result: dict[str, torch.Tensor] = {}
    for name, param in weights.items():
        r, c = param.shape[0], param.shape[1]
        # Build truncated identity + perturbation on CPU
        cpu_w = torch.zeros(r, c, dtype=torch.float32, device="cpu")
        min_dim = min(r, c)
        cpu_w[:min_dim, :min_dim] = torch.eye(min_dim, dtype=torch.float32)
        cpu_w += torch.randn(r, c, dtype=torch.float32) * 0.01
        # Normalize by spectral norm
        if r >= 2 and c >= 2:
            sigma = float(torch.linalg.svdvals(cpu_w)[0])
            if sigma > 0:
                cpu_w = cpu_w / sigma
        result[name] = cpu_w.to(device)
    return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

INITIALIZERS: dict[str, Callable[..., dict[str, torch.Tensor]]] = {
    "xavier_uniform": xavier_uniform_init,
    "xavier_normal": xavier_normal_init,
    "kaiming_uniform": kaiming_uniform_init,
    "kaiming_normal": kaiming_normal_init,
    "orthogonal": orthogonal_init,
    "sparse": sparse_init,
    "fixup": fixup_init,
    "mimetic": mimetic_init,
    "spectral_delta": spectral_delta_init,
}


def get_available_initializers() -> list[str]:
    """Return sorted list of available initialization method names."""
    return sorted(INITIALIZERS.keys())


def initialize(
    model: nn.Module, method_name: str, config: str | None = None
) -> dict[str, torch.Tensor]:
    """Dispatch to a named initialization method.

    Args:
        model: GPT model with get_weight_tensors() interface.
        method_name: Name of the initialization strategy.
        config: Optional configuration passed to the initializer.

    Returns:
        Dictionary mapping parameter names to initialized tensors.

    Raises:
        ValueError: If method_name is not a registered initializer.
    """
    if method_name not in INITIALIZERS:
        available = ", ".join(get_available_initializers())
        raise ValueError(
            f"Unknown initializer '{method_name}'. Available: {available}"
        )
    return INITIALIZERS[method_name](model, config)
