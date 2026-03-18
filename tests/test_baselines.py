"""Tests for neurogen/baselines/initializers.py."""

import math

import pytest
import torch

from neurogen.baselines.initializers import (
    INITIALIZERS,
    get_available_initializers,
    initialize,
    xavier_uniform_init,
    xavier_normal_init,
    kaiming_uniform_init,
    kaiming_normal_init,
    orthogonal_init,
    sparse_init,
    fixup_init,
    mimetic_init,
    spectral_delta_init,
)
from neurogen.config import GPTConfig
from neurogen.model.gpt import GPT


EXPECTED_INITIALIZERS = [
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "orthogonal",
    "sparse",
    "fixup",
    "mimetic",
    "spectral_delta",
]


class TestBaselineRegistry:
    """Tests for initializer registry."""

    def test_all_baselines_registered(self):
        """All 9 methods should be in INITIALIZERS."""
        for name in EXPECTED_INITIALIZERS:
            assert name in INITIALIZERS, f"'{name}' should be registered in INITIALIZERS"
        assert len(INITIALIZERS) == 9, (
            f"Expected 9 initializers, found {len(INITIALIZERS)}"
        )

    def test_get_available_initializers(self):
        """get_available_initializers returns sorted list of names."""
        available = get_available_initializers()
        assert available == sorted(available), "Should return sorted list"
        assert len(available) == 9, f"Should have 9, got {len(available)}"

    def test_initialize_dispatch(self, tiny_model):
        """initialize() dispatches correctly by name."""
        weights = initialize(tiny_model, "xavier_normal")
        assert len(weights) > 0, "Should return non-empty weight dict"

    def test_initialize_unknown_raises(self, tiny_model):
        """Unknown initializer name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown initializer"):
            initialize(tiny_model, "nonexistent_init")


class TestBaselineShapes:
    """Tests for output shapes."""

    @pytest.mark.parametrize("init_name", EXPECTED_INITIALIZERS)
    def test_baseline_shapes(self, tiny_model, init_name):
        """Each baseline produces correct shapes matching model weight tensors."""
        model_weights = tiny_model.get_weight_tensors()
        init_weights = initialize(tiny_model, init_name)
        for name in model_weights:
            assert name in init_weights, (
                f"'{init_name}' init missing key '{name}'"
            )
            assert init_weights[name].shape == model_weights[name].shape, (
                f"'{init_name}' shape mismatch for '{name}': "
                f"expected {model_weights[name].shape}, got {init_weights[name].shape}"
            )


class TestBaselineFinite:
    """Tests for finiteness."""

    @pytest.mark.parametrize("init_name", EXPECTED_INITIALIZERS)
    def test_baseline_finite(self, tiny_model, init_name):
        """No NaN/Inf in any baseline output."""
        weights = initialize(tiny_model, init_name)
        for name, tensor in weights.items():
            assert torch.isfinite(tensor).all(), (
                f"'{init_name}' produced non-finite values in '{name}'"
            )


class TestBaselineStatistics:
    """Tests for statistical properties."""

    @pytest.mark.parametrize("init_name", EXPECTED_INITIALIZERS)
    def test_baseline_statistics(self, tiny_model, init_name):
        """Mean near 0, std in reasonable range for all baselines."""
        weights = initialize(tiny_model, init_name)
        for name, tensor in weights.items():
            if tensor.numel() < 10:
                continue
            mean = tensor.float().mean().item()
            std = tensor.float().std().item()
            assert abs(mean) < 1.0, (
                f"'{init_name}' mean too far from 0 for '{name}': {mean}"
            )
            assert 0 < std < 5.0, (
                f"'{init_name}' std out of range for '{name}': {std}"
            )


class TestOrthogonalProperty:
    """Tests for orthogonal initialization."""

    def test_orthogonal_property(self, tiny_model):
        """For square matrices, W @ W.T should approximate I."""
        weights = orthogonal_init(tiny_model)
        for name, tensor in weights.items():
            if tensor.dim() == 2:
                # Only check if close to square
                h, w = tensor.shape
                if h == w:
                    # Move to CPU for linalg operations
                    t_cpu = tensor.detach().cpu().float()
                    product = t_cpu @ t_cpu.T
                    identity = torch.eye(h, dtype=torch.float32)
                    # Orthogonal: W @ W.T should be close to I
                    diff = (product - identity).abs().max().item()
                    assert diff < 0.1, (
                        f"Orthogonal W @ W.T should be near I for '{name}', "
                        f"max diff = {diff:.4f}"
                    )


class TestBaselineInterfaceMatchesCA:
    """Tests that baseline interface matches model.get_weight_tensors()."""

    @pytest.mark.parametrize("init_name", EXPECTED_INITIALIZERS)
    def test_baseline_interface_matches_ca(self, tiny_model, init_name):
        """Same key format as model.get_weight_tensors()."""
        model_keys = set(tiny_model.get_weight_tensors().keys())
        init_keys = set(initialize(tiny_model, init_name).keys())
        assert model_keys == init_keys, (
            f"'{init_name}' keys mismatch. "
            f"Missing: {model_keys - init_keys}, Extra: {init_keys - model_keys}"
        )


class TestXavierBounds:
    """Tests for Xavier initialization value bounds."""

    def test_xavier_uniform_bounds(self, tiny_model):
        """Xavier uniform values should be within expected bounds."""
        weights = xavier_uniform_init(tiny_model)
        for name, tensor in weights.items():
            if tensor.dim() == 2:
                fan_in, fan_out = tensor.shape[1], tensor.shape[0]
                # Xavier uniform bound: sqrt(6 / (fan_in + fan_out))
                bound = math.sqrt(6.0 / (fan_in + fan_out))
                max_val = tensor.abs().max().item()
                assert max_val <= bound + 1e-6, (
                    f"Xavier uniform '{name}' max abs value {max_val:.4f} "
                    f"exceeds bound {bound:.4f}"
                )
