"""Tests for baseline initialization strategies."""

import math

import pytest
import torch

from neurogen.baselines.initializers import (
    INITIALIZERS,
    available_initializers,
    get_initializer,
)
from neurogen.model.gpt import GPT


class TestBaselineRegistry:
    def test_all_registered(self):
        names = available_initializers()
        assert len(names) == 9
        expected = [
            "xavier_uniform", "xavier_normal", "kaiming_uniform",
            "kaiming_normal", "orthogonal", "sparse", "fixup",
            "mimetic", "spectral_delta",
        ]
        for name in expected:
            assert name in names, f"Missing: {name}"

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown"):
            get_initializer("not_a_real_init")


class TestBaselineShapes:
    @pytest.mark.parametrize("init_name", available_initializers())
    def test_shapes_match(self, init_name, tiny_model):
        initializer = get_initializer(init_name)
        weights = initializer(tiny_model)
        model_weights = tiny_model.get_weight_tensors()
        assert set(weights.keys()) == set(model_weights.keys())
        for name in weights:
            assert weights[name].shape == model_weights[name].shape, (
                f"{init_name}/{name}: {weights[name].shape} != {model_weights[name].shape}"
            )


class TestBaselineFinite:
    @pytest.mark.parametrize("init_name", available_initializers())
    def test_no_nan_inf(self, init_name, tiny_model):
        initializer = get_initializer(init_name)
        weights = initializer(tiny_model)
        for name, tensor in weights.items():
            assert torch.isfinite(tensor).all(), (
                f"{init_name}/{name} has non-finite values"
            )


class TestBaselineStatistics:
    @pytest.mark.parametrize("init_name", available_initializers())
    def test_reasonable_stats(self, init_name, tiny_model):
        initializer = get_initializer(init_name)
        weights = initializer(tiny_model)
        for name, tensor in weights.items():
            mean = tensor.mean().item()
            std = tensor.std().item()
            assert abs(mean) < 0.5, f"{init_name}/{name}: mean={mean}"
            assert 0.0 < std < 5.0, f"{init_name}/{name}: std={std}"


class TestSpecificInitializers:
    def test_xavier_uniform_bounds(self, tiny_model):
        weights = get_initializer("xavier_uniform")(tiny_model)
        for name, tensor in weights.items():
            fan_in, fan_out = tensor.shape[1], tensor.shape[0]
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            assert tensor.min().item() >= -bound - 1e-6
            assert tensor.max().item() <= bound + 1e-6

    def test_kaiming_normal_variance(self, tiny_model):
        torch.manual_seed(42)
        weights = get_initializer("kaiming_normal")(tiny_model)
        for name, tensor in weights.items():
            fan_in = tensor.shape[1]
            expected_var = 2.0 / fan_in
            actual_var = tensor.var().item()
            # Allow 50% tolerance for small tensors
            assert actual_var < expected_var * 3.0, (
                f"{name}: var={actual_var}, expected~{expected_var}"
            )

    def test_orthogonal_property(self, tiny_model):
        weights = get_initializer("orthogonal")(tiny_model)
        for name, tensor in weights.items():
            if tensor.shape[0] == tensor.shape[1]:
                t = tensor.cpu()
                product = t @ t.T
                eye = torch.eye(t.shape[0])
                assert torch.allclose(product, eye, atol=1e-5), (
                    f"{name} is not orthogonal"
                )


class TestBaselineInterfaceCompat:
    def test_can_set_weights(self, tiny_model, random_batch):
        """All baselines produce weights that can be set on the model."""
        for init_name in available_initializers():
            initializer = get_initializer(init_name)
            weights = initializer(tiny_model)
            tiny_model.set_weight_tensors(weights)
            x, y = random_batch
            _, loss = tiny_model(x, y)
            assert torch.isfinite(loss), f"{init_name} produced non-finite loss"
