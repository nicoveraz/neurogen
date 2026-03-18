"""Tests for the CA Weight Engine and all variants."""

import pytest
import torch

from neurogen.ca.engine import CAWeightEngine, CA_VARIANTS
from neurogen.config import CAConfig, GPTConfig
from neurogen.model.gpt import GPT


@pytest.fixture
def ca_model():
    """Small model for CA tests (CPU-based for reliability)."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=2, n_head=2, n_embd=64, dropout=0.0
    )
    return GPT(config)


ALL_VARIANTS = sorted(CA_VARIANTS.keys())


class TestCAEngineRegistry:
    def test_all_variants_registered(self):
        variants = CAWeightEngine.available_variants()
        assert len(variants) == 5
        expected = ["grid_ca", "neural_ca", "spectral_ca", "topo_ca", "reaction_diffusion"]
        for v in expected:
            assert v in variants

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            CAWeightEngine("not_a_variant")


class TestCADevelopWeights:
    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_shapes(self, variant, ca_model):
        engine = CAWeightEngine(variant, device="cpu")
        weights = engine.develop_weights(ca_model)
        model_weights = ca_model.get_weight_tensors()
        for name in model_weights:
            assert name in weights, f"Missing {name}"
            assert weights[name].shape == model_weights[name].shape, (
                f"{variant}/{name}: {weights[name].shape} != {model_weights[name].shape}"
            )
            assert weights[name].dtype == torch.float32

    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_finite(self, variant, ca_model):
        engine = CAWeightEngine(variant, device="cpu")
        weights = engine.develop_weights(ca_model)
        for name, tensor in weights.items():
            assert torch.isfinite(tensor).all(), f"{variant}/{name} has non-finite values"

    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_magnitude(self, variant, ca_model):
        engine = CAWeightEngine(variant, device="cpu")
        weights = engine.develop_weights(ca_model)
        for name, tensor in weights.items():
            std = tensor.std().item()
            assert std > 1e-6, f"{variant}/{name}: std={std} (collapsed)"
            assert std < 10.0, f"{variant}/{name}: std={std} (exploded)"
            assert tensor.abs().max().item() < 100.0

    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_deterministic(self, variant, ca_model):
        engine = CAWeightEngine(variant, device="cpu")
        torch.manual_seed(42)
        w1 = engine.develop_weights(ca_model)
        torch.manual_seed(42)
        w2 = engine.develop_weights(ca_model)
        for name in w1:
            assert torch.allclose(w1[name], w2[name], atol=1e-6), (
                f"{variant}/{name} not deterministic"
            )

    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_different_seeds(self, variant, ca_model):
        engine = CAWeightEngine(variant, device="cpu")
        w1 = engine.develop_weights(ca_model, seed=42)
        w2 = engine.develop_weights(ca_model, seed=99)
        any_different = any(
            not torch.allclose(w1[n], w2[n], atol=1e-6) for n in w1
        )
        assert any_different, f"{variant} produced same weights for different seeds"


class TestCAGenomeProperties:
    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_genome_size(self, variant, ca_model):
        engine = CAWeightEngine(variant, device="cpu")
        size = engine.genome_size()
        assert size > 0
        model_params = ca_model.count_parameters()
        assert size < model_params, (
            f"{variant} genome ({size}) >= model ({model_params})"
        )

    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_compression_ratio(self, variant, ca_model):
        engine = CAWeightEngine(variant, device="cpu")
        ratio = engine.compression_ratio(ca_model)
        assert ratio > 1, f"{variant} compression ratio {ratio} <= 1"

    @pytest.mark.parametrize("variant", ["grid_ca", "neural_ca"])
    def test_gradient_flow(self, variant, ca_model):
        engine = CAWeightEngine(variant, device="cpu")
        # Develop with gradient tracking
        weights = engine.develop_weights(ca_model)
        loss = sum(w.sum() for w in weights.values())
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in engine.genome.parameters()
        )
        assert has_grad, f"{variant} has no gradient flow to genome"


class TestCATargetShapeFlexibility:
    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_various_shapes(self, variant):
        engine = CAWeightEngine(variant, device="cpu")
        shapes = [(32, 32), (64, 128), (128, 64)]
        for shape in shapes:
            w = engine.genome.develop(
                seed=None, target_shape=shape, n_steps=8
            )
            assert w.shape == shape, (
                f"{variant}: expected {shape}, got {w.shape}"
            )


class TestCAStepCount:
    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_different_steps_different_output(self, variant):
        engine = CAWeightEngine(variant, device="cpu")
        torch.manual_seed(42)
        w1 = engine.genome.develop(seed=None, target_shape=(32, 32), n_steps=8)
        torch.manual_seed(42)
        w2 = engine.genome.develop(seed=None, target_shape=(32, 32), n_steps=32)
        assert not torch.allclose(w1, w2, atol=1e-6), (
            f"{variant}: same output for different step counts"
        )


class TestCAModelIntegration:
    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_can_set_and_forward(self, variant, ca_model):
        """CA-developed weights can be set and used for forward pass."""
        engine = CAWeightEngine(variant, device="cpu")
        weights = engine.develop_weights(ca_model)
        ca_model.set_weight_tensors(weights)
        x = torch.randint(0, 256, (2, 32))
        logits, _ = ca_model(x)
        assert torch.isfinite(logits).all()
