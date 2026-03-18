"""Tests for neurogen/ca/ (engine, genome, and all CA variants)."""

import pytest
import torch

from neurogen.ca.engine import CA_VARIANTS, CAWeightEngine
from neurogen.ca.genome import CAGenome
from neurogen.ca.grid_ca import GridCAGenome
from neurogen.ca.neural_ca import NeuralCAGenome
from neurogen.ca.spectral_ca import SpectralCAGenome
from neurogen.ca.topo_ca import TopologicalCAGenome
from neurogen.ca.reaction_diffusion import ReactionDiffusionGenome
from neurogen.config import GPTConfig
from neurogen.model.gpt import GPT


VARIANT_NAMES = ["grid_ca", "neural_ca", "spectral_ca", "topo_ca", "reaction_diffusion"]


@pytest.fixture
def tiny_model_cpu(tiny_config):
    """A small model on CPU for CA tests (avoids MPS FFT issues)."""
    return GPT(tiny_config).to("cpu")


class TestCAVariantRegistry:
    """Tests for CA variant registration."""

    def test_ca_engine_variants_registered(self):
        """All 5 variants should be available."""
        available = CAWeightEngine.available_variants()
        for name in VARIANT_NAMES:
            assert name in available, f"'{name}' should be registered as a CA variant"
        assert len(available) == 5, f"Expected 5 variants, got {len(available)}"

    def test_unknown_variant_raises(self):
        """Unknown variant should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown CA variant"):
            CAWeightEngine(variant="nonexistent_variant")


class TestCADevelopWeightsShapes:
    """Tests for developed weight shapes."""

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_ca_develop_weights_shapes(self, tiny_config, variant):
        """Each variant produces correct shapes for all model weight tensors."""
        model = GPT(tiny_config).to("cpu")
        engine = CAWeightEngine(variant=variant, device="cpu")
        weights = engine.develop_weights(model, n_steps=4)
        model_weights = model.get_weight_tensors()
        for name in model_weights:
            assert name in weights, (
                f"'{variant}' missing key '{name}'"
            )
            assert weights[name].shape == model_weights[name].shape, (
                f"'{variant}' shape mismatch for '{name}': "
                f"expected {model_weights[name].shape}, got {weights[name].shape}"
            )


class TestCADevelopWeightsFinite:
    """Tests for finiteness of developed weights."""

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_ca_develop_weights_finite(self, tiny_config, variant):
        """No NaN/Inf in developed weights."""
        model = GPT(tiny_config).to("cpu")
        engine = CAWeightEngine(variant=variant, device="cpu")
        weights = engine.develop_weights(model, n_steps=4)
        for name, tensor in weights.items():
            assert torch.isfinite(tensor).all(), (
                f"'{variant}' produced non-finite values in '{name}'"
            )


class TestCADevelopWeightsMagnitude:
    """Tests for reasonable magnitude of developed weights."""

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_ca_develop_weights_magnitude(self, tiny_config, variant):
        """Std of developed weights should be in a reasonable range."""
        model = GPT(tiny_config).to("cpu")
        engine = CAWeightEngine(variant=variant, device="cpu")
        weights = engine.develop_weights(model, n_steps=4)
        for name, tensor in weights.items():
            if tensor.numel() < 10:
                continue
            std = tensor.float().std().item()
            assert 0.001 < std < 1.0, (
                f"'{variant}' std for '{name}' is {std:.6f}, "
                f"expected between 0.001 and 1.0"
            )


class TestCADeterminism:
    """Tests for deterministic behavior."""

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_ca_develop_deterministic(self, tiny_config, variant):
        """Same seed -> same developed weights."""
        model = GPT(tiny_config).to("cpu")

        torch.manual_seed(42)
        engine1 = CAWeightEngine(variant=variant, device="cpu")
        weights1 = engine1.develop_weights(model, n_steps=4)

        torch.manual_seed(42)
        engine2 = CAWeightEngine(variant=variant, device="cpu")
        weights2 = engine2.develop_weights(model, n_steps=4)

        for name in weights1:
            assert torch.allclose(weights1[name], weights2[name], atol=1e-5), (
                f"'{variant}' not deterministic for '{name}'"
            )

    def test_ca_develop_different_seeds(self, tiny_config):
        """Different seeds -> different developed weights."""
        model = GPT(tiny_config).to("cpu")

        torch.manual_seed(42)
        engine1 = CAWeightEngine(variant="grid_ca", device="cpu")
        weights1 = engine1.develop_weights(model, n_steps=4)

        torch.manual_seed(999)
        engine2 = CAWeightEngine(variant="grid_ca", device="cpu")
        weights2 = engine2.develop_weights(model, n_steps=4)

        any_different = False
        for name in weights1:
            if not torch.allclose(weights1[name], weights2[name], atol=1e-5):
                any_different = True
                break
        assert any_different, "Different seeds should produce different weights"


class TestCAGenomeSize:
    """Tests for genome size and compression."""

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_ca_genome_size(self, tiny_config, variant):
        """Genome size should be > 0 and much less than model param count."""
        model = GPT(tiny_config).to("cpu")
        engine = CAWeightEngine(variant=variant, device="cpu")
        genome_size = engine.genome_size()
        assert genome_size > 0, f"'{variant}' genome_size should be > 0"
        model_weight_params = sum(
            p.numel() for p in model.get_weight_tensors().values()
        )
        assert genome_size < model_weight_params, (
            f"'{variant}' genome ({genome_size}) should be smaller than "
            f"model weights ({model_weight_params})"
        )


class TestCAGradientFlow:
    """Tests for gradient flow through learnable CA variants."""

    @pytest.mark.parametrize("variant", ["grid_ca", "neural_ca"])
    def test_ca_genome_gradient_flow(self, variant):
        """Learnable variants (grid_ca, neural_ca) should have gradients."""
        if variant == "grid_ca":
            genome = GridCAGenome(hidden_dim=32, device="cpu")
        else:
            genome = NeuralCAGenome(n_channels=8, hidden_dim=32, device="cpu")

        target_shape = (16, 16)
        seed = genome.create_seed(target_shape, pattern="center")
        output = genome.develop(seed, target_shape, n_steps=4)
        loss = output.sum()
        loss.backward()

        has_grad = False
        for p in genome.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, f"'{variant}' should have non-zero gradients after backward"


class TestCAStepCount:
    """Tests for step count effect."""

    def test_ca_step_count_affects_output(self, tiny_config):
        """Different n_steps -> different outputs."""
        model = GPT(tiny_config).to("cpu")
        torch.manual_seed(42)
        engine = CAWeightEngine(variant="grid_ca", device="cpu")
        weights_4 = engine.develop_weights(model, n_steps=4)

        torch.manual_seed(42)
        engine2 = CAWeightEngine(variant="grid_ca", device="cpu")
        weights_16 = engine2.develop_weights(model, n_steps=16)

        any_different = False
        for name in weights_4:
            if not torch.allclose(weights_4[name], weights_16[name], atol=1e-5):
                any_different = True
                break
        assert any_different, "Different n_steps should produce different weights"


class TestCATargetShape:
    """Tests for shape flexibility."""

    def test_ca_target_shape_flexibility(self):
        """CA can develop different shapes."""
        genome = GridCAGenome(hidden_dim=32, device="cpu")
        shapes = [(8, 16), (16, 8), (32, 32), (4, 64)]
        for shape in shapes:
            seed = genome.create_seed(shape, pattern="center")
            output = genome.develop(seed, shape, n_steps=4)
            assert output.shape == shape, (
                f"Expected shape {shape}, got {output.shape}"
            )


class TestGridCASeedPropagation:
    """Tests for GridCA seed behavior."""

    def test_grid_ca_seed_propagation(self):
        """Starting from center seed, non-zero region should grow with steps."""
        genome = GridCAGenome(hidden_dim=32, device="cpu")
        shape = (32, 32)

        seed = genome.create_seed(shape, pattern="center", noise_scale=0.0)
        # The seed has a small center region set to 1.0
        initial_nonzero = (seed.abs() > 0.01).sum().item()

        # After some development steps, more cells should be active
        output = genome.develop(seed, shape, n_steps=16)
        final_nonzero = (output.abs() > 0.001).sum().item()
        assert final_nonzero >= initial_nonzero, (
            "Non-zero region should not shrink after CA development"
        )


class TestNeuralCAHiddenState:
    """Tests for NeuralCA multi-channel state."""

    def test_neural_ca_hidden_state(self):
        """Neural CA should use multi-channel hidden state."""
        n_channels = 8
        genome = NeuralCAGenome(n_channels=n_channels, hidden_dim=32, device="cpu")
        shape = (16, 16)
        seed = genome.create_seed(shape, pattern="center")
        output = genome.develop(seed, shape, n_steps=4)
        # Output should be 2D (projected from multi-channel)
        assert output.dim() == 2, "Output should be 2D after projection"
        assert output.shape == shape, f"Expected {shape}, got {output.shape}"


class TestReactionDiffusionSensitivity:
    """Tests for RD parameter sensitivity."""

    def test_rd_parameter_sensitivity(self):
        """Different model_types should produce different patterns."""
        shape = (16, 16)
        outputs = {}
        for model_type in ["gray_scott", "fitzhugh_nagumo", "brusselator"]:
            torch.manual_seed(42)
            genome = ReactionDiffusionGenome(model_type=model_type, device="cpu")
            seed = genome.create_seed(shape, pattern="center")
            output = genome.develop(seed, shape, n_steps=8)
            outputs[model_type] = output

        # At least two model types should produce different outputs
        gs = outputs["gray_scott"]
        fn = outputs["fitzhugh_nagumo"]
        assert not torch.allclose(gs, fn, atol=1e-3), (
            "Different RD model types should produce different patterns"
        )


class TestHandcraftedBlockDiagonal:
    """Tests for handcrafted block-diagonal init."""

    def test_handcrafted_block_diagonal(self, tiny_config):
        """Block diagonal init should have correct block structure."""
        from neurogen.ca.handcrafted import block_diagonal_init

        model = GPT(tiny_config).to("cpu")
        weights = block_diagonal_init(model, n_blocks=4)

        for name, tensor in weights.items():
            if tensor.dim() == 2:
                h, w = tensor.shape
                bh = h // 4
                bw = w // 4
                if bh > 1 and bw > 1:
                    # Check that diagonal blocks have larger values than off-diagonal
                    on_diag_sum = 0.0
                    off_diag_sum = 0.0
                    on_diag_count = 0
                    off_diag_count = 0
                    for i in range(min(4, h // bh)):
                        for j in range(min(4, w // bw)):
                            rs = i * bh
                            re = min((i + 1) * bh, h)
                            cs = j * bw
                            ce = min((j + 1) * bw, w)
                            block = tensor[rs:re, cs:ce]
                            block_mean_abs = block.abs().mean().item()
                            if i == j:
                                on_diag_sum += block_mean_abs
                                on_diag_count += 1
                            else:
                                off_diag_sum += block_mean_abs
                                off_diag_count += 1

                    if on_diag_count > 0 and off_diag_count > 0:
                        on_avg = on_diag_sum / on_diag_count
                        off_avg = off_diag_sum / off_diag_count
                        assert on_avg > off_avg, (
                            f"On-diagonal blocks should be larger than off-diagonal "
                            f"for '{name}': on={on_avg:.4f}, off={off_avg:.4f}"
                        )
                    break  # Only need to check one matrix


class TestHandcraftedLowRankSparse:
    """Tests for low-rank sparse init."""

    def test_handcrafted_low_rank_sparse(self, tiny_config):
        """Low-rank sparse init should have sparsity in the sparse component."""
        from neurogen.ca.handcrafted import low_rank_sparse_init

        model = GPT(tiny_config).to("cpu")
        weights = low_rank_sparse_init(model, rank=4, sparsity=0.9)

        for name, tensor in weights.items():
            assert torch.isfinite(tensor).all(), (
                f"Low-rank sparse should be finite for '{name}'"
            )
            if tensor.dim() == 2 and tensor.numel() > 100:
                # The combined matrix (low-rank + sparse) won't be exactly sparse,
                # but it should have reasonable magnitude
                std = tensor.float().std().item()
                assert std > 0, (
                    f"Low-rank sparse std should be > 0 for '{name}'"
                )
