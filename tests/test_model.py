"""Tests for neurogen/model/gpt.py."""

import pytest
import torch
from neurogen.config import GPTConfig
from neurogen.model.gpt import GPT


class TestGPTInstantiation:
    """Tests for GPT model creation."""

    def test_gpt_instantiation(self, tiny_config, device):
        """Model creates without error using tiny_config."""
        model = GPT(tiny_config).to(device)
        assert model is not None, "Model should be instantiated"
        assert model.config is tiny_config, "Model should store its config"

    def test_gpt_parameter_count(self, tiny_model):
        """Model has a positive number of trainable parameters."""
        n_params = tiny_model.count_parameters()
        assert n_params > 0, "Model should have trainable parameters"


class TestGPTForward:
    """Tests for the forward pass."""

    def test_gpt_forward_shape_no_targets(self, tiny_model, random_batch, tiny_config):
        """Logits shape == (batch, seq_len, vocab_size), loss is None without targets."""
        x, _ = random_batch
        logits, loss = tiny_model(x)
        assert logits.shape == (4, tiny_config.block_size, tiny_config.vocab_size), (
            f"Expected logits shape (4, {tiny_config.block_size}, {tiny_config.vocab_size}), "
            f"got {logits.shape}"
        )
        assert loss is None, "Loss should be None when no targets are provided"

    def test_gpt_forward_shape_with_targets(self, tiny_model, random_batch, tiny_config):
        """Loss is a scalar when targets are provided."""
        x, y = random_batch
        logits, loss = tiny_model(x, y)
        assert logits.shape == (4, tiny_config.block_size, tiny_config.vocab_size), (
            f"Expected logits shape (4, {tiny_config.block_size}, {tiny_config.vocab_size}), "
            f"got {logits.shape}"
        )
        assert loss is not None, "Loss should not be None when targets are provided"
        assert loss.dim() == 0, "Loss should be a scalar"
        assert torch.isfinite(loss), "Loss should be finite"


class TestGPTBackward:
    """Tests for backward pass."""

    def test_gpt_backward(self, tiny_model, random_batch):
        """Forward + backward: all params have non-None finite gradients."""
        x, y = random_batch
        _, loss = tiny_model(x, y)
        loss.backward()
        for name, param in tiny_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"Gradient for '{name}' should not be None after backward"
                )
                assert torch.isfinite(param.grad).all(), (
                    f"Gradient for '{name}' should be finite"
                )


class TestGPTGenerate:
    """Tests for token generation."""

    def test_gpt_generate(self, tiny_model, tiny_config, device):
        """Generate 20 tokens: correct shape, all tokens in valid range."""
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 5)).to(device)
        max_new_tokens = 20
        output = tiny_model.generate(prompt, max_new_tokens=max_new_tokens)
        assert output.shape == (1, 5 + max_new_tokens), (
            f"Expected shape (1, {5 + max_new_tokens}), got {output.shape}"
        )
        assert (output >= 0).all() and (output < tiny_config.vocab_size).all(), (
            "All generated tokens should be in [0, vocab_size)"
        )


class TestGPTWeightInterface:
    """Tests for get_weight_tensors / set_weight_tensors."""

    def test_gpt_weight_interface_get(self, tiny_model):
        """Keys include attn and ffn weights, no bias/LN, all require_grad."""
        weights = tiny_model.get_weight_tensors()
        assert len(weights) > 0, "Should return at least one weight tensor"
        for name, tensor in weights.items():
            assert "bias" not in name, f"'{name}' should not contain bias"
            assert "ln_" not in name, f"'{name}' should not contain LayerNorm"
            assert "pos_emb" not in name, f"'{name}' should not contain pos_emb"
            assert "lm_head" not in name, f"'{name}' should not contain lm_head (tied)"
        # Check some expected weight patterns exist
        names_str = " ".join(weights.keys())
        assert "attn" in names_str or "c_attn" in names_str, (
            "Should include attention weights"
        )
        assert "c_fc" in names_str or "ffn" in names_str, (
            "Should include FFN weights"
        )

    def test_gpt_weight_interface_set(self, tiny_model):
        """Fill one tensor with zeros, verify it took effect."""
        weights = tiny_model.get_weight_tensors()
        first_key = list(weights.keys())[0]
        zero_weights = {first_key: torch.zeros_like(weights[first_key])}
        tiny_model.set_weight_tensors(zero_weights)

        updated_weights = tiny_model.get_weight_tensors()
        assert torch.all(updated_weights[first_key] == 0), (
            f"Weight '{first_key}' should be all zeros after set"
        )

    def test_gpt_weight_interface_roundtrip(self, tiny_model, random_batch):
        """Get then set unchanged weights -> identical output."""
        x, _ = random_batch
        with torch.no_grad():
            logits_before, _ = tiny_model(x)

        weights = tiny_model.get_weight_tensors()
        # Make copies of the weight data
        weight_copies = {k: v.clone() for k, v in weights.items()}
        tiny_model.set_weight_tensors(weight_copies)

        with torch.no_grad():
            logits_after, _ = tiny_model(x)

        assert torch.allclose(logits_before, logits_after, atol=1e-6), (
            "Output should be identical after setting unchanged weights"
        )


class TestGPTWeightTying:
    """Tests for embedding weight tying."""

    def test_gpt_weight_tying(self, tiny_model):
        """tok_emb and lm_head share the same data_ptr."""
        assert tiny_model.tok_emb.weight.data_ptr() == tiny_model.lm_head.weight.data_ptr(), (
            "tok_emb.weight and lm_head.weight should share the same data"
        )


class TestGPTConfigs:
    """Tests for various model configurations."""

    def test_gpt_valid_configs(self, device):
        """Several valid configs should work."""
        configs = [
            GPTConfig(block_size=16, vocab_size=50, n_layer=1, n_head=1, n_embd=32, dropout=0.0),
            GPTConfig(block_size=64, vocab_size=100, n_layer=4, n_head=4, n_embd=128, dropout=0.1),
            GPTConfig(block_size=32, vocab_size=256, n_layer=2, n_head=2, n_embd=64, dropout=0.0),
        ]
        for cfg in configs:
            model = GPT(cfg).to(device)
            x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size)).to(device)
            logits, _ = model(x)
            assert logits.shape == (2, cfg.block_size, cfg.vocab_size), (
                f"Config {cfg} should produce correct output shape"
            )

    def test_gpt_non_divisible_n_embd_n_head(self):
        """Non-divisible n_embd/n_head should raise an error."""
        cfg = GPTConfig(
            block_size=32, vocab_size=100, n_layer=2,
            n_head=3, n_embd=64, dropout=0.0,
        )
        with pytest.raises(AssertionError, match="divisible"):
            GPT(cfg)


class TestGPTDeterminism:
    """Tests for deterministic behavior."""

    def test_gpt_determinism(self, tiny_config, device):
        """Same seed -> same output."""
        x = torch.randint(0, tiny_config.vocab_size, (2, tiny_config.block_size)).to(device)

        torch.manual_seed(42)
        model1 = GPT(tiny_config).to(device)
        model1.eval()
        with torch.no_grad():
            out1, _ = model1(x)

        torch.manual_seed(42)
        model2 = GPT(tiny_config).to(device)
        model2.eval()
        with torch.no_grad():
            out2, _ = model2(x)

        assert torch.allclose(out1, out2, atol=1e-6), (
            "Same seed should produce identical outputs"
        )
