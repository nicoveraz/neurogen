"""Tests for MicroGPT model."""

import pytest
import torch

from neurogen.config import GPTConfig
from neurogen.model.gpt import GPT


class TestGPTInstantiation:
    def test_creates_without_error(self, tiny_config):
        model = GPT(tiny_config)
        assert model is not None

    def test_parameter_count(self, tiny_config):
        model = GPT(tiny_config)
        count = model.count_parameters()
        assert count > 0
        # Rough check: should be in a reasonable range for this config
        assert count < 10_000_000  # tiny config shouldn't be huge

    def test_requires_vocab_size(self):
        config = GPTConfig(vocab_size=0, n_layer=2, n_head=2, n_embd=64)
        with pytest.raises(AssertionError):
            GPT(config)

    def test_n_embd_divisible_by_n_head(self):
        with pytest.raises(ValueError, match="divisible"):
            GPTConfig(vocab_size=256, n_layer=2, n_head=3, n_embd=64)


class TestGPTForward:
    def test_forward_shape(self, tiny_model, random_batch, tiny_config):
        x, y = random_batch
        logits, loss = tiny_model(x, y)
        assert logits.shape == (4, tiny_config.block_size, tiny_config.vocab_size)
        assert loss is not None
        assert loss.dim() == 0  # scalar

    def test_forward_no_targets(self, tiny_model, random_batch):
        x, _ = random_batch
        logits, loss = tiny_model(x)
        assert logits is not None
        assert loss is None

    def test_backward(self, tiny_model, random_batch):
        x, y = random_batch
        _, loss = tiny_model(x, y)
        loss.backward()
        for name, param in tiny_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"


class TestGPTGenerate:
    def test_generate_shape(self, tiny_model, device):
        prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
        output = tiny_model.generate(prompt, max_new_tokens=50)
        assert output.shape == (1, 51)  # 1 prompt + 50 generated

    def test_generate_valid_tokens(self, tiny_model, tiny_config, device):
        prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
        output = tiny_model.generate(prompt, max_new_tokens=50)
        assert (output >= 0).all()
        assert (output < tiny_config.vocab_size).all()


class TestGPTWeightInterface:
    def test_get_weight_tensors(self, tiny_model):
        weights = tiny_model.get_weight_tensors()
        assert isinstance(weights, dict)
        assert len(weights) > 0
        for name, tensor in weights.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.requires_grad

    def test_get_excludes_bias_and_layernorm(self, tiny_model):
        weights = tiny_model.get_weight_tensors()
        for name in weights:
            assert "bias" not in name
            assert "ln_" not in name

    def test_get_includes_attn_and_ffn(self, tiny_model):
        weights = tiny_model.get_weight_tensors()
        names = list(weights.keys())
        # Should have attention weights and FFN weights for each layer
        has_attn = any("c_attn" in n for n in names)
        has_ffn = any("c_fc" in n for n in names)
        assert has_attn, f"No attention weights found in {names}"
        assert has_ffn, f"No FFN weights found in {names}"

    def test_set_weight_tensors(self, tiny_model, device):
        weights = tiny_model.get_weight_tensors()
        # Modify one tensor
        key = list(weights.keys())[0]
        weights[key] = torch.zeros_like(weights[key])
        tiny_model.set_weight_tensors(weights)
        # Verify it took effect
        new_weights = tiny_model.get_weight_tensors()
        assert (new_weights[key] == 0).all()

    def test_set_preserves_forward(self, tiny_model, random_batch):
        weights = tiny_model.get_weight_tensors()
        tiny_model.set_weight_tensors(weights)
        x, y = random_batch
        logits, loss = tiny_model(x, y)
        assert not torch.isnan(loss)

    def test_roundtrip(self, tiny_model, random_batch, device):
        x, y = random_batch
        _, loss_before = tiny_model(x, y)
        weights = tiny_model.get_weight_tensors()
        tiny_model.set_weight_tensors(weights)
        _, loss_after = tiny_model(x, y)
        assert torch.allclose(loss_before, loss_after)

    def test_weight_tying(self, tiny_model):
        """Token embedding and LM head should share the same weight."""
        wte = tiny_model.transformer.wte.weight
        lm_head = tiny_model.lm_head.weight
        assert wte.data_ptr() == lm_head.data_ptr()

    def test_no_duplicate_tied_weight(self, tiny_model):
        """Tied weight (wte/lm_head) should not appear in weight tensors.

        Since wte and lm_head share the same parameter, and both are
        excluded from get_weight_tensors (wte explicitly, lm_head via
        weight tying deduplication), neither should appear.
        """
        weights = tiny_model.get_weight_tensors()
        names = list(weights.keys())
        assert not any("wte" in n for n in names), "wte should be excluded (tied)"
        # Due to weight tying, lm_head.weight is the same object as wte.weight
        # and named_parameters() deduplicates, so it won't appear separately
        # This is correct — the embedding is not a target for CA initialization
        data_ptrs = [p.data_ptr() for p in weights.values()]
        assert len(data_ptrs) == len(set(data_ptrs)), "No duplicate tensors"


class TestGPTConfigs:
    @pytest.mark.parametrize(
        "n_layer,n_head,n_embd",
        [(1, 1, 32), (2, 2, 64), (4, 4, 128), (3, 3, 96)],
    )
    def test_various_configs(self, n_layer, n_head, n_embd):
        config = GPTConfig(
            block_size=32,
            vocab_size=256,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=0.0,
        )
        model = GPT(config)
        x = torch.randint(0, 256, (2, 32))
        logits, _ = model(x)
        assert logits.shape == (2, 32, 256)


class TestGPTDeterminism:
    def test_deterministic_forward(self, tiny_config, device):
        torch.manual_seed(42)
        model = GPT(tiny_config).to(device)
        x = torch.randint(0, 256, (2, 32), device=device)

        model.eval()
        logits1, _ = model(x)
        logits2, _ = model(x)
        assert torch.allclose(logits1, logits2)
