"""Tests for device handling and MPS compatibility."""

import pytest
import torch

from neurogen.config import GPTConfig, HARDWARE_PROFILES, get_device
from neurogen.model.gpt import GPT
from tests.conftest import mps_available, requires_mps


class TestDeviceDetection:
    def test_returns_valid_device(self):
        device = get_device()
        assert device in ["cuda", "mps", "cpu"]

    def test_can_create_tensor(self):
        device = get_device()
        t = torch.zeros(2, 2, device=device)
        assert str(t.device).startswith(device)


class TestModelOnDevice:
    def test_model_forward_on_device(self, tiny_config, device):
        model = GPT(tiny_config).to(device)
        x = torch.randint(0, tiny_config.vocab_size, (2, tiny_config.block_size)).to(
            device
        )
        logits, _ = model(x)
        assert str(logits.device).startswith(device)

    def test_model_backward_on_device(self, tiny_config, device):
        model = GPT(tiny_config).to(device)
        x = torch.randint(0, tiny_config.vocab_size, (2, tiny_config.block_size)).to(
            device
        )
        y = torch.randint(0, tiny_config.vocab_size, (2, tiny_config.block_size)).to(
            device
        )
        _, loss = model(x, y)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"


@requires_mps
class TestMPS:
    def test_forward(self, tiny_config):
        model = GPT(tiny_config).to("mps")
        x = torch.randint(0, tiny_config.vocab_size, (2, tiny_config.block_size)).to(
            "mps"
        )
        logits, _ = model(x)
        assert logits.device.type == "mps"
        assert torch.isfinite(logits).all()

    def test_generate(self, tiny_config):
        model = GPT(tiny_config).to("mps")
        prompt = torch.zeros((1, 1), dtype=torch.long, device="mps")
        output = model.generate(prompt, max_new_tokens=20)
        assert (output >= 0).all()
        assert (output < tiny_config.vocab_size).all()


class TestHardwareProfiles:
    def test_profiles_exist(self):
        assert "macbook_m1pro_16gb" in HARDWARE_PROFILES
        assert "cpu_only" in HARDWARE_PROFILES

    def test_profile_has_required_keys(self):
        for name, profile in HARDWARE_PROFILES.items():
            assert "description" in profile, f"{name} missing description"
            assert "max_config" in profile, f"{name} missing max_config"
            assert "safe_batch_size" in profile, f"{name} missing safe_batch_size"
