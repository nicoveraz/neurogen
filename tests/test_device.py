"""Tests for device handling across the codebase."""

import pytest
import torch

from neurogen.analysis.weight_analysis import spectral_norm
from neurogen.config import HARDWARE_PROFILES, GPTConfig, get_device
from neurogen.model.gpt import GPT


class TestDeviceAutodetect:
    """Tests for get_device()."""

    def test_device_autodetect(self):
        """get_device returns one of ['cuda', 'mps', 'cpu']."""
        device = get_device()
        assert device in ("cuda", "mps", "cpu"), (
            f"get_device() should return 'cuda', 'mps', or 'cpu', got '{device}'"
        )

    def test_device_consistency(self):
        """Calling get_device() twice returns the same value."""
        d1 = get_device()
        d2 = get_device()
        assert d1 == d2, "get_device() should return consistent results"


class TestModelOnDevice:
    """Tests for model device placement."""

    def test_model_on_device(self, tiny_config, device):
        """Model should work on the detected device."""
        model = GPT(tiny_config).to(device)
        x = torch.randint(0, tiny_config.vocab_size, (2, tiny_config.block_size)).to(device)
        logits, _ = model(x)
        # Check that output is on the expected device
        assert str(logits.device).startswith(device), (
            f"Output should be on {device}, got {logits.device}"
        )
        assert torch.isfinite(logits).all(), "Output should be finite on any device"

    def test_model_forward_backward_on_device(self, tiny_model, random_batch):
        """Forward + backward should work on the detected device."""
        x, y = random_batch
        logits, loss = tiny_model(x, y)
        loss.backward()
        assert torch.isfinite(loss), "Loss should be finite"
        for name, param in tiny_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Gradient for '{name}' should be finite on this device"
                )


class TestAnalysisCPUFallback:
    """Tests for CPU fallback in analysis functions."""

    def test_analysis_cpu_fallback(self, device):
        """spectral_norm works regardless of device."""
        mat = torch.randn(20, 20, dtype=torch.float32, device=device)
        sn = spectral_norm(mat)
        assert isinstance(sn, float), "spectral_norm should return float"
        assert sn > 0, "spectral_norm should be positive for random matrix"
        # The function internally moves to CPU, so this tests the fallback

    def test_analysis_cpu_fallback_1d(self, device):
        """spectral_norm handles 1D tensors on any device."""
        vec = torch.randn(20, dtype=torch.float32, device=device)
        sn = spectral_norm(vec)
        assert sn == 0.0, "spectral_norm of 1D tensor should be 0.0"


class TestHardwareProfiles:
    """Tests for hardware profile definitions."""

    def test_hardware_profiles_exist(self):
        """Profiles should exist and have required keys."""
        assert len(HARDWARE_PROFILES) >= 3, (
            f"Should have at least 3 hardware profiles, got {len(HARDWARE_PROFILES)}"
        )
        required_keys = [
            "device", "max_batch_size", "max_n_embd",
            "max_n_layer", "max_block_size", "compile", "dtype",
        ]
        for profile_name, profile in HARDWARE_PROFILES.items():
            for key in required_keys:
                assert key in profile, (
                    f"Profile '{profile_name}' missing key '{key}'"
                )

    def test_hardware_profiles_valid_devices(self):
        """Profile devices should be valid device strings."""
        valid_devices = {"cpu", "cuda", "mps"}
        for profile_name, profile in HARDWARE_PROFILES.items():
            assert profile["device"] in valid_devices, (
                f"Profile '{profile_name}' has invalid device '{profile['device']}'"
            )

    def test_hardware_profiles_positive_values(self):
        """Numeric profile values should be positive."""
        for profile_name, profile in HARDWARE_PROFILES.items():
            assert profile["max_batch_size"] > 0, (
                f"Profile '{profile_name}' max_batch_size should be > 0"
            )
            assert profile["max_n_embd"] > 0, (
                f"Profile '{profile_name}' max_n_embd should be > 0"
            )
