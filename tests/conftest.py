"""Shared test fixtures for NeuroGen."""

import pytest
import torch

from neurogen.config import GPTConfig, TrainConfig, get_device
from neurogen.model.gpt import GPT


@pytest.fixture
def device():
    """Auto-detected device for tests."""
    return get_device()


@pytest.fixture
def tiny_config():
    """Ultra-small config for fast tests."""
    return GPTConfig(
        block_size=32,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
    )


@pytest.fixture
def tiny_train_config(device):
    """Minimal training config for tests."""
    return TrainConfig(
        max_steps=50,
        eval_interval=25,
        eval_steps=5,
        batch_size=4,
        lr=1e-3,
        grad_clip=1.0,
        device=device,
        log_interval=10,
    )


@pytest.fixture
def tiny_model(tiny_config, device):
    """Pre-instantiated tiny model on best available device."""
    return GPT(tiny_config).to(device)


@pytest.fixture
def random_batch(tiny_config, device):
    """Random batch matching tiny config on best available device."""
    x = torch.randint(0, tiny_config.vocab_size, (4, tiny_config.block_size)).to(
        device
    )
    y = torch.randint(0, tiny_config.vocab_size, (4, tiny_config.block_size)).to(
        device
    )
    return x, y


def mps_available():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


requires_mps = pytest.mark.skipif(not mps_available(), reason="MPS not available")
