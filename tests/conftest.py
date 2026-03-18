import pytest
import torch
from neurogen.config import GPTConfig, TrainConfig, get_device
from neurogen.model.gpt import GPT


@pytest.fixture
def device():
    return get_device()


@pytest.fixture
def tiny_config():
    return GPTConfig(
        block_size=32,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
    )


@pytest.fixture
def tiny_train_config():
    return TrainConfig(
        max_steps=50,
        eval_interval=25,
        batch_size=4,
        lr=1e-3,
        grad_clip=1.0,
    )


@pytest.fixture
def tiny_model(tiny_config, device):
    return GPT(tiny_config).to(device)


@pytest.fixture
def random_batch(tiny_config, device):
    x = torch.randint(0, tiny_config.vocab_size, (4, tiny_config.block_size)).to(device)
    y = torch.randint(0, tiny_config.vocab_size, (4, tiny_config.block_size)).to(device)
    return x, y


def mps_available():
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


requires_mps = pytest.mark.skipif(not mps_available(), reason="MPS not available")
