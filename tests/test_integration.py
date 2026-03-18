"""End-to-end integration tests."""

import pytest
import torch

from neurogen.ca.engine import CAWeightEngine
from neurogen.config import GPTConfig, TrainConfig, get_device
from neurogen.data.shakespeare import ShakespeareDataset
from neurogen.model.gpt import GPT
from neurogen.training.trainer import train


@pytest.fixture(scope="module")
def dataset():
    return ShakespeareDataset()


@pytest.fixture
def device():
    return get_device()


@pytest.mark.integration
class TestE2EBaseline:
    def test_full_pipeline(self, dataset, device):
        """data -> model -> xavier init -> train 200 steps -> eval -> metrics"""
        config = GPTConfig(
            block_size=32, vocab_size=dataset.vocab_size,
            n_layer=2, n_head=2, n_embd=64, dropout=0.0,
        )
        torch.manual_seed(42)
        model = GPT(config)

        from neurogen.baselines.initializers import get_initializer
        weights = get_initializer("xavier_normal")(model)
        model.set_weight_tensors(weights)

        train_config = TrainConfig(
            max_steps=200, eval_interval=100, eval_steps=5,
            batch_size=8, lr=1e-3, device=device, log_interval=50,
        )
        metrics = train(model, dataset, train_config)
        assert metrics["final_val_loss"] < 5.0


@pytest.mark.integration
class TestE2ECA:
    def test_ca_training(self, dataset, device):
        """data -> model -> grid_ca init -> train 200 steps -> eval"""
        config = GPTConfig(
            block_size=32, vocab_size=dataset.vocab_size,
            n_layer=2, n_head=2, n_embd=64, dropout=0.0,
        )
        torch.manual_seed(42)
        model = GPT(config)

        engine = CAWeightEngine("grid_ca", device="cpu")
        weights = engine.develop_weights(model)
        model.set_weight_tensors(weights)

        train_config = TrainConfig(
            max_steps=200, eval_interval=100, eval_steps=5,
            batch_size=8, lr=1e-3, device=device, log_interval=50,
        )
        metrics = train(model, dataset, train_config)
        assert metrics["final_val_loss"] is not None
        assert torch.isfinite(torch.tensor(metrics["final_val_loss"]))

    def test_ca_generates_text(self, dataset, device):
        """CA-initialized model can generate valid text."""
        config = GPTConfig(
            block_size=32, vocab_size=dataset.vocab_size,
            n_layer=2, n_head=2, n_embd=64, dropout=0.0,
        )
        model = GPT(config).to(device)
        engine = CAWeightEngine("grid_ca", device="cpu")
        weights = engine.develop_weights(model)
        model.set_weight_tensors(weights)

        prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
        output = model.generate(prompt, max_new_tokens=50)
        text = dataset.decode(output[0])
        assert len(text) > 0


@pytest.mark.integration
class TestE2EComparison:
    def test_ca_vs_baseline(self, dataset, device):
        """Both CA and baseline produce valid, comparable metrics."""
        config = GPTConfig(
            block_size=32, vocab_size=dataset.vocab_size,
            n_layer=2, n_head=2, n_embd=64, dropout=0.0,
        )
        train_config = TrainConfig(
            max_steps=100, eval_interval=50, eval_steps=5,
            batch_size=8, lr=1e-3, device=device, log_interval=50,
        )

        # Baseline
        torch.manual_seed(42)
        model_base = GPT(config)
        from neurogen.baselines.initializers import get_initializer
        model_base.set_weight_tensors(get_initializer("xavier_normal")(model_base))
        metrics_base = train(model_base, dataset, train_config)

        # CA
        torch.manual_seed(42)
        model_ca = GPT(config)
        engine = CAWeightEngine("grid_ca", device="cpu")
        model_ca.set_weight_tensors(engine.develop_weights(model_ca))
        metrics_ca = train(model_ca, dataset, train_config)

        # Both should produce valid metrics
        assert metrics_base["final_val_loss"] is not None
        assert metrics_ca["final_val_loss"] is not None
        # Loss difference should be computable
        diff = abs(metrics_base["final_val_loss"] - metrics_ca["final_val_loss"])
        assert diff >= 0  # trivially true, but ensures both are numbers
