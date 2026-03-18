"""End-to-end integration tests for the NeuroGen pipeline."""

import pytest
import torch

from neurogen.baselines.initializers import initialize, xavier_normal_init
from neurogen.ca.engine import CAWeightEngine
from neurogen.ca.live import AlphaSchedule, LocalNormCA
from neurogen.config import GPTConfig, LiveCAConfig, TrainConfig
from neurogen.model.gpt import GPT
from neurogen.training.live_ca_trainer import LiveCATrainer
from neurogen.training.trainer import train


class MockDataset:
    """Minimal mock dataset for integration tests."""

    def __init__(self, vocab_size: int = 256, data_len: int = 10000):
        self.vocab_size = vocab_size
        self._data = torch.randint(0, vocab_size, (data_len,))

    @property
    def train_data(self) -> torch.Tensor:
        n = int(0.9 * len(self._data))
        return self._data[:n]

    @property
    def val_data(self) -> torch.Tensor:
        n = int(0.9 * len(self._data))
        return self._data[n:]

    def get_batch(
        self, split: str, batch_size: int, block_size: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)


@pytest.fixture
def mock_dataset():
    return MockDataset(vocab_size=256)


@pytest.fixture
def e2e_train_config():
    return TrainConfig(
        max_steps=50,
        eval_interval=25,
        batch_size=4,
        lr=1e-3,
        grad_clip=1.0,
        warmup_steps=5,
        min_lr=1e-5,
    )


class TestE2EBaselineTraining:
    """End-to-end test: baseline init -> training -> verify."""

    def test_e2e_baseline_training(self, tiny_config, mock_dataset, e2e_train_config, device):
        """Full pipeline with xavier init: model initializes, trains, loss decreases."""
        model = GPT(tiny_config).to(device)

        def init_fn(m):
            return xavier_normal_init(m)

        metrics = train(
            model=model,
            dataset=mock_dataset,
            config=e2e_train_config,
            init_fn=init_fn,
            device=device,
        )

        # Verify training ran to completion
        assert len(metrics["train_losses"]) == 50, "Should have 50 train losses"
        assert len(metrics["val_losses"]) > 0, "Should have val losses"
        assert metrics["best_val_loss"] < float("inf"), "Should have finite best val loss"

        # Verify loss decreased
        losses = metrics["train_losses"]
        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, (
            f"Loss should decrease: early {early_avg:.4f} > late {late_avg:.4f}"
        )

        # Verify model produces valid outputs after training
        x = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.block_size)).to(device)
        model.eval()
        with torch.no_grad():
            logits, _ = model(x)
        assert torch.isfinite(logits).all(), "Post-training logits should be finite"

        # Verify generation works
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 5)).to(device)
        generated = model.generate(prompt, max_new_tokens=10)
        assert generated.shape == (1, 15), "Should generate 10 new tokens"
        assert (generated >= 0).all() and (generated < tiny_config.vocab_size).all(), (
            "Generated tokens should be valid"
        )


class TestE2ECATraining:
    """End-to-end test: CA init -> training -> verify."""

    def test_e2e_ca_training(self, tiny_config, mock_dataset, e2e_train_config, device):
        """Full pipeline with grid_ca init: model initializes, loss is finite."""
        model = GPT(tiny_config).to(device)

        engine = CAWeightEngine(variant="grid_ca", device=str(device))

        def ca_init_fn(m):
            return engine.develop_weights(m, n_steps=4)

        metrics = train(
            model=model,
            dataset=mock_dataset,
            config=e2e_train_config,
            init_fn=ca_init_fn,
            device=device,
        )

        # Verify training completed
        assert len(metrics["train_losses"]) == 50, "Should have 50 train losses"
        assert metrics["final_train_loss"] < float("inf"), (
            "Final train loss should be finite"
        )
        # Verify all losses are finite
        for i, loss in enumerate(metrics["train_losses"]):
            assert loss == loss, f"Train loss at step {i} should not be NaN"

        # Verify model weights are finite after training
        weight_tensors = model.get_weight_tensors()
        for name, tensor in weight_tensors.items():
            assert torch.isfinite(tensor).all(), (
                f"Weight '{name}' should be finite after CA-initialized training"
            )


class TestE2ELiveCATraining:
    """End-to-end test: LiveCATrainer with CA during training."""

    def test_e2e_live_ca_training(self, tiny_config, mock_dataset, device):
        """Full pipeline with LiveCATrainer: metrics include CA-specific ones."""
        model = GPT(tiny_config).to(device)

        ca_rules = {
            "attn": LocalNormCA(neighborhood_size=3, target_std=0.02),
            "ffn": LocalNormCA(neighborhood_size=3, target_std=0.02),
        }
        alpha_schedule = AlphaSchedule(
            mode="exponential_decay", alpha_0=0.01,
            decay=0.001, total_steps=30,
        )
        live_config = LiveCAConfig(
            ca_type="local_norm", ca_interval=1,
            alpha_schedule="exponential_decay",
            alpha_0=0.01, decay=0.001, total_steps=30,
            ca_sees_gradients=True, clamp_weights=True, max_weight=3.0,
        )
        trainer = LiveCATrainer(
            model=model, ca_rules=ca_rules,
            alpha_schedule=alpha_schedule,
            config=live_config, device=device,
        )
        trainer.configure_optimizer(lr=1e-3, weight_decay=0.1)

        metrics = trainer.train(mock_dataset)

        # Verify standard training metrics
        assert len(metrics["train_losses"]) == 30, "Should have 30 losses"
        assert metrics["final_loss"] < float("inf"), "Final loss should be finite"

        # Verify CA-specific metrics
        assert "ca_magnitudes" in metrics, "Should have CA magnitude metrics"
        assert "grad_magnitudes" in metrics, "Should have gradient magnitude metrics"
        assert "ca_alignments" in metrics, "Should have CA-gradient alignment metrics"
        assert "ca_contributions" in metrics, "Should have CA contribution ratio metrics"

        # Verify metrics are lists of correct length
        assert len(metrics["ca_magnitudes"]) == 30, "Should have 30 CA magnitude values"
        assert len(metrics["ca_contributions"]) == 30, (
            "Should have 30 CA contribution values"
        )

        # Verify model is still functional after live CA training
        x = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.block_size)).to(device)
        model.eval()
        with torch.no_grad():
            logits, _ = model(x)
        assert torch.isfinite(logits).all(), (
            "Post-LiveCA-training logits should be finite"
        )

        # Verify weights are finite
        for name, tensor in model.get_weight_tensors().items():
            assert torch.isfinite(tensor).all(), (
                f"Weight '{name}' should be finite after live CA training"
            )
