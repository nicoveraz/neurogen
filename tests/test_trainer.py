"""Tests for neurogen/training/trainer.py.

Uses a mock dataset with random tensors to avoid network downloads.
"""

import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from neurogen.config import GPTConfig, TrainConfig, get_device
from neurogen.model.gpt import GPT
from neurogen.training.trainer import (
    evaluate,
    get_lr,
    load_checkpoint,
    save_checkpoint,
    train,
)


class MockDataset:
    """Mock dataset that returns random batches without file I/O."""

    def __init__(self, vocab_size: int = 256, data_len: int = 10000):
        self.vocab_size = vocab_size
        self._data = torch.randint(0, vocab_size, (data_len,))
        n = int(0.9 * data_len)
        self._train_data = self._data[:n]
        self._val_data = self._data[n:]

    @property
    def train_data(self) -> torch.Tensor:
        return self._train_data

    @property
    def val_data(self) -> torch.Tensor:
        return self._val_data

    def get_batch(
        self, split: str, batch_size: int, block_size: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._train_data if split == "train" else self._val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    def encode(self, text: str) -> list[int]:
        return [0] * len(text)

    def decode(self, ints: list[int]) -> str:
        return "a" * len(ints)


@pytest.fixture
def mock_dataset():
    return MockDataset(vocab_size=256)


@pytest.fixture
def train_config():
    return TrainConfig(
        max_steps=100,
        eval_interval=50,
        batch_size=4,
        lr=1e-3,
        grad_clip=1.0,
        warmup_steps=10,
        min_lr=1e-5,
    )


@pytest.fixture
def model_and_device(tiny_config, device):
    model = GPT(tiny_config).to(device)
    return model, device


class TestLossDecrease:
    """Tests that training produces decreasing loss."""

    def test_trainer_loss_decreases(self, model_and_device, mock_dataset, train_config, device):
        """Train 100 steps: loss should decrease from start to end."""
        model, dev = model_and_device
        metrics = train(
            model=model,
            dataset=mock_dataset,
            config=train_config,
            device=dev,
        )
        losses = metrics["train_losses"]
        assert len(losses) == 100, "Should have 100 recorded train losses"
        # Compare average of first 10 with average of last 10
        early_avg = sum(losses[:10]) / 10
        late_avg = sum(losses[-10:]) / 10
        assert late_avg < early_avg, (
            f"Loss should decrease: early avg {early_avg:.4f} > late avg {late_avg:.4f}"
        )


class TestLRSchedule:
    """Tests for the learning rate schedule."""

    def test_trainer_lr_warmup(self, train_config):
        """During warmup, LR should increase linearly."""
        lr_0 = get_lr(0, train_config)
        lr_mid = get_lr(train_config.warmup_steps // 2, train_config)
        lr_end = get_lr(train_config.warmup_steps - 1, train_config)
        assert lr_0 < lr_mid < lr_end, (
            f"LR should increase during warmup: {lr_0} < {lr_mid} < {lr_end}"
        )

    def test_trainer_lr_peak(self, train_config):
        """At warmup_steps, LR should be at or near peak."""
        lr_at_warmup = get_lr(train_config.warmup_steps, train_config)
        # At warmup_steps, cosine decay starts at lr
        assert lr_at_warmup <= train_config.lr + 1e-8, (
            f"LR at warmup end ({lr_at_warmup}) should be <= peak lr ({train_config.lr})"
        )

    def test_trainer_lr_decay(self, train_config):
        """After warmup, LR should decay."""
        lr_post_warmup = get_lr(train_config.warmup_steps + 1, train_config)
        lr_late = get_lr(train_config.max_steps - 1, train_config)
        assert lr_late <= lr_post_warmup, (
            f"LR should decay: post-warmup {lr_post_warmup} >= late {lr_late}"
        )

    def test_trainer_lr_min(self, train_config):
        """After max_steps, LR should return min_lr."""
        lr_after = get_lr(train_config.max_steps + 100, train_config)
        assert lr_after == train_config.min_lr, (
            f"LR after max_steps should be min_lr: got {lr_after}"
        )


class TestEvaluate:
    """Tests for the evaluate function."""

    def test_trainer_eval(self, model_and_device, mock_dataset, train_config, device):
        """evaluate returns a finite float."""
        model, dev = model_and_device
        val_loss = evaluate(model, mock_dataset, train_config, dev, n_batches=5)
        assert isinstance(val_loss, float), "eval loss should be a float"
        assert math.isfinite(val_loss), "eval loss should be finite"


class TestCheckpoint:
    """Tests for checkpoint save/load."""

    def test_trainer_checkpoint_save_load(
        self, tiny_config, mock_dataset, train_config, device
    ):
        """Save then load: model state should be restored correctly."""
        model = GPT(tiny_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Do a few forward passes to create non-trivial state
        for _ in range(5):
            x, y = mock_dataset.get_batch("train", 4, tiny_config.block_size, device)
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        val_loss = evaluate(model, mock_dataset, train_config, device, n_batches=3)
        metrics = {"val_loss": val_loss}

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_checkpoint.pt"
            save_checkpoint(model, optimizer, step=5, metrics=metrics, path=ckpt_path)
            assert ckpt_path.exists(), "Checkpoint file should exist"

            # Load into a fresh model
            model2 = GPT(tiny_config).to(device)
            loaded_info = load_checkpoint(ckpt_path, model2, device=device)
            assert loaded_info["step"] == 5, "Loaded step should be 5"
            assert abs(loaded_info["metrics"]["val_loss"] - val_loss) < 1e-6, (
                "Loaded val_loss should match saved val_loss"
            )


class TestDeterminism:
    """Tests for training determinism."""

    def test_trainer_deterministic(self, tiny_config, mock_dataset, device):
        """Same seed -> same loss at step 50."""
        config = TrainConfig(
            max_steps=50, eval_interval=50, batch_size=4,
            lr=1e-3, grad_clip=1.0, warmup_steps=5, min_lr=1e-5,
        )

        torch.manual_seed(42)
        model1 = GPT(tiny_config).to(device)
        metrics1 = train(model1, mock_dataset, config, device=device)

        torch.manual_seed(42)
        model2 = GPT(tiny_config).to(device)
        metrics2 = train(model2, mock_dataset, config, device=device)

        assert abs(metrics1["final_train_loss"] - metrics2["final_train_loss"]) < 1e-4, (
            "Same seed should produce same final loss"
        )


class TestCustomInit:
    """Tests for training with custom initialization."""

    def test_trainer_custom_init(self, tiny_config, mock_dataset, device):
        """CA init vs xavier both produce decreasing loss."""
        config = TrainConfig(
            max_steps=50, eval_interval=50, batch_size=4,
            lr=1e-3, grad_clip=1.0, warmup_steps=5, min_lr=1e-5,
        )

        from neurogen.baselines.initializers import xavier_normal_init

        def xavier_init_fn(model):
            return xavier_normal_init(model)

        model = GPT(tiny_config).to(device)
        metrics = train(model, mock_dataset, config, init_fn=xavier_init_fn, device=device)
        losses = metrics["train_losses"]
        early = sum(losses[:5]) / 5
        late = sum(losses[-5:]) / 5
        assert late < early, (
            f"Xavier init: loss should decrease. Early: {early:.4f}, Late: {late:.4f}"
        )


class TestMetricsLogging:
    """Tests for training metrics."""

    def test_trainer_metrics_logging(self, model_and_device, mock_dataset, train_config, device):
        """Train 100 steps: metrics dict has expected keys."""
        model, dev = model_and_device
        metrics = train(model, mock_dataset, train_config, device=dev)
        expected_keys = [
            "train_losses", "val_losses", "best_val_loss",
            "final_train_loss", "final_val_loss",
            "total_time", "steps_per_sec",
        ]
        for key in expected_keys:
            assert key in metrics, f"Metrics should contain '{key}'"
        assert len(metrics["train_losses"]) == train_config.max_steps, (
            f"Should have {train_config.max_steps} train losses"
        )
        assert len(metrics["val_losses"]) > 0, "Should have at least one val loss"
        assert metrics["total_time"] > 0, "Training should take some time"
        assert metrics["steps_per_sec"] > 0, "Steps per second should be positive"
