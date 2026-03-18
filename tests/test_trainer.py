"""Tests for the training loop."""

import pytest
import torch

from neurogen.config import GPTConfig, TrainConfig, get_device
from neurogen.data.shakespeare import ShakespeareDataset
from neurogen.model.gpt import GPT
from neurogen.training.trainer import evaluate, get_lr, train


@pytest.fixture(scope="module")
def dataset():
    return ShakespeareDataset()


class TestLRSchedule:
    def test_warmup(self):
        config = TrainConfig(
            max_steps=1000, warmup_steps=100, lr=1e-3, min_lr=1e-4
        )
        # At step 0, lr should be near 0
        assert get_lr(0, config) < config.lr
        # At warmup end, lr should be max
        assert abs(get_lr(99, config) - config.lr) < 1e-6

    def test_decay(self):
        config = TrainConfig(
            max_steps=1000, warmup_steps=100, lr=1e-3, min_lr=1e-4
        )
        lr_mid = get_lr(550, config)
        lr_end = get_lr(999, config)
        assert lr_mid > lr_end
        assert lr_end >= config.min_lr - 1e-8

    def test_monotonic_after_warmup(self):
        config = TrainConfig(
            max_steps=1000, warmup_steps=100, lr=1e-3, min_lr=1e-4
        )
        lrs = [get_lr(s, config) for s in range(100, 1000)]
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1] + 1e-10


class TestTraining:
    def test_loss_decreases(self, dataset, device):
        config = GPTConfig(
            block_size=32, vocab_size=dataset.vocab_size,
            n_layer=2, n_head=2, n_embd=64, dropout=0.0,
        )
        torch.manual_seed(42)
        model = GPT(config)
        train_config = TrainConfig(
            max_steps=100, eval_interval=50, eval_steps=5,
            batch_size=8, lr=1e-3, device=device, log_interval=10,
        )
        metrics = train(model, dataset, train_config)
        losses = [v["val_loss"] for v in metrics["val_loss"]]
        assert losses[-1] < losses[0], "Val loss should decrease over training"

    def test_eval_metrics(self, dataset, device):
        config = GPTConfig(
            block_size=32, vocab_size=dataset.vocab_size,
            n_layer=2, n_head=2, n_embd=64, dropout=0.0,
        )
        model = GPT(config).to(device)
        train_config = TrainConfig(
            max_steps=50, eval_interval=25, batch_size=4,
            lr=1e-3, device=device, eval_steps=5, log_interval=10,
        )
        eval_result = evaluate(model, dataset, train_config)
        assert "val_loss" in eval_result
        assert "train_loss" in eval_result
        assert not torch.isnan(torch.tensor(eval_result["val_loss"]))

    def test_metrics_structure(self, dataset, device):
        config = GPTConfig(
            block_size=32, vocab_size=dataset.vocab_size,
            n_layer=2, n_head=2, n_embd=64, dropout=0.0,
        )
        torch.manual_seed(42)
        model = GPT(config)
        train_config = TrainConfig(
            max_steps=100, eval_interval=25, eval_steps=5,
            batch_size=4, lr=1e-3, device=device, log_interval=10,
        )
        metrics = train(model, dataset, train_config)
        assert "train_loss" in metrics
        assert "val_loss" in metrics
        assert "gradient_norm" in metrics
        assert "total_train_time_s" in metrics
        assert metrics["best_val_loss"] is not None

    def test_deterministic(self, dataset, device):
        config = GPTConfig(
            block_size=32, vocab_size=dataset.vocab_size,
            n_layer=2, n_head=2, n_embd=64, dropout=0.0,
        )
        train_config = TrainConfig(
            max_steps=50, eval_interval=50, eval_steps=5,
            batch_size=4, lr=1e-3, device=device, log_interval=50,
        )

        torch.manual_seed(42)
        model1 = GPT(config)
        m1 = train(model1, dataset, train_config)

        torch.manual_seed(42)
        model2 = GPT(config)
        m2 = train(model2, dataset, train_config)

        loss1 = m1["val_loss"][-1]["val_loss"]
        loss2 = m2["val_loss"][-1]["val_loss"]
        assert abs(loss1 - loss2) < 1e-4, f"Non-deterministic: {loss1} vs {loss2}"
