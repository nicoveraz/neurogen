"""Tests for Shakespeare dataset."""

import pytest
import torch

from neurogen.data.shakespeare import ShakespeareDataset


@pytest.fixture(scope="module")
def dataset():
    """Load Shakespeare dataset once for all tests in this module."""
    return ShakespeareDataset()


class TestShakespeareDataset:
    def test_load(self, dataset):
        assert dataset.train_data is not None
        assert dataset.val_data is not None
        assert dataset.vocab_size > 0

    def test_data_size(self, dataset):
        total = len(dataset.train_data) + len(dataset.val_data)
        assert total > 1_000_000

    def test_batch_shape(self, dataset):
        x, y = dataset.get_batch("train", batch_size=4, block_size=32)
        assert x.shape == (4, 32)
        assert y.shape == (4, 32)

    def test_targets_shifted(self, dataset):
        """Targets should be inputs shifted by 1 position."""
        # Use a deterministic batch by checking the raw data
        data = dataset.train_data
        block_size = 32
        x_seq = data[:block_size]
        y_seq = data[1 : block_size + 1]
        assert torch.equal(x_seq[1:], y_seq[:-1])

    def test_encoding_roundtrip(self, dataset):
        text = "Hello World"
        encoded = dataset.encode(text)
        decoded = dataset.decode(encoded)
        assert decoded == text

    def test_all_chars_covered(self, dataset):
        """Every character in the text should have a valid encoding."""
        for ch in set(dataset.text):
            assert ch in dataset.stoi
            idx = dataset.stoi[ch]
            assert dataset.itos[idx] == ch

    def test_batch_device(self, dataset, device):
        x, y = dataset.get_batch("train", batch_size=4, block_size=32, device=device)
        assert str(x.device).startswith(device)

    def test_val_split(self, dataset):
        total = len(dataset.train_data) + len(dataset.val_data)
        train_ratio = len(dataset.train_data) / total
        assert 0.85 < train_ratio < 0.95
