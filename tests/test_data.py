"""Tests for neurogen/data/shakespeare.py.

Uses a temporary file with synthetic text to avoid network downloads.
"""

import os
import tempfile
from pathlib import Path

import pytest
import torch

from neurogen.data.shakespeare import ShakespeareDataset


@pytest.fixture
def mock_shakespeare_dir():
    """Create a temporary directory with a small synthetic Shakespeare file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        text = (
            "First Citizen:\n"
            "Before we proceed any further, hear me speak.\n\n"
            "All:\n"
            "Speak, speak.\n\n"
            "First Citizen:\n"
            "You are all resolved rather to die than to famish?\n\n"
            "All:\n"
            "Resolved. resolved.\n\n"
            "First Citizen:\n"
            "First, you know Caius Marcius is chief enemy to the people.\n\n"
        )
        # Repeat to ensure enough data for train/val splits
        text = text * 20
        data_file = Path(tmpdir) / "shakespeare.txt"
        data_file.write_text(text, encoding="utf-8")
        yield tmpdir


@pytest.fixture
def dataset(mock_shakespeare_dir):
    """Create a ShakespeareDataset from the mock data."""
    return ShakespeareDataset(data_dir=mock_shakespeare_dir)


class TestShakespeareLoad:
    """Tests for dataset loading."""

    def test_shakespeare_load(self, dataset):
        """Dataset creates successfully with valid vocab and data splits."""
        assert dataset.vocab_size > 0, "Vocabulary should be non-empty"
        assert len(dataset.train_data) > 0, "Train data should be non-empty"
        assert len(dataset.val_data) > 0, "Val data should be non-empty"

    def test_shakespeare_vocab_characters(self, dataset):
        """Vocab includes expected characters from the mock data."""
        # Use characters known to be in our mock data
        test_text = "Speak"
        encoded = dataset.encode(test_text)
        assert len(encoded) == len(test_text), (
            f"Encoding '{test_text}' should produce {len(test_text)} tokens"
        )


class TestShakespeareBatch:
    """Tests for batch sampling."""

    def test_shakespeare_batch_shape(self, dataset, device):
        """get_batch returns correct shapes (batch_size, block_size)."""
        batch_size = 4
        block_size = 16
        x, y = dataset.get_batch("train", batch_size, block_size, device)
        assert x.shape == (batch_size, block_size), (
            f"Expected x shape ({batch_size}, {block_size}), got {x.shape}"
        )
        assert y.shape == (batch_size, block_size), (
            f"Expected y shape ({batch_size}, {block_size}), got {y.shape}"
        )

    def test_shakespeare_batch_targets(self, dataset, device):
        """y is x shifted by 1 position in the source data."""
        batch_size = 2
        block_size = 8
        # Use a fixed seed for reproducibility
        torch.manual_seed(42)
        x, y = dataset.get_batch("train", batch_size, block_size, device)
        # x and y should be on the correct device
        assert x.device.type == device or (
            device == "mps" and x.device.type == "mps"
        ), f"x should be on {device}"
        # Shapes must match
        assert x.shape == y.shape, "x and y should have the same shape"


class TestShakespeareEncoding:
    """Tests for encode/decode."""

    def test_shakespeare_encoding_roundtrip(self, dataset):
        """encode then decode matches original for known chars."""
        original = "Speak, speak."
        encoded = dataset.encode(original)
        decoded = dataset.decode(encoded)
        assert decoded == original, (
            f"Roundtrip failed: expected '{original}', got '{decoded}'"
        )

    def test_shakespeare_encoding_types(self, dataset):
        """encode returns list of ints, decode returns str."""
        encoded = dataset.encode("hello")
        assert isinstance(encoded, list), "encode should return a list"
        assert all(isinstance(i, int) for i in encoded), (
            "All encoded values should be integers"
        )
        decoded = dataset.decode(encoded)
        assert isinstance(decoded, str), "decode should return a string"
