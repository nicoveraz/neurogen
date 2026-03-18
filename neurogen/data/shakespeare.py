"""Shakespeare character-level dataset for language modeling."""

import os
import urllib.request
from pathlib import Path

import torch

_DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DATA_FILE = _DATA_DIR / "shakespeare.txt"


class ShakespeareDataset:
    """Character-level Shakespeare dataset.

    Downloads the Tiny Shakespeare corpus, builds a character-level vocabulary,
    and provides train/val splits with batch sampling.

    Args:
        data_dir: Directory to cache the downloaded text. Defaults to project data/.
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        if data_dir is not None:
            self._data_dir = Path(data_dir)
            self._data_file = self._data_dir / "shakespeare.txt"
        else:
            self._data_dir = _DATA_DIR
            self._data_file = _DATA_FILE

        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._download()
        self._build_vocab()
        self._split_data()

    def _download(self) -> None:
        """Download the Shakespeare text if not already cached."""
        if self._data_file.exists():
            return
        print(f"Downloading Shakespeare to {self._data_file}...")
        urllib.request.urlretrieve(_DATA_URL, self._data_file)
        print("Download complete.")

    def _build_vocab(self) -> None:
        """Build character-level vocabulary from the text."""
        with open(self._data_file, "r", encoding="utf-8") as f:
            self._text = f.read()

        chars = sorted(set(self._text))
        self._stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self._itos: dict[int, str] = {i: ch for i, ch in enumerate(chars)}

    def _split_data(self) -> None:
        """Split encoded data into 90% train / 10% val."""
        data = torch.tensor(self.encode(self._text), dtype=torch.long)
        n = int(0.9 * len(data))
        self._train_data = data[:n]
        self._val_data = data[n:]

    @property
    def vocab_size(self) -> int:
        """Number of unique characters in the vocabulary."""
        return len(self._stoi)

    @property
    def train_data(self) -> torch.Tensor:
        """Training split as a 1-D tensor of token indices."""
        return self._train_data

    @property
    def val_data(self) -> torch.Tensor:
        """Validation split as a 1-D tensor of token indices."""
        return self._val_data

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token indices.

        Args:
            text: Input text string.

        Returns:
            List of integer token indices.
        """
        return [self._stoi[ch] for ch in text]

    def decode(self, ints: list[int]) -> str:
        """Decode a list of token indices into a string.

        Args:
            ints: List of integer token indices.

        Returns:
            Decoded text string.
        """
        return "".join(self._itos[i] for i in ints)

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of (input, target) pairs.

        Args:
            split: "train" or "val".
            batch_size: Number of sequences in the batch.
            block_size: Sequence length (context window).
            device: Device to place tensors on.

        Returns:
            Tuple of (x, y) tensors, each of shape (batch_size, block_size).
        """
        data = self._train_data if split == "train" else self._val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)
