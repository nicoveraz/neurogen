"""
Fixed data preparation and evaluation for NeuroGen autoresearch.
Downloads Shakespeare, provides dataloaders and evaluation.
DO NOT MODIFY THIS FILE — it is the fixed evaluation harness.

Usage: python prepare.py
"""

import os
import math
import urllib.request
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 256         # context length
TIME_BUDGET = 120         # training time budget in seconds (2 minutes)
EVAL_BATCHES = 20         # number of batches for validation eval
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_FILE = DATA_DIR / "shakespeare.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def download_data():
    """Download Shakespeare text if not cached."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        print(f"Data already exists at {DATA_FILE}")
        return
    print(f"Downloading Shakespeare to {DATA_FILE}...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    print("Download complete.")


def load_data():
    """Load and encode Shakespeare data. Returns (train_data, val_data, vocab_size, encode, decode)."""
    if not DATA_FILE.exists():
        download_data()

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(l):
        return "".join(itos[i] for i in l)

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, len(chars), encode, decode


def get_batch(data, batch_size, block_size, device):
    """Sample a random batch from data."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def evaluate_val_loss(model, val_data, batch_size, block_size, device,
                      n_batches=EVAL_BATCHES):
    """Evaluate model on validation set. Returns average val_loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(val_data, batch_size, block_size, device)
            _, loss = model(x, y)
            total_loss += loss.item()
    model.train()
    return total_loss / n_batches


def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Main: one-time data preparation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    download_data()
    train_data, val_data, vocab_size, encode, decode = load_data()
    print(f"Vocab size: {vocab_size}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    print(f"Device: {get_device()}")
    print("Data preparation complete.")
