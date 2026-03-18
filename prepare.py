"""
Fixed data preparation and evaluation for NeuroGen autoresearch.
Downloads TinyStories, creates byte-encoded shards, provides evaluation.
DO NOT MODIFY THIS FILE — it is the fixed evaluation harness.

Usage: uv run prepare.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256              # byte-level encoding
MAX_SEQ_LEN = 256             # context length
TIME_BUDGET = 120             # seconds per experiment (2 minutes)
EVAL_TOKENS = 100_000         # tokens for validation eval
DATA_DIR = Path.home() / ".cache" / "neurogen"
SHARD_SIZE = 1_000_000        # tokens per shard

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def download_and_prepare():
    """Download TinyStories and create byte-encoded shards in DATA_DIR."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if (DATA_DIR / "train_000.bin").exists():
        print(f"Data already prepared in {DATA_DIR}")
        return

    print("Downloading TinyStories dataset...")
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories")

    for split_name, split_key in [("train", "train"), ("val", "validation")]:
        print(f"Processing {split_name} split...")
        all_tokens = bytearray()
        for example in ds[split_key]:
            text = example["text"]
            all_tokens.extend(text.encode("utf-8"))

        tokens = np.frombuffer(bytes(all_tokens), dtype=np.uint8)
        print(f"  {split_name}: {len(tokens):,} tokens")

        # Write shards
        n_shards = max(1, len(tokens) // SHARD_SIZE)
        for i in range(n_shards):
            start = i * SHARD_SIZE
            end = min(start + SHARD_SIZE, len(tokens))
            shard = tokens[start:end]
            path = DATA_DIR / f"{split_name}_{i:03d}.bin"
            shard.tofile(path)

        # Write any remainder
        remainder_start = n_shards * SHARD_SIZE
        if remainder_start < len(tokens):
            shard = tokens[remainder_start:]
            path = DATA_DIR / f"{split_name}_{n_shards:03d}.bin"
            shard.tofile(path)

    print(f"Data prepared in {DATA_DIR}")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(split: str = "train") -> torch.Tensor:
    """Load tokenized data from binary shards. Returns a 1D LongTensor."""
    shards = sorted(DATA_DIR.glob(f"{split}_*.bin"))
    if not shards:
        raise FileNotFoundError(
            f"No {split} shards found in {DATA_DIR}. Run: uv run prepare.py"
        )
    arrays = [np.fromfile(p, dtype=np.uint8) for p in shards]
    return torch.tensor(np.concatenate(arrays), dtype=torch.long)


def get_batch(
    data: torch.Tensor, batch_size: int, block_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) sequences."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_val_bpb(
    model,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
    n_tokens: int = EVAL_TOKENS,
) -> float:
    """Evaluate model on validation data. Returns bits per byte (lower is better).

    For byte-level encoding: bpb = cross_entropy_loss / ln(2).
    """
    model.eval()
    n_batches = max(1, n_tokens // (batch_size * block_size))
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(val_data, batch_size, block_size, device)
            _, loss = model(x, y)
            total_loss += loss.item()
    model.train()
    return (total_loss / n_batches) / math.log(2)

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_peak_memory_mb() -> float:
    """Get peak memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / 1024**2  # macOS: bytes
    return rss / 1024  # Linux: KB

# ---------------------------------------------------------------------------
# Main: one-time data preparation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    download_and_prepare()
    train_data = load_data("train")
    val_data = load_data("val")
    print(f"\nVocab size: {VOCAB_SIZE}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    print(f"Sequence length: {MAX_SEQ_LEN}")
    print(f"Device: {get_device()}")
    print("Ready for training.")
