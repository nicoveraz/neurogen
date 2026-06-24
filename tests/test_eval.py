"""Tests for the bits-per-byte evaluation math (prepare.evaluate_val_bpb).

Anchors the constant 8.0 that evaluate_quality.py relies on: a model that
predicts a uniform distribution over the 256 byte values has cross-entropy
ln(256) nats, i.e. exactly 8.0 bits per byte. Runs on CPU, no data download.
"""
import math

import torch

from prepare import evaluate_val_bpb, get_batch, VOCAB_SIZE


class _UniformModel(torch.nn.Module):
    """Predicts a uniform 256-way distribution: cross-entropy == ln(256) nats."""

    def forward(self, x, y):
        loss = torch.tensor(math.log(VOCAB_SIZE))
        return None, loss


def test_uniform_logits_give_bpb_8():
    torch.manual_seed(0)  # get_batch uses the global RNG; seed for determinism
    val_data = torch.randint(0, VOCAB_SIZE, (10_000,), dtype=torch.long)
    bpb = evaluate_val_bpb(_UniformModel(), val_data, batch_size=4, block_size=64,
                           device="cpu", n_tokens=4 * 64 * 5)
    assert abs(bpb - 8.0) < 1e-6  # ln(256)/ln(2) == 8 up to float roundoff


def test_bpb_is_nats_over_ln2():
    # bpb = cross_entropy_nats / ln(2); a 2-nat loss is 2/ln2 bits per byte.
    class _TwoNat(torch.nn.Module):
        def forward(self, x, y):
            return None, torch.tensor(2.0)

    torch.manual_seed(0)
    val_data = torch.randint(0, VOCAB_SIZE, (5_000,), dtype=torch.long)
    bpb = evaluate_val_bpb(_TwoNat(), val_data, batch_size=2, block_size=32,
                           device="cpu", n_tokens=2 * 32 * 3)
    assert abs(bpb - 2.0 / math.log(2)) < 1e-6


def test_get_batch_shapes_and_determinism():
    data = torch.arange(1000, dtype=torch.long)
    torch.manual_seed(123)
    x1, y1 = get_batch(data, batch_size=8, block_size=16, device="cpu")
    assert x1.shape == (8, 16) and y1.shape == (8, 16)
    # y is x shifted by one (next-token targets).
    torch.manual_seed(123)
    x2, y2 = get_batch(data, batch_size=8, block_size=16, device="cpu")
    assert torch.equal(x1, x2) and torch.equal(y1, y2)  # same global seed -> same batch
    # check the shift relationship on the first row
    start = int(x1[0, 0].item())
    assert torch.equal(x1[0], torch.arange(start, start + 16))
    assert torch.equal(y1[0], torch.arange(start + 1, start + 17))
