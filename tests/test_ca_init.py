"""Tests for CA / structured weight initializers (ca_rules.py)."""
import torch

from ca_rules import block_diagonal_init


def test_block_diagonal_is_sparse_off_blocks():
    torch.manual_seed(0)
    h, w, n_blocks = 16, 16, 4
    W = block_diagonal_init((h, w), n_blocks=n_blocks, target_std=0.02)
    assert W.shape == (h, w)
    bh, bw = h // n_blocks, w // n_blocks
    on_block = torch.zeros(h, w, dtype=torch.bool)
    for b in range(n_blocks):
        on_block[b * bh:b * bh + bh, b * bw:b * bw + bw] = True
    # Everything outside the diagonal blocks must be exactly zero.
    assert torch.all(W[~on_block] == 0.0)
    # The diagonal blocks must carry the signal.
    assert W[on_block].abs().sum() > 0


def test_block_diagonal_rescaled_to_target_std():
    torch.manual_seed(0)
    W = block_diagonal_init((64, 64), n_blocks=4, target_std=0.02)
    # rescale() normalizes the std of the FULL tensor (zeros included) to target.
    assert abs(W.std().item() - 0.02) < 1e-3
