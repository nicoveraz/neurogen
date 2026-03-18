"""Hand-designed CA rules encoding known structural priors.

These are non-learned rules that produce specific weight structures:
- Block-diagonal (modular)
- Low-rank + sparse
- Frequency-biased attention
"""

import math

import torch

from neurogen.model.gpt import GPT


def block_diagonal_init(model: GPT, n_blocks: int = 4, **kwargs) -> dict[str, torch.Tensor]:
    """Initialize with block-diagonal structure (modularity prior).

    Args:
        model: The GPT model.
        n_blocks: Number of diagonal blocks.

    Returns:
        Dict of weight tensors with block-diagonal structure.
    """
    weights = {}
    for name, param in model.get_weight_tensors().items():
        H, W = param.shape
        w = torch.zeros(H, W, dtype=param.dtype)
        block_h = H // n_blocks
        block_w = W // n_blocks

        for b in range(n_blocks):
            h_start = b * block_h
            w_start = b * block_w
            h_end = min(H, h_start + block_h)
            w_end = min(W, w_start + block_w)
            block = torch.randn(h_end - h_start, w_end - w_start)
            fan_in = w_end - w_start
            block *= math.sqrt(2.0 / fan_in)
            w[h_start:h_end, w_start:w_end] = block

        weights[name] = w.to(param.device)
    return weights


def low_rank_sparse_init(
    model: GPT, rank: int = 8, sparsity: float = 0.95, **kwargs
) -> dict[str, torch.Tensor]:
    """Initialize as low-rank matrix plus sparse perturbation.

    W = U @ V + S, where U is (H, rank), V is (rank, W), S is sparse.

    Args:
        model: The GPT model.
        rank: Rank of the low-rank component.
        sparsity: Fraction of zeros in sparse component.

    Returns:
        Dict of weight tensors.
    """
    weights = {}
    for name, param in model.get_weight_tensors().items():
        H, W = param.shape
        # Low-rank component
        U = torch.randn(H, rank) * (1.0 / math.sqrt(rank))
        V = torch.randn(rank, W) * (1.0 / math.sqrt(rank))
        low_rank = U @ V

        # Sparse component
        sparse = torch.randn(H, W) * 0.01
        mask = torch.rand(H, W) > sparsity
        sparse = sparse * mask

        w = low_rank + sparse
        # Scale
        if w.std() > 1e-8:
            w = w / w.std() * 0.02

        weights[name] = w.to(param.device)
    return weights


def frequency_biased_init(model: GPT, **kwargs) -> dict[str, torch.Tensor]:
    """Initialize attention heads with different frequency biases.

    Early heads attend to low frequencies (broad context),
    later heads attend to high frequencies (local detail).

    Args:
        model: The GPT model.

    Returns:
        Dict of weight tensors.
    """
    weights = {}
    n_head = model.config.n_head
    n_embd = model.config.n_embd
    head_dim = n_embd // n_head

    for name, param in model.get_weight_tensors().items():
        H, W = param.shape

        if "c_attn" in name:
            # Q, K, V are concatenated: (3*n_embd, n_embd)
            w = torch.zeros(H, W, dtype=param.dtype)
            for head_idx in range(n_head):
                # Frequency bias: head 0 = low freq, head n-1 = high freq
                freq_scale = (head_idx + 1) / n_head
                for qkv_idx in range(3):
                    row_start = qkv_idx * n_embd + head_idx * head_dim
                    row_end = row_start + head_dim
                    block = torch.randn(head_dim, W) * 0.02
                    # Apply frequency-dependent scaling
                    for d in range(head_dim):
                        freq = (d + 1) * freq_scale
                        block[d] *= math.exp(-0.1 * abs(freq - head_dim / 2))
                    w[row_start:row_end] = block
        elif "c_proj" in name:
            w = torch.randn(H, W) * (0.02 / math.sqrt(2 * model.config.n_layer))
        elif "c_fc" in name:
            w = torch.randn(H, W) * math.sqrt(2.0 / W)
        else:
            w = torch.randn(H, W) * 0.02

        weights[name] = w.to(param.device)
    return weights


HANDCRAFTED_INITIALIZERS = {
    "block_diagonal": block_diagonal_init,
    "low_rank_sparse": low_rank_sparse_init,
    "frequency_biased": frequency_biased_init,
}
