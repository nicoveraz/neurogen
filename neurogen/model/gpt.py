"""MicroGPT: Minimal GPT implementation following Karpathy's nanoGPT style."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurogen.config import GPTConfig
from neurogen.model.components import Block


class GPT(nn.Module):
    """Minimal GPT language model.

    Features:
    - Configurable layers, heads, embedding dimension
    - Pre-norm transformer blocks
    - Weight tying between token embedding and LM head
    - get/set weight tensor interface for CA initialization
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.vocab_size > 0, "vocab_size must be set before creating model"
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Apply scaled init to output projections per GPT-2
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module) -> None:
        """Default weight initialization (GPT-2 style)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            idx: Token indices, shape (B, T).
            targets: Target token indices, shape (B, T). If None, no loss computed.

        Returns:
            Tuple of (logits, loss). Loss is None if targets not provided.
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Generate new tokens autoregressively.

        Args:
            idx: Conditioning token indices, shape (B, T).
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_k: If set, only sample from top-k logits.

        Returns:
            Extended token sequence, shape (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)

            # MPS-safe multinomial sampling
            try:
                idx_next = torch.multinomial(probs, num_samples=1)
            except RuntimeError:
                # Fallback to CPU for multinomial on MPS
                idx_next = torch.multinomial(
                    probs.cpu(), num_samples=1
                ).to(idx.device)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def get_weight_tensors(self) -> dict[str, torch.Tensor]:
        """Get all trainable weight matrices (no biases, no LayerNorm).

        Returns dict mapping parameter names to weight tensors.
        The token embedding (wte) is excluded since it's tied to lm_head.
        """
        weights = {}
        for name, param in self.named_parameters():
            # Skip biases
            if "bias" in name:
                continue
            # Skip LayerNorm parameters
            if "ln_" in name:
                continue
            # Skip wte (tied to lm_head)
            if "wte" in name:
                continue
            # Skip positional embedding (not a weight matrix in the CA sense)
            if "wpe" in name:
                continue
            # Skip dropout (not a parameter)
            weights[name] = param
        return weights

    def set_weight_tensors(self, weights: dict[str, torch.Tensor]) -> None:
        """Inject weight tensors into the model.

        Args:
            weights: Dict mapping parameter names to new weight tensors.
                     Names must match those from get_weight_tensors().
        """
        current_weights = self.get_weight_tensors()
        for name, tensor in weights.items():
            if name not in current_weights:
                raise KeyError(f"Unknown weight tensor: {name}")
            expected_shape = current_weights[name].shape
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch for {name}: "
                    f"expected {expected_shape}, got {tensor.shape}"
                )
            # Navigate to the actual parameter and set its data
            parts = name.split(".")
            module = self
            for part in parts[:-1]:
                module = getattr(module, part)
            param = getattr(module, parts[-1])
            param.data.copy_(tensor)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
