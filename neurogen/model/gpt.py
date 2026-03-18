"""MicroGPT: A minimal character-level GPT implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurogen.config import GPTConfig
from neurogen.model.components import Block


class GPT(nn.Module):
    """Minimal GPT for character-level language modeling.

    A decoder-only transformer with token embeddings, positional embeddings,
    a stack of transformer blocks, and a language modeling head. Uses weight
    tying between the token embedding and the LM head.

    Args:
        config: GPT model configuration.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.vocab_size > 0, "vocab_size must be set before creating model"
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd, dtype=torch.float32)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd, dtype=torch.float32)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, dtype=torch.float32)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False, dtype=torch.float32)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Apply default initialization to module parameters.

        Args:
            module: The module to initialize.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass through the model.

        Args:
            idx: Input token indices of shape (B, T).
            targets: Target token indices of shape (B, T), or None.

        Returns:
            Tuple of (logits, loss). Loss is None if targets is None.
        """
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok = self.tok_emb(idx)
        positional = self.pos_emb(pos)
        x = self.drop(tok + positional)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    def get_weight_tensors(self) -> dict[str, torch.Tensor]:
        """Extract trainable weight matrices (not biases, LayerNorm, or pos emb).

        Returns a dict mapping parameter names to their tensors. Due to weight
        tying, lm_head.weight is not included separately (it is the same tensor
        as tok_emb.weight).

        Returns:
            Dictionary of {name: tensor} for trainable weight matrices.
        """
        weights: dict[str, torch.Tensor] = {}
        for name, param in self.named_parameters():
            # Skip biases
            if "bias" in name:
                continue
            # Skip LayerNorm parameters
            if "ln_" in name:
                continue
            # Skip positional embeddings
            if "pos_emb" in name:
                continue
            # Skip lm_head (tied to tok_emb)
            if "lm_head" in name:
                continue
            # Skip dropout (not a parameter, but be safe)
            if "drop" in name:
                continue
            weights[name] = param.data
        return weights

    def set_weight_tensors(self, weights: dict[str, torch.Tensor]) -> None:
        """Inject weight tensors into the model.

        Args:
            weights: Dictionary of {name: tensor} matching get_weight_tensors keys.
        """
        state = dict(self.named_parameters())
        for name, tensor in weights.items():
            if name in state:
                state[name].data.copy_(tensor)

    def count_parameters(self) -> int:
        """Count total number of trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
            idx: Conditioning token indices of shape (B, T).
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_k: If set, only sample from the top k most likely tokens.

        Returns:
            Token indices of shape (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)

            # MPS fallback for multinomial
            try:
                idx_next = torch.multinomial(probs, num_samples=1)
            except RuntimeError:
                probs_cpu = probs.cpu()
                idx_next = torch.multinomial(probs_cpu, num_samples=1).to(idx.device)

            idx = torch.cat([idx, idx_next], dim=1)

        return idx
