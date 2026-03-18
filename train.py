"""
NeuroGen train.py — the single modifiable file for autoresearch.

This file contains the model, initialization, and training loop.
The AI researcher modifies this file to test different approaches.
DO NOT import from the neurogen package — keep everything self-contained.

Usage: python train.py
Output: prints a result summary dict to stdout (parsed by the harness).
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    load_data, get_batch, evaluate_val_loss, get_device,
    MAX_SEQ_LEN, TIME_BUDGET,
)

# ---------------------------------------------------------------------------
# Model hyperparameters (tune these)
# ---------------------------------------------------------------------------

N_LAYER = 4
N_HEAD = 4
N_EMBD = 128
DROPOUT = 0.1
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
INIT_METHOD = "xavier_normal"  # try: "default", "xavier_normal", "kaiming", "ca"

# ---------------------------------------------------------------------------
# Model definition (same as neurogen GPT, but self-contained)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.dropout_p = dropout
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ---------------------------------------------------------------------------
# Initialization methods
# ---------------------------------------------------------------------------

def init_default(model):
    """PyTorch default initialization (already applied)."""
    pass

def init_xavier_normal(model):
    """Xavier normal initialization for weight matrices."""
    for name, p in model.named_parameters():
        if p.dim() >= 2 and "ln_" not in name and "pos_emb" not in name:
            nn.init.xavier_normal_(p)

def init_kaiming(model):
    """Kaiming normal initialization for weight matrices."""
    for name, p in model.named_parameters():
        if p.dim() >= 2 and "ln_" not in name and "pos_emb" not in name:
            nn.init.kaiming_normal_(p, nonlinearity="relu")

def init_scaled_normal(model, std=0.02):
    """Scaled normal initialization (GPT-2 style)."""
    for name, p in model.named_parameters():
        if p.dim() >= 2 and "ln_" not in name and "pos_emb" not in name:
            nn.init.normal_(p, mean=0.0, std=std)

def init_orthogonal(model):
    """Orthogonal initialization for weight matrices."""
    for name, p in model.named_parameters():
        if p.dim() >= 2 and "ln_" not in name and "pos_emb" not in name:
            nn.init.orthogonal_(p)

INIT_METHODS = {
    "default": init_default,
    "xavier_normal": init_xavier_normal,
    "kaiming": init_kaiming,
    "scaled_normal": init_scaled_normal,
    "orthogonal": init_orthogonal,
}

# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    device = get_device()
    print(f"Device: {device}")

    # Load data
    train_data, val_data, vocab_size, encode, decode = load_data()
    block_size = MAX_SEQ_LEN

    # Create model
    model = GPT(vocab_size, block_size, N_LAYER, N_HEAD, N_EMBD, DROPOUT).to(device)
    num_params = model.count_parameters()
    print(f"Parameters: {num_params:,}")

    # Apply initialization
    init_fn = INIT_METHODS.get(INIT_METHOD, init_xavier_normal)
    init_fn(model)
    print(f"Init method: {INIT_METHOD}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Training
    model.train()
    step = 0
    warmup_steps = 100
    max_steps = 100_000  # will be cut short by time budget
    min_lr = LEARNING_RATE / 10

    t0 = time.time()

    while True:
        elapsed = time.time() - t0
        if elapsed >= TIME_BUDGET:
            break

        # Get batch
        x, y = get_batch(train_data, BATCH_SIZE, block_size, device)

        # LR schedule
        lr = get_lr(step, warmup_steps, max_steps, LEARNING_RATE, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward + backward
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.1f}s")

        step += 1

    training_time = time.time() - t0

    # Evaluate
    val_loss = evaluate_val_loss(model, val_data, BATCH_SIZE, block_size, device)
    total_time = time.time() - t0

    # Generate a sample
    prompt = encode("\n")
    prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        gen_ids = prompt_t
        for _ in range(200):
            idx_cond = gen_ids[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / 0.8
            probs = F.softmax(logits, dim=-1)
            try:
                next_id = torch.multinomial(probs, num_samples=1)
            except RuntimeError:
                next_id = torch.multinomial(probs.cpu(), num_samples=1).to(device)
            gen_ids = torch.cat([gen_ids, next_id], dim=1)
    sample = decode(gen_ids[0].tolist())
    print(f"\n--- Sample ---\n{sample}\n--- End Sample ---\n")

    # Print results
    print(f"RESULT: val_loss={val_loss:.4f} | "
          f"train_loss={loss.item():.4f} | "
          f"steps={step} | "
          f"params={num_params} | "
          f"init={INIT_METHOD} | "
          f"n_layer={N_LAYER} | "
          f"n_head={N_HEAD} | "
          f"n_embd={N_EMBD} | "
          f"lr={LEARNING_RATE} | "
          f"batch_size={BATCH_SIZE} | "
          f"training_time={training_time:.1f}s | "
          f"total_time={total_time:.1f}s")


if __name__ == "__main__":
    train()
