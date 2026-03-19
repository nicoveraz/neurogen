"""
NeuroGen train.py — model + training loop + CA hooks.

The AI researcher modifies this file to test CA initialization and live CA rules.
Self-contained: does NOT import from any local package except prepare.py and ca_rules.py.

Usage:
    uv run train.py                  # default 2 min (autoresearch speed)
    uv run train.py --minutes 10     # medium validation
    uv run train.py --minutes 30     # long validation
"""

import argparse
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    load_data, get_batch, evaluate_val_bpb, get_device, get_peak_memory_mb,
    VOCAB_SIZE, MAX_SEQ_LEN, TIME_BUDGET,
)
from ca_rules import grid_ca_develop

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DEPTH = 4                     # number of transformer layers
CHANNELS = DEPTH * 64         # embedding dimension (256 for depth 4)
N_HEADS = DEPTH               # number of attention heads (4 for depth 4)
N_KV_HEADS = N_HEADS          # KV heads (set < N_HEADS for GQA)
BATCH_SIZE = 32
LR = 2.5e-3
WEIGHT_DECAY = 0.1
DEVICE = get_device()

# ---------------------------------------------------------------------------
# Model (nanochat architecture, adapted for MPS)
# ---------------------------------------------------------------------------

def rms_norm(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


def has_ve(layer_idx, n_layer):
    """Value embedding on alternating layers, last always included."""
    return layer_idx % 2 == (n_layer - 1) % 2


class Attention(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, n_layer, layer_idx):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.c_q = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.ve_gate = (
            nn.Linear(min(12, n_embd), n_kv_head, bias=False)
            if has_ve(layer_idx, n_layer) else None
        )
        self._ve_channels = min(12, n_embd)

    def forward(self, x, ve, cos, sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None and self.ve_gate is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self._ve_channels]))
            v = v + gate.unsqueeze(-1) * ve
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = rms_norm(q) * 1.2, rms_norm(k) * 1.2
        if self.n_kv_head < self.n_head:
            r = self.n_head // self.n_kv_head
            k = k.repeat_interleave(r, dim=2)
            v = v.repeat_interleave(r, dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        try:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        except RuntimeError:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = q @ k.transpose(-2, -1) * scale
            mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
            att = att.masked_fill(~mask, float("-inf"))
            y = F.softmax(att, dim=-1) @ v
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, n_layer, layer_idx):
        super().__init__()
        self.attn = Attention(n_embd, n_head, n_kv_head, n_layer, layer_idx)
        self.mlp = MLP(n_embd)

    def forward(self, x, ve, cos, sin):
        x = x + self.attn(rms_norm(x), ve, cos, sin)
        x = x + self.mlp(rms_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_kv_head, n_embd):
        super().__init__()
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        head_dim = n_embd // n_head
        kv_dim = n_kv_head * head_dim

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, n_kv_head, n_layer, i) for i in range(n_layer)
        ])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Per-layer scalars
        self.resid_lambdas = nn.Parameter(torch.ones(n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(n_layer))

        # Smear: bigram prior from previous token
        smear_ch = min(24, n_embd)
        self.smear_gate = nn.Linear(smear_ch, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self._smear_ch = smear_ch

        # Backout: subtract mid-layer residual
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))

        # Value embeddings (ResFormer, alternating layers)
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(vocab_size, kv_dim)
            for i in range(n_layer) if has_ve(i, n_layer)
        })

        # Precompute RoPE
        ch = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (100000.0 ** (ch / head_dim))
        t = torch.arange(block_size, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, :, None, :], persistent=False)

        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        """Nanochat-style initialization."""
        n = self.n_embd
        s = 3**0.5 * n**-0.5

        nn.init.normal_(self.wte.weight, 0, 0.8)
        nn.init.normal_(self.lm_head.weight, 0, 0.001)

        for b in self.blocks:
            nn.init.uniform_(b.attn.c_q.weight, -s, s)
            nn.init.uniform_(b.attn.c_k.weight, -s, s)
            nn.init.uniform_(b.attn.c_v.weight, -s, s)
            nn.init.zeros_(b.attn.c_proj.weight)
            nn.init.uniform_(b.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            nn.init.zeros_(b.mlp.c_proj.weight)
            if b.attn.ve_gate is not None:
                nn.init.uniform_(b.attn.ve_gate.weight, 0, 0.02)

        for i in range(self.n_layer):
            self.resid_lambdas.data[i] = 1.15 - 0.10 * i / max(self.n_layer - 1, 1)
            self.x0_lambdas.data[i] = 0.20 - 0.15 * i / max(self.n_layer - 1, 1)

        for ve in self.value_embeds.values():
            nn.init.uniform_(ve.weight, -s, s)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        cos, sin = self.cos[:, :T], self.sin[:, :T]

        x = rms_norm(self.wte(idx))

        # Smear: bigram prior
        if T > 1:
            gate = self.smear_lambda * torch.sigmoid(
                self.smear_gate(x[:, 1:, :self._smear_ch])
            )
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

        x0 = x
        mid = self.n_layer // 2
        x_mid = None

        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos, sin)
            if i == mid:
                x_mid = x

        if x_mid is not None:
            x = x - self.backout_lambda * x_mid

        logits = self.lm_head(rms_norm(x)).float()
        logits = 15 * torch.tanh(logits / 15)  # soft-cap

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ---------------------------------------------------------------------------
# CA hooks (agent replaces these with CA variants)
# ---------------------------------------------------------------------------

def _is_ca_target(name: str, p: torch.Tensor) -> bool:
    """Check if a parameter should receive CA initialization."""
    if p.dim() < 2:
        return False
    if any(skip in name for skip in ("wte", "lm_head", "ve_gate")):
        return False
    if min(p.shape) < 8:
        return False
    return True


def initialize_weights(model):
    """Xavier + CA perturbation: xavier base with 10% CA structure on top."""
    seeds = ["center", "diagonal", "distributed", "random"]
    with torch.no_grad():
        for i, (name, p) in enumerate(model.named_parameters()):
            if _is_ca_target(name, p):
                # Xavier base
                nn.init.xavier_uniform_(p)
                # Add small CA perturbation (10% of xavier scale)
                seed = seeds[i % len(seeds)]
                ca_pattern = grid_ca_develop(
                    p.shape, n_steps=64, seed=seed, target_std=p.std().item() * 0.1
                )
                p.data.add_(ca_pattern.to(p.device))


def ca_step(model, step, grad_dict=None):
    """Called after each optimizer step. Default: no-op.
    Agent adds live CA rules here."""
    pass

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_diagnostics(model, step, is_init=False, ca_deltas=None, grad_deltas=None):
    """Print diagnostic metrics. All linalg ops on CPU for MPS safety.

    Args:
        model: GPT model.
        step: Current training step.
        is_init: If True, print init-quality metrics (call once at step 0).
        ca_deltas: Dict of {name: delta_tensor} from CA step, or None.
        grad_deltas: Dict of {name: grad_tensor} from optimizer, or None.
    """
    if is_init:
        # Weight std across all non-embedding 2D params
        stds = []
        for name, p in model.named_parameters():
            if p.dim() >= 2 and "wte" not in name and "lm_head" not in name:
                stds.append(p.data.cpu().float().std().item())
        init_weight_std = sum(stds) / len(stds) if stds else 0.0
        print(f"init_weight_std: {init_weight_std:.6f}")

        # Head diversity: mean pairwise cosine distance of Q-projection weights
        q_vecs = []
        for block in model.blocks:
            w = block.attn.c_q.weight.data.cpu().float()
            hd = w.shape[0] // block.attn.n_head
            for h in range(block.attn.n_head):
                q_vecs.append(w[h * hd : (h + 1) * hd].flatten())
        if len(q_vecs) >= 2:
            dists = []
            for i in range(len(q_vecs)):
                for j in range(i + 1, len(q_vecs)):
                    cos = torch.cosine_similarity(
                        q_vecs[i].unsqueeze(0), q_vecs[j].unsqueeze(0)
                    ).item()
                    dists.append(1.0 - cos)
            print(f"init_head_diversity: {sum(dists) / len(dists):.6f}")

        # Block-diagonal ratio for attention weights
        ratios = []
        for block in model.blocks:
            for attr in ["c_q", "c_k", "c_v"]:
                w = getattr(block.attn, attr).weight.data.cpu().float()
                rows, cols = w.shape
                nb = min(4, min(rows, cols))
                bh, bw = rows // nb, cols // nb
                diag_e = sum(
                    w[b * bh : (b + 1) * bh, b * bw : (b + 1) * bw].pow(2).sum().item()
                    for b in range(nb)
                )
                total_e = w.pow(2).sum().item()
                if total_e > 1e-12:
                    ratios.append(diag_e / total_e)
        if ratios:
            print(f"init_block_diag_ratio: {sum(ratios) / len(ratios):.6f}")

        # Layer similarity: cosine sim between adjacent layers
        # Use only the core block params (attn + mlp) to ensure equal sizes
        layer_vecs = []
        for block in model.blocks:
            parts = []
            for name, p in block.named_parameters():
                if "ve_gate" not in name:  # skip variable-size params
                    parts.append(p.data.cpu().float().flatten())
            layer_vecs.append(torch.cat(parts))
        if len(layer_vecs) >= 2:
            # Trim to minimum size across layers
            min_len = min(v.shape[0] for v in layer_vecs)
            sims = []
            for i in range(len(layer_vecs) - 1):
                s = torch.cosine_similarity(
                    layer_vecs[i][:min_len].unsqueeze(0),
                    layer_vecs[i + 1][:min_len].unsqueeze(0),
                ).item()
                sims.append(s)
            print(f"init_layer_similarity: {sum(sims) / len(sims):.6f}")

    # Live CA metrics (print at eval intervals when CA is active)
    if ca_deltas is not None:
        ca_norm = sum(d.norm().item() ** 2 for d in ca_deltas.values()) ** 0.5
        print(f"ca_delta_norm: {ca_norm:.6f}")

        if grad_deltas is not None:
            grad_norm = sum(d.norm().item() ** 2 for d in grad_deltas.values()) ** 0.5
            print(f"grad_delta_norm: {grad_norm:.6f}")

            # Cosine alignment between CA and gradient updates
            ca_flat = torch.cat([ca_deltas[k].cpu().flatten() for k in sorted(ca_deltas)])
            grad_flat = torch.cat([grad_deltas[k].cpu().flatten() for k in sorted(grad_deltas)])
            if ca_flat.shape == grad_flat.shape:
                alignment = torch.cosine_similarity(
                    ca_flat.unsqueeze(0), grad_flat.unsqueeze(0)
                ).item()
                print(f"ca_grad_alignment: {alignment:.6f}")

            # Contribution ratio
            total = ca_norm + grad_norm
            if total > 1e-12:
                print(f"ca_contribution_ratio: {ca_norm / total:.6f}")

        # Weight sparsity
        n_near_zero = 0
        n_total = 0
        for name, p in model.named_parameters():
            if p.dim() >= 2 and "wte" not in name and "lm_head" not in name:
                n_near_zero += (p.data.abs() < 1e-4).sum().item()
                n_total += p.numel()
        if n_total > 0:
            print(f"weight_sparsity: {n_near_zero / n_total:.6f}")

# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(time_budget: float | None = None):
    """Run training loop.

    Args:
        time_budget: Training time in seconds. Defaults to TIME_BUDGET from prepare.py.
    """
    if time_budget is None:
        time_budget = TIME_BUDGET

    print(f"device: {DEVICE}")
    print(f"time_budget: {time_budget:.0f}s ({time_budget/60:.1f} min)")

    train_data = load_data("train")
    val_data = load_data("val")
    block_size = MAX_SEQ_LEN

    # Create model
    model = GPT(VOCAB_SIZE, block_size, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS).to(DEVICE)
    print(f"params: {model.count_parameters():,}")

    # Apply custom init (agent modifies this)
    initialize_weights(model)

    # Measure init quality
    init_bpb = evaluate_val_bpb(model, val_data, BATCH_SIZE, block_size, DEVICE)
    x0, y0 = get_batch(train_data, BATCH_SIZE, block_size, DEVICE)
    _, init_loss_val = model(x0, y0)
    print(f"init_loss: {init_loss_val.item():.4f}")
    print(f"init_bpb: {init_bpb:.4f}")
    print_diagnostics(model, 0, is_init=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Training loop with periodic val_bpb evaluation
    model.train()
    step = 0
    warmup = 100
    max_steps = 100_000
    min_lr = LR / 10
    t0 = time.time()

    # Eval interval: ~10 checkpoints over the training run
    eval_interval = max(50, int(time_budget / 0.4 / 10))  # ~0.4s per step

    while True:
        elapsed = time.time() - t0
        if elapsed >= time_budget:
            break

        x, y = get_batch(train_data, BATCH_SIZE, block_size, DEVICE)

        # LR schedule
        lr = get_lr(step, warmup, max_steps, LR, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Live CA hook (agent injects CA rules here)
        ca_step(model, step)

        if step % eval_interval == 0:
            # Periodic val_bpb for convergence tracking
            val_bpb_ckpt = evaluate_val_bpb(model, val_data, BATCH_SIZE, block_size, DEVICE)
            elapsed = time.time() - t0
            print(f"step: {step}  train_loss: {loss.item():.4f}  val_bpb: {val_bpb_ckpt:.4f}  elapsed_s: {elapsed:.1f}")
        elif step % 100 == 0:
            elapsed = time.time() - t0
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.1f}s")

        step += 1

    training_time = time.time() - t0

    # Evaluate
    val_bpb = evaluate_val_bpb(model, val_data, BATCH_SIZE, block_size, DEVICE)
    total_time = time.time() - t0
    peak_mem = get_peak_memory_mb()

    # Sample
    model.eval()
    with torch.no_grad():
        ids = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)  # start with null byte
        for _ in range(200):
            logits, _ = model(ids[:, -block_size:])
            probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
            try:
                nxt = torch.multinomial(probs, 1)
            except RuntimeError:
                nxt = torch.multinomial(probs.cpu(), 1).to(DEVICE)
            ids = torch.cat([ids, nxt], 1)
        sample_bytes = bytes(ids[0].tolist())
        sample_text = sample_bytes.decode("utf-8", errors="replace")
    print(f"\n--- Sample ---\n{sample_text}\n--- End ---\n")

    # Print results (grep-friendly format)
    print(f"val_bpb: {val_bpb:.4f}")
    print(f"init_loss: {init_loss_val.item():.4f}")
    print(f"final_train_loss: {loss.item():.4f}")
    print(f"total_steps: {step}")
    print(f"params: {model.count_parameters()}")
    print(f"peak_memory_mb: {peak_mem:.0f}")
    print(f"wall_time_s: {total_time:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroGen training")
    parser.add_argument(
        "--minutes", type=float, default=None,
        help="Training time in minutes (default: use TIME_BUDGET from prepare.py)",
    )
    args = parser.parse_args()
    budget = args.minutes * 60 if args.minutes is not None else None
    train(time_budget=budget)
