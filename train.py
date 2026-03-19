"""
NeuroGen train.py — model + training loop + CA hooks.

The AI researcher modifies this file to test CA initialization and live CA rules.
Self-contained: does NOT import from any local package except prepare.py and ca_rules.py.

Usage:
    uv run train.py                              # default 2 min (autoresearch speed)
    uv run train.py --minutes 10                  # medium validation
    uv run train.py --minutes 30 --seed 42        # long validation, fixed seed
    uv run train.py --init xavier --minutes 10    # pure xavier init
    uv run train.py --init xavier_ca10 --seed 42  # xavier + 10% grid CA
"""

import argparse
import time
import math
import json
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    load_data, get_batch, evaluate_val_bpb, get_device, get_peak_memory_mb,
    VOCAB_SIZE, MAX_SEQ_LEN, TIME_BUDGET,
)
from ca_rules import (
    grid_ca_develop, reaction_diffusion_init, modular_init, block_diagonal_init,
    rescale,
)

# ---------------------------------------------------------------------------
# Hyperparameters (Round 1 best: depth 2, LR 3.5e-3, batch 32, WD 0.05)
# ---------------------------------------------------------------------------

DEPTH = 2                     # number of transformer layers
CHANNELS = DEPTH * 64         # embedding dimension (128 for depth 2)
N_HEADS = DEPTH               # number of attention heads (2 for depth 2)
N_KV_HEADS = N_HEADS          # KV heads (set < N_HEADS for GQA)
BATCH_SIZE = 32
LR = 3.5e-3
WEIGHT_DECAY = 0.05
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
# CA hooks — configurable initialization
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


def _spectral_ca_pattern(shape: tuple[int, int], n_modes: int = 8,
                          target_std: float = 0.02) -> torch.Tensor:
    """Generate weight pattern via spectral (Fourier) CA.
    Creates smooth multi-scale structure by composing random Fourier modes."""
    h, w = shape
    pattern = torch.zeros(h, w)
    for _ in range(n_modes):
        freq_h = torch.randint(1, max(2, h // 4), (1,)).item()
        freq_w = torch.randint(1, max(2, w // 4), (1,)).item()
        phase_h = torch.rand(1).item() * 2 * math.pi
        phase_w = torch.rand(1).item() * 2 * math.pi
        amp = torch.randn(1).item()
        rows = torch.sin(torch.linspace(0, freq_h * 2 * math.pi, h) + phase_h)
        cols = torch.sin(torch.linspace(0, freq_w * 2 * math.pi, w) + phase_w)
        pattern += amp * rows.unsqueeze(1) * cols.unsqueeze(0)
    return rescale(pattern, target_std)


def initialize_weights(model, init_method: str = "xavier_ca10"):
    """Apply initialization based on method name.

    Supported methods:
        xavier          — pure Xavier uniform
        xavier_ca5      — Xavier + 5% grid CA perturbation
        xavier_ca10     — Xavier + 10% grid CA perturbation (Round 1 best)
        xavier_ca15     — Xavier + 15% grid CA perturbation
        xavier_ca20     — Xavier + 20% grid CA perturbation
        xavier_ca30     — Xavier + 30% grid CA perturbation
        xavier_grid_ca  — Xavier + grid CA at best blend (=xavier_ca10)
        xavier_rd_spots — Xavier + reaction-diffusion spots (feed=0.03, kill=0.06)
        xavier_rd_stripes — Xavier + reaction-diffusion stripes (feed=0.04, kill=0.06)
        xavier_block_ca — Xavier + block-diagonal CA
        xavier_spectral_ca — Xavier + spectral (Fourier) CA
    """
    # Parse blend ratio from name
    blend = 0.0
    ca_fn = None

    if init_method == "xavier":
        blend = 0.0
    elif init_method.startswith("xavier_ca"):
        pct = init_method.replace("xavier_ca", "")
        blend = int(pct) / 100.0
        ca_fn = "grid"
    elif init_method == "xavier_grid_ca":
        blend = 0.10
        ca_fn = "grid"
    elif init_method == "xavier_rd_spots":
        blend = 0.10
        ca_fn = "rd_spots"
    elif init_method == "xavier_rd_stripes":
        blend = 0.10
        ca_fn = "rd_stripes"
    elif init_method == "xavier_block_ca":
        blend = 0.10
        ca_fn = "block"
    elif init_method == "xavier_spectral_ca":
        blend = 0.10
        ca_fn = "spectral"
    else:
        print(f"WARNING: Unknown init method '{init_method}', using xavier_ca10")
        blend = 0.10
        ca_fn = "grid"

    seeds = ["center", "diagonal", "distributed", "random"]
    with torch.no_grad():
        for i, (name, p) in enumerate(model.named_parameters()):
            if _is_ca_target(name, p):
                # Xavier base
                nn.init.xavier_uniform_(p)
                if blend > 0 and ca_fn is not None:
                    xavier_std = p.std().item()
                    if ca_fn == "grid":
                        seed = seeds[i % len(seeds)]
                        ca_pattern = grid_ca_develop(
                            p.shape, n_steps=64, seed=seed,
                            target_std=xavier_std * blend
                        )
                    elif ca_fn == "rd_spots":
                        ca_pattern = reaction_diffusion_init(
                            p.shape, feed=0.03, kill=0.06, n_steps=200,
                            target_std=xavier_std * blend
                        )
                    elif ca_fn == "rd_stripes":
                        ca_pattern = reaction_diffusion_init(
                            p.shape, feed=0.04, kill=0.06, n_steps=200,
                            target_std=xavier_std * blend
                        )
                    elif ca_fn == "block":
                        ca_pattern = block_diagonal_init(
                            p.shape, n_blocks=min(4, min(p.shape)),
                            target_std=xavier_std * blend
                        )
                    elif ca_fn == "spectral":
                        ca_pattern = _spectral_ca_pattern(
                            p.shape, target_std=xavier_std * blend
                        )
                    else:
                        continue
                    p.data.add_(ca_pattern.to(p.device))


def ca_step(model, step, grad_dict=None):
    """Called after each optimizer step. Default: no-op.
    Agent adds live CA rules here."""
    pass

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
# Text generation
# ---------------------------------------------------------------------------

PROMPTS = [
    "Once upon a time, there was a little",
    "The cat sat on the mat and looked at",
    "\"Can you help me?\" asked the",
    "It was a beautiful sunny day. The children",
    "The most important thing about being kind is",
    "There was a little bear who lived in the forest. Every morning, he would wake up and",
    "Sarah was sad because she lost her",
    "The dog ran fast because",
    "One day, a bird flew into the",
    "Mom said, \"It's time to",
]


def generate(model, prompt_bytes: bytes, max_tokens: int = 100,
             temperature: float = 0.8, block_size: int = 256) -> str:
    """Generate text from a byte-level prompt."""
    model.eval()
    ids = torch.tensor([list(prompt_bytes)], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for _ in range(max_tokens):
            logits, _ = model(ids[:, -block_size:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            try:
                nxt = torch.multinomial(probs, 1)
            except RuntimeError:
                nxt = torch.multinomial(probs.cpu(), 1).to(DEVICE)
            ids = torch.cat([ids, nxt], 1)
    return bytes(ids[0].tolist()).decode("utf-8", errors="replace")


def compute_text_quality(texts: list[str]) -> dict:
    """Compute text quality metrics over a list of generated texts."""
    all_tokens = []
    n_repetitions = 0
    n_trigrams = 0
    n_complete = 0

    for text in texts:
        tokens = text.split()
        all_tokens.extend(tokens)
        # 3-gram repetition
        for j in range(len(tokens) - 2):
            trigram = (tokens[j], tokens[j+1], tokens[j+2])
            n_trigrams += 1
            # Check if this trigram appears again
            for k in range(j + 1, len(tokens) - 2):
                if (tokens[k], tokens[k+1], tokens[k+2]) == trigram:
                    n_repetitions += 1
                    break
        # Sentence completion (ends with . ! ? or similar)
        stripped = text.strip()
        if stripped and stripped[-1] in '.!?"\'':
            n_complete += 1

    unique_ratio = len(set(all_tokens)) / max(len(all_tokens), 1)
    repetition_rate = n_repetitions / max(n_trigrams, 1)
    completion_rate = n_complete / max(len(texts), 1)

    return {
        "unique_token_ratio": unique_ratio,
        "trigram_repetition_rate": repetition_rate,
        "sentence_completion_rate": completion_rate,
        "total_tokens": len(all_tokens),
    }

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(time_budget: float | None = None, seed: int | None = None,
          init_method: str = "xavier_ca10", quiet: bool = False):
    """Run training loop.

    Args:
        time_budget: Training time in seconds. Defaults to TIME_BUDGET from prepare.py.
        seed: Random seed for reproducibility. None = no seeding.
        init_method: Initialization method name (see initialize_weights).
        quiet: If True, suppress per-step output (only emit JSON result).

    Returns:
        Dict with all results and loss curve data.
    """
    if time_budget is None:
        time_budget = TIME_BUDGET

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if not quiet:
        print(f"device: {DEVICE}")
        print(f"time_budget: {time_budget:.0f}s ({time_budget/60:.1f} min)")
        print(f"init_method: {init_method}")
        if seed is not None:
            print(f"seed: {seed}")

    train_data = load_data("train")
    val_data = load_data("val")
    block_size = MAX_SEQ_LEN

    # Create model
    model = GPT(VOCAB_SIZE, block_size, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS).to(DEVICE)
    if not quiet:
        print(f"params: {model.count_parameters():,}")

    # Apply custom init
    initialize_weights(model, init_method)

    # Measure init quality
    init_bpb = evaluate_val_bpb(model, val_data, BATCH_SIZE, block_size, DEVICE)
    x0, y0 = get_batch(train_data, BATCH_SIZE, block_size, DEVICE)
    _, init_loss_val = model(x0, y0)
    if not quiet:
        print(f"init_loss: {init_loss_val.item():.4f}")
        print(f"init_bpb: {init_bpb:.4f}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Training loop with periodic val_bpb evaluation
    model.train()
    step = 0
    warmup = 200
    max_steps = 100_000
    min_lr = LR / 10
    t0 = time.time()

    # Loss curve: record val_bpb at intervals
    # Every 50 steps or 30s, whichever is less frequent
    curve_step_interval = 50
    curve_time_interval = 30.0
    last_curve_time = 0.0
    loss_curve = []  # list of (step, elapsed_s, val_bpb)

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

        # Live CA hook
        ca_step(model, step)

        # Loss curve checkpoint
        elapsed = time.time() - t0
        should_record = (
            step > 0
            and step % curve_step_interval == 0
            and (elapsed - last_curve_time) >= curve_time_interval
        )
        if should_record or step == 0:
            val_bpb_ckpt = evaluate_val_bpb(model, val_data, BATCH_SIZE, block_size, DEVICE)
            loss_curve.append((step, round(elapsed, 1), round(val_bpb_ckpt, 4)))
            last_curve_time = elapsed
            if not quiet:
                print(f"step: {step}  train_loss: {loss.item():.4f}  val_bpb: {val_bpb_ckpt:.4f}  elapsed_s: {elapsed:.1f}")
        elif not quiet and step % 200 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.1f}s")

        step += 1

    # Final evaluation
    val_bpb = evaluate_val_bpb(model, val_data, BATCH_SIZE, block_size, DEVICE)
    total_time = time.time() - t0
    peak_mem = get_peak_memory_mb()

    # Add final point to loss curve
    loss_curve.append((step, round(total_time, 1), round(val_bpb, 4)))

    # Sample (quick, for display)
    model.eval()
    with torch.no_grad():
        ids = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)
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

    if not quiet:
        print(f"\n--- Sample ---\n{sample_text}\n--- End ---\n")

    # Print results (grep-friendly format)
    if not quiet:
        print(f"val_bpb: {val_bpb:.4f}")
        print(f"init_loss: {init_loss_val.item():.4f}")
        print(f"final_train_loss: {loss.item():.4f}")
        print(f"total_steps: {step}")
        print(f"params: {model.count_parameters()}")
        print(f"peak_memory_mb: {peak_mem:.0f}")
        print(f"wall_time_s: {total_time:.1f}")

    result = {
        "val_bpb": round(val_bpb, 4),
        "init_loss": round(init_loss_val.item(), 4),
        "init_bpb": round(init_bpb, 4),
        "final_train_loss": round(loss.item(), 4),
        "total_steps": step,
        "params": model.count_parameters(),
        "peak_memory_mb": round(peak_mem, 0),
        "wall_time_s": round(total_time, 1),
        "init_method": init_method,
        "seed": seed,
        "time_budget_s": time_budget,
        "loss_curve": loss_curve,
    }

    # Also store model for potential Track 3 use
    result["_model"] = model

    # Print JSON result line for programmatic parsing
    result_json = {k: v for k, v in result.items() if k != "_model"}
    print(f"RESULT_JSON: {json.dumps(result_json)}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroGen training")
    parser.add_argument(
        "--minutes", type=float, default=None,
        help="Training time in minutes (default: use TIME_BUDGET from prepare.py)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--init", type=str, default="xavier_ca10",
        help="Initialization method (xavier, xavier_ca10, xavier_rd_spots, etc.)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-step output, only emit JSON result",
    )
    args = parser.parse_args()
    budget = args.minutes * 60 if args.minutes is not None else None
    train(time_budget=budget, seed=args.seed, init_method=args.init, quiet=args.quiet)
