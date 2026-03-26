"""
NeuroGen 125M Parameter Validation on H100.

Scales the 3.4M findings to 125M parameters to test whether developmental
attention windows generalize to production-scale models.

Usage (on Lambda AI H100):
    # Setup
    pip install torch numpy datasets
    python prepare_125m.py  # download & shard FineWeb-Edu

    # Throughput audit
    python train_125m.py --throughput

    # Single run
    python train_125m.py --arch baseline --steps 50000 --seed 42

    # Full validation (4 configs × 5 seeds)
    python train_125m.py --tier1

    # Analysis
    python analyze_125m.py
"""

import argparse, time, math, json, sys, os
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Enable TF32 tensor cores for ~2x faster fp32 matmuls on Ampere+
torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------------
# Model Configuration — 125M parameters
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    vocab_size: int = 50257      # GPT-2 tokenizer
    max_seq_len: int = 1024      # standard context
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    # Window config
    window_mode: str = "none"    # "none", "power_2.0", "power_4.0", etc.
    # Induction prewire
    induction_prewire: bool = False

    @property
    def head_dim(self):
        return self.n_embd // self.n_head

# Architecture presets
# At 12 layers (vs 4), optimal exponent likely shifts higher:
# - 4 layers: exponent 4.0 best (3/4 layers local)
# - 12 layers: exponent 6-10 might be best (keep 9-11/12 layers local)
# Hypothesis: optimal_exponent ≈ n_layers (keep all but last ~1-2 layers local)
CONFIGS = {
    "baseline": ModelConfig(),
    "window_quadratic": ModelConfig(window_mode="power_2.0"),
    "window_power_3.0": ModelConfig(window_mode="power_3.0"),
    "window_power_4.0": ModelConfig(window_mode="power_4.0"),
    "window_power_6.0": ModelConfig(window_mode="power_6.0"),
    "window_power_8.0": ModelConfig(window_mode="power_8.0"),
    "window_power_10.0": ModelConfig(window_mode="power_10.0"),
    "window_power_12.0": ModelConfig(window_mode="power_12.0"),
    "window_quad_induction": ModelConfig(window_mode="power_2.0", induction_prewire=True),
    "window_p4_induction": ModelConfig(window_mode="power_4.0", induction_prewire=True),
}

# Training hyperparams for 125M
BATCH_SIZE = 32                # micro batch
GRAD_ACCUM = 4                 # effective batch = 128
MAX_LR = 6e-4                  # standard GPT-2 small LR
MIN_LR = 6e-5
WARMUP_STEPS = 2000
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
GRAD_CLIP = 1.0
EVAL_INTERVAL = 500
EVAL_TOKENS = 500_000          # 500k tokens for stable eval

# ---------------------------------------------------------------------------
# Window mask computation (same logic as train_r4.py, adapted for 1024 ctx)
# ---------------------------------------------------------------------------
def compute_window_mask(T: int, layer_idx: int, n_layer: int,
                        mode: str, device: torch.device) -> torch.Tensor | None:
    """Causal mask with layer-dependent attention window."""
    if mode == "none":
        return None
    progress = (layer_idx + 1) / n_layer
    base = 16  # minimum window (scaled up from 8 for 1024 ctx)
    seq_len = T
    if mode.startswith("power_"):
        exp = float(mode.split("_")[1])
        window = max(base, int(base + progress ** exp * (seq_len - base)))
    else:
        return None
    window = min(window, seq_len)
    rows = torch.arange(T, device=device).unsqueeze(1)
    cols = torch.arange(T, device=device).unsqueeze(0)
    mask = (cols <= rows) & (cols >= rows - window + 1)
    return mask

# ---------------------------------------------------------------------------
# Model (clean GPT-2 architecture)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, :, None, :], persistent=False)

    def forward(self, T):
        return self.cos[:, :T], self.sin[:, :T]

def apply_rotary(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, -x2 * sin + x1 * cos], dim=-1)

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

def _compute_window_size(layer_idx: int, n_layer: int, seq_len: int, mode: str) -> int | None:
    """Compute sliding window size for a layer. Returns None for full attention."""
    if mode == "none":
        return None
    progress = (layer_idx + 1) / n_layer
    base = 16
    if mode.startswith("power_"):
        exp = float(mode.split("_")[1])
        window = max(base, int(base + progress ** exp * (seq_len - base)))
    else:
        return None
    return min(window, seq_len) if window < seq_len else None

class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.layer_idx = layer_idx
        self.n_layer = config.n_layer
        self.window_mode = config.window_mode
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Compute window size for this layer (None = full attention)
        self.window_size = _compute_window_size(
            layer_idx, config.n_layer, config.max_seq_len, config.window_mode)

        # Fallback: pre-compute additive bias for SDPA when flash_attn unavailable
        if self.window_size is not None and not HAS_FLASH_ATTN:
            mask = compute_window_mask(config.max_seq_len, layer_idx, config.n_layer,
                                        self.window_mode, torch.device("cpu"))
            if mask is not None:
                bias = torch.zeros(config.max_seq_len, config.max_seq_len)
                bias.masked_fill_(~mask, float("-inf"))
                causal = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool))
                bias.masked_fill_(~causal, float("-inf"))
                self.register_buffer("attn_bias", bias, persistent=False)
            else:
                self.attn_bias = None
        else:
            self.attn_bias = None

    def forward(self, x, rope_cos, rope_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        q = apply_rotary(q, rope_cos, rope_sin)
        k = apply_rotary(k, rope_cos, rope_sin)

        if HAS_FLASH_ATTN:
            # flash_attn requires fp16/bf16 — cast explicitly for torch.compile compat
            dtype = q.dtype
            if dtype not in (torch.float16, torch.bfloat16):
                q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
            if self.window_size is not None:
                y = flash_attn_func(q, k, v, causal=True,
                                    window_size=(self.window_size - 1, 0))
            else:
                y = flash_attn_func(q, k, v, causal=True)
            if dtype not in (torch.float16, torch.bfloat16):
                y = y.to(dtype)
            return self.c_proj(y.contiguous().view(B, T, C))

        # SDPA fallback (needs B, H, T, D layout)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.attn_bias is not None:
            y = F.scaled_dot_product_attention(q, k, v,
                attn_mask=self.attn_bias[:T, :T])
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = Attention(config, layer_idx)
        self.ln2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.ln1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT125M(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len)
        self.blocks = nn.ModuleList([
            Block(config, i) for i in range(config.n_layer)
        ])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.wte.weight = self.lm_head.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)
        # Scale residual projections
        for block in self.blocks:
            nn.init.normal_(block.attn.c_proj.weight, 0, 0.02 / math.sqrt(2 * self.config.n_layer))
            nn.init.normal_(block.mlp.c_proj.weight, 0, 0.02 / math.sqrt(2 * self.config.n_layer))

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.wte(idx)
        rope_cos, rope_sin = self.rope(T)
        for block in self.blocks:
            x = block(x, rope_cos, rope_sin)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def apply_induction_prewire(self):
        """Pre-wire induction circuit in layers 0-1, head 0."""
        if self.config.n_layer < 2:
            return
        scale = 0.03
        hd = self.config.head_dim
        with torch.no_grad():
            # Layer 0: V/O near-identity for head 0
            b0 = self.blocks[0]
            for i in range(min(hd, self.config.n_embd)):
                b0.attn.c_v.weight.data[i, i] += scale
                b0.attn.c_proj.weight.data[i, i] += scale
            # Layer 1: Q/K near-identity + V/O near-identity for head 0
            b1 = self.blocks[1]
            for i in range(min(hd, self.config.n_embd)):
                b1.attn.c_q.weight.data[i, i] += scale
                b1.attn.c_k.weight.data[i, i] += scale
                b1.attn.c_v.weight.data[i, i] += scale
                b1.attn.c_proj.weight.data[i, i] += scale

# ---------------------------------------------------------------------------
# Data loading (FineWeb-Edu, GPT-2 tokenizer)
# ---------------------------------------------------------------------------
DATA_DIR = Path("data_fineweb")

def prepare_data():
    """Download and shard FineWeb-Edu with GPT-2 tokenizer."""
    import numpy as np
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        print("Install tiktoken: pip install tiktoken")
        sys.exit(1)

    DATA_DIR.mkdir(exist_ok=True)
    if (DATA_DIR / "train_000.bin").exists():
        print(f"Data already prepared in {DATA_DIR}")
        return

    print("Downloading FineWeb-Edu sample...")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                       split="train", streaming=True)

    # Collect ~100M tokens for train, ~5M for val
    for split, target_tokens in [("train", 100_000_000), ("val", 5_000_000)]:
        print(f"Tokenizing {split} ({target_tokens/1e6:.0f}M tokens)...")
        all_tokens = []
        total = 0
        for example in ds:
            tokens = enc.encode_ordinary(example["text"])
            all_tokens.extend(tokens)
            total += len(tokens)
            if total >= target_tokens:
                break
            if total % 10_000_000 < len(tokens):
                print(f"  {total/1e6:.0f}M tokens...")

        tokens_np = np.array(all_tokens[:target_tokens], dtype=np.uint16)
        shard_size = 5_000_000
        n_shards = max(1, len(tokens_np) // shard_size)
        for i in range(n_shards):
            start = i * shard_size
            end = min(start + shard_size, len(tokens_np))
            path = DATA_DIR / f"{split}_{i:03d}.bin"
            tokens_np[start:end].tofile(path)
        print(f"  {split}: {len(tokens_np):,} tokens in {n_shards} shards")

def load_data(split="train"):
    """Load tokenized data from shards."""
    import numpy as np
    shards = sorted(DATA_DIR.glob(f"{split}_*.bin"))
    if not shards:
        raise FileNotFoundError(f"No {split} shards in {DATA_DIR}. Run: python train_125m.py --prepare")
    arrays = [np.fromfile(p, dtype=np.uint16) for p in shards]
    data = torch.tensor(np.concatenate(arrays), dtype=torch.long)
    if torch.cuda.is_available():
        data = data.pin_memory()
    return data

def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    # Vectorized indexing — avoids Python loop over batch
    offsets = torch.arange(seq_len, device=ix.device).unsqueeze(0)
    indices = ix.unsqueeze(1) + offsets  # [batch_size, seq_len]
    x = data[indices].to(device, non_blocking=True)
    y = data[indices + 1].to(device, non_blocking=True)
    return x, y

@torch.no_grad()
def evaluate(model, val_data, batch_size, seq_len, device, n_tokens=EVAL_TOKENS):
    model.eval()
    n_batches = max(1, n_tokens // (batch_size * seq_len))
    total_loss = 0.0
    for _ in range(n_batches):
        x, y = get_batch(val_data, batch_size, seq_len, device)
        _, loss = model(x, y)
        total_loss += loss.item()
    model.train()
    return total_loss / n_batches

# ---------------------------------------------------------------------------
# LR schedule (cosine with warmup)
# ---------------------------------------------------------------------------
def get_lr(step, warmup, max_steps, resume_step=0):
    if resume_step > 0:
        # For resumed runs: cosine from MIN_LR*3 down to MIN_LR over remaining steps
        # Small warmup to let optimizer rebuild state
        resume_warmup = min(500, (max_steps - resume_step) // 10)
        if step < resume_step + resume_warmup:
            progress = (step - resume_step) / resume_warmup
            return MIN_LR + progress * MIN_LR * 2  # ramp up to 3x MIN_LR
        progress = (step - resume_step - resume_warmup) / (max_steps - resume_step - resume_warmup)
        return MIN_LR + 0.5 * (MIN_LR * 2) * (1 + math.cos(math.pi * progress))
    if step < warmup:
        return MAX_LR * (step + 1) / warmup
    if step >= max_steps:
        return MIN_LR
    progress = (step - warmup) / (max_steps - warmup)
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * progress))

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(arch: str, max_steps: int = 50000, seed: int = 42,
          eval_interval: int = EVAL_INTERVAL, resume: str | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "This script requires CUDA (H100)"

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    config = CONFIGS[arch]
    model = GPT125M(config).to(device)

    start_step = 0
    results = []

    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        # Strip _orig_mod. prefix from torch.compile'd checkpoints
        state_dict = ckpt["model_state_dict"]
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        start_step = ckpt["summary"].get("max_steps", 0)
        # Load previous curve if available
        ckpt_results_path = Path("results_125m") / f"{arch}_s{seed}.json"
        if ckpt_results_path.exists():
            prev = json.load(open(ckpt_results_path))
            results = prev.get("curve", [])
        print(f"Resuming from {resume} at step {start_step}")

    if config.induction_prewire and not resume:
        model.apply_induction_prewire()

    # Compile for H100
    if hasattr(torch, "compile"):
        model = torch.compile(model)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"arch: {arch}")
    print(f"params: {params:,} ({params/1e6:.1f}M)")
    print(f"window: {config.window_mode}")
    print(f"induction_prewire: {config.induction_prewire}")
    print(f"seed: {seed}")
    print(f"max_steps: {max_steps}")
    print(f"start_step: {start_step}")
    print(f"effective_batch: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"device: {device} ({torch.cuda.get_device_name()})")

    train_data = load_data("train")
    val_data = load_data("val")
    seq_len = config.max_seq_len

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY,
        betas=(BETA1, BETA2), fused=True
    )
    scaler = torch.amp.GradScaler("cuda")

    if resume:
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

    t0 = time.time()

    for step in range(start_step, max_steps + 1):
        # Eval
        if step % eval_interval == 0:
            val_loss = evaluate(model, val_data, BATCH_SIZE, seq_len, device)
            val_bpb = val_loss / math.log(2)
            elapsed = time.time() - t0
            tokens_seen = step * BATCH_SIZE * GRAD_ACCUM * seq_len

            row = {
                "arch": arch, "seed": seed, "step": step,
                "val_loss": round(val_loss, 6),
                "val_bpb": round(val_bpb, 6),
                "elapsed_s": round(elapsed, 1),
                "tokens_seen": tokens_seen,
            }
            results.append(row)
            print(f"step:{step:6d}  val_loss:{val_loss:.4f}  val_bpb:{val_bpb:.4f}  "
                  f"tokens:{tokens_seen/1e6:.0f}M  {elapsed:.0f}s")

        if step >= max_steps:
            break

        # Training step with gradient accumulation
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(GRAD_ACCUM):
            x, y = get_batch(train_data, BATCH_SIZE, seq_len, device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / GRAD_ACCUM
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        lr = get_lr(step, WARMUP_STEPS, max_steps, resume_step=start_step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        scaler.step(optimizer)
        scaler.update()

    total_time = time.time() - t0
    final = results[-1]
    summary = {
        "arch": arch, "seed": seed, "max_steps": max_steps,
        "final_val_loss": final["val_loss"],
        "final_val_bpb": final["val_bpb"],
        "total_time_s": round(total_time, 1),
        "steps_per_sec": round(max_steps / total_time, 2),
        "params": params,
        "tokens_seen": final["tokens_seen"],
    }

    # Save results + checkpoint
    out_dir = Path("results_125m")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{arch}_s{seed}.json"
    with open(path, "w") as f:
        json.dump({"summary": summary, "curve": results}, f, indent=2)

    ckpt_dir = Path("checkpoints_125m")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"{arch}_s{seed}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config.__dict__,
        "summary": summary,
    }, ckpt_path)

    print(f"\nFINAL: val_bpb={final['val_bpb']:.4f}  time={total_time:.0f}s  saved={path}")
    print(f"checkpoint: {ckpt_path}")
    print(f"RESULT_125M: {json.dumps(summary)}")
    return summary

# ---------------------------------------------------------------------------
# Throughput audit
# ---------------------------------------------------------------------------
def throughput_audit():
    device = "cuda"
    print("=== Throughput Audit (200 steps each) ===")
    for arch_name in ["baseline", "window_quadratic", "window_power_4.0",
                       "window_power_8.0", "window_power_12.0", "window_quad_induction"]:
        config = CONFIGS[arch_name]
        torch.manual_seed(42)
        model = GPT125M(config).to(device)
        if hasattr(torch, "compile"):
            model = torch.compile(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, fused=True)
        scaler = torch.amp.GradScaler("cuda")
        train_data = load_data("train")

        # Warmup
        for _ in range(10):
            x, y = get_batch(train_data, BATCH_SIZE, config.max_seq_len, device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t0 = time.time()
        for s in range(200):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(GRAD_ACCUM):
                x, y = get_batch(train_data, BATCH_SIZE, config.max_seq_len, device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                    loss = loss / GRAD_ACCUM
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        sps = 200 / elapsed
        tps = 200 * BATCH_SIZE * GRAD_ACCUM * config.max_seq_len / elapsed
        print(f"  {arch_name:<28} {sps:.1f} steps/s  {tps/1e6:.1f}M tok/s  {elapsed:.1f}s")
        del model, optimizer

# ---------------------------------------------------------------------------
# Tier 1: Full validation
# ---------------------------------------------------------------------------
def run_tier1(max_steps=20000):
    configs = ["baseline", "window_power_4.0", "window_power_8.0",
               "window_quadratic", "window_quad_induction"]
    seeds = [42, 137, 256, 789, 1337]
    print(f"=== 125M Validation Tier 1 ({max_steps} steps) ===")
    print(f"Configs: {configs}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {len(configs) * len(seeds)}")
    for arch in configs:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  {arch} seed={seed}")
            print(f"{'='*60}")
            train(arch=arch, max_steps=max_steps, seed=seed)

# ---------------------------------------------------------------------------
# Inference / Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(model, prompt_tokens: list[int], max_new_tokens: int = 200,
             temperature: float = 0.8, top_k: int = 50) -> list[int]:
    """Autoregressive generation from a trained model."""
    model.eval()
    device = next(model.parameters()).device
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx[0].tolist()

def run_inference(ckpt_path: str, prompts: list[str] | None = None,
                  max_tokens: int = 200, temperature: float = 0.8):
    """Load checkpoint and generate text."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ModelConfig(**ckpt["config"])
    model = GPT125M(config).to(device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd)
    model.eval()

    summary = ckpt.get("summary", {})
    print(f"Loaded: {summary.get('arch', '?')} seed={summary.get('seed', '?')} "
          f"bpb={summary.get('final_val_bpb', '?')}")

    if prompts is None:
        prompts = [
            "The meaning of life is",
            "In a distant galaxy,",
            "The scientist discovered that",
            "Once upon a time,",
        ]

    for prompt in prompts:
        tokens = enc.encode(prompt)
        out = generate(model, tokens, max_new_tokens=max_tokens, temperature=temperature)
        text = enc.decode(out)
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*60}")
        print(text)
    print()

def compare_models(ckpt_paths: list[str], prompts: list[str] | None = None,
                   max_tokens: int = 200, temperature: float = 0.8, seed: int = 42):
    """Side-by-side generation from multiple checkpoints with same RNG seed."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if prompts is None:
        prompts = [
            "The meaning of life is",
            "In a distant galaxy,",
            "The scientist discovered that",
            "Once upon a time,",
        ]

    models = []
    for path in ckpt_paths:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = ModelConfig(**ckpt["config"])
        model = GPT125M(config).to(device)
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(sd)
        model.eval()
        summary = ckpt.get("summary", {})
        models.append((path, summary, model))
        print(f"Loaded: {summary.get('arch', '?')} seed={summary.get('seed', '?')} "
              f"bpb={summary.get('final_val_bpb', '?')}")

    for prompt in prompts:
        tokens = enc.encode(prompt)
        print(f"\n{'='*70}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*70}")
        for path, summary, model in models:
            torch.manual_seed(seed)
            out = generate(model, tokens, max_new_tokens=max_tokens, temperature=temperature)
            text = enc.decode(out)
            arch = summary.get("arch", Path(path).stem)
            print(f"\n--- {arch} ---")
            print(text)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NeuroGen 125M Validation")
    parser.add_argument("--arch", type=str, default="baseline", choices=list(CONFIGS.keys()))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, metavar="CKPT", help="Resume training from checkpoint")
    parser.add_argument("--prepare", action="store_true", help="Download and prepare data")
    parser.add_argument("--throughput", action="store_true", help="Throughput audit")
    parser.add_argument("--tier1", action="store_true", help="Run all Tier 1 experiments")
    parser.add_argument("--generate", type=str, metavar="CKPT", help="Generate from checkpoint")
    parser.add_argument("--compare", nargs="+", metavar="CKPT", help="Compare generations from multiple checkpoints")
    parser.add_argument("--prompt", type=str, help="Custom prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    if args.prepare:
        prepare_data()
    elif args.throughput:
        throughput_audit()
    elif args.tier1:
        run_tier1(max_steps=args.steps)
    elif args.generate:
        prompts = [args.prompt] if args.prompt else None
        run_inference(args.generate, prompts, args.max_tokens, args.temperature)
    elif args.compare:
        prompts = [args.prompt] if args.prompt else None
        compare_models(args.compare, prompts, args.max_tokens, args.temperature)
    else:
        train(arch=args.arch, max_steps=args.steps, seed=args.seed, resume=args.resume)

if __name__ == "__main__":
    main()
