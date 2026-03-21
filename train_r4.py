"""
NeuroGen Round 4: CA Inside the Transformer.

Modular architecture supporting 15+ experimental variants via --arch flag.
Depth 4, Channels 256 (~1.5M params). Default 2h training on M1 Pro.

Usage:
    uv run train_r4.py --arch baseline --minutes 40 --seed 42
    uv run train_r4.py --arch window_quadratic --minutes 120 --seed 42
    uv run train_r4.py --arch ca_modulate_attn --minutes 120 --seed 42
"""

import argparse, time, math, json, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare import (
    load_data, get_batch, evaluate_val_bpb, get_device, get_peak_memory_mb,
    VOCAB_SIZE, MAX_SEQ_LEN, TIME_BUDGET,
)
from ca_rules import grid_ca_develop, rescale, homeostatic_step, competition_step

# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------
ARCHS = {
    "baseline": {},
    "window_linear": {"window": "linear"},
    "window_quadratic": {"window": "quadratic"},
    "window_step": {"window": "step"},
    "attn_evolve": {"attn_evolve": True},
    "attn_bias_layer": {"attn_bias": "per_layer"},
    "attn_bias_head": {"attn_bias": "per_head"},
    "ca_mod_attn": {"ca_mod": "attn"},
    "ca_mod_both": {"ca_mod": "both"},
    "ca_mod_add": {"ca_mod": "additive"},
    "ca_multiscale": {"ca_mod": "multiscale"},
    "token_vitality": {"vitality": True},
    "ca_ffn": {"ca_ffn": True, "ca_ffn_steps": 4},
    "ca_ffn_8": {"ca_ffn": True, "ca_ffn_steps": 8},
    "cross_layer_ca": {"cross_ca": True},
    "dev_dropout": {"dev_dropout": True},
    "sleep": {"sleep": True, "sleep_interval": 500, "sleep_steps": 10},
    "sleep_competition": {"sleep": True, "sleep_interval": 500, "sleep_steps": 10, "sleep_rule": "competition"},
    # Universal circuit pre-wiring (from interpretability research)
    "induction_prewire": {"universal": "induction"},
    "layer_roles": {"universal": "layer_roles"},
    "diverse_heads": {"universal": "diverse_heads"},
    "universal_all": {"universal": "all"},
    # Best from E1 + universal circuits
    "window_quad_induction": {"window": "quadratic", "universal": "induction"},
    "window_quad_universal": {"window": "quadratic", "universal": "all"},
    # Track J: Embryogenic CA — activity-dependent development
    "embryo_strengthen": {"embryo": "strengthen", "embryo_freq": 10, "embryo_crit": 0.2},
    "embryo_hebbian": {"embryo": "hebbian", "embryo_freq": 10, "embryo_crit": 0.2},
    "embryo_strengthen_long": {"embryo": "strengthen", "embryo_freq": 10, "embryo_crit": 0.4},
    "embryo_plus_window": {"embryo": "strengthen", "embryo_freq": 10, "embryo_crit": 0.2, "window": "quadratic"},
    "embryo_plus_induction": {"embryo": "strengthen", "embryo_freq": 10, "embryo_crit": 0.2, "universal": "induction"},
    # Window function search (Phase W1-W2)
    "window_power_0.5": {"window": "power_0.5"},
    "window_power_1.5": {"window": "power_1.5"},
    "window_power_2.5": {"window": "power_2.5"},
    "window_power_3.0": {"window": "power_3.0"},
    "window_power_4.0": {"window": "power_4.0"},
    "window_power_5.0": {"window": "power_5.0"},
    "window_sigmoid_0.3": {"window": "sigmoid_0.3"},
    "window_sigmoid_0.5": {"window": "sigmoid_0.5"},
    "window_sigmoid_0.7": {"window": "sigmoid_0.7"},
    "window_logarithmic": {"window": "logarithmic"},
    "window_exponential": {"window": "exponential"},
    "window_fibonacci": {"window": "fibonacci"},
    # Best window + induction (Phase W4, to be filled after W1-W2)
    "window_best_induction": {"window": "power_3.0", "universal": "induction"},  # placeholder
    # Phase J2: Embryogenic + winning architecture (window_quad_induction)
    "embryo_heb_wqi": {"embryo": "hebbian", "embryo_freq": 10, "embryo_crit": 0.2, "window": "quadratic", "universal": "induction"},
    "embryo_str_long_wqi": {"embryo": "strengthen", "embryo_freq": 10, "embryo_crit": 0.4, "window": "quadratic", "universal": "induction"},
    "embryo_targeted_wqi": {"embryo": "targeted", "embryo_freq": 10, "embryo_crit": 0.3, "window": "quadratic", "universal": "induction"},
    "embryo_long60_wqi": {"embryo": "strengthen", "embryo_freq": 10, "embryo_crit": 0.6, "window": "quadratic", "universal": "induction"},
    # Phase J3: Smarter CA rules on wqi
    "embryo_gradalign_wqi": {"embryo": "gradalign", "embryo_freq": 10, "embryo_crit": 0.2, "window": "quadratic", "universal": "induction"},
    # Phase J5: Long horizon validation
    "wqi_2h": {"window": "quadratic", "universal": "induction"},
}

# ---------------------------------------------------------------------------
# Hyperparameters (Round 4: depth 4, channels 256)
# ---------------------------------------------------------------------------
DEPTH = 4
CHANNELS = 256
N_HEADS = 4
N_KV_HEADS = N_HEADS
BATCH_SIZE = 32
LR = 2e-3          # will tune; depth 4 may need different LR than depth 2
WEIGHT_DECAY = 0.05
WARMUP = 200
DEVICE = get_device()

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def rms_norm(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)

def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

# ---------------------------------------------------------------------------
# Window mask computation
# ---------------------------------------------------------------------------
def compute_window_mask(T: int, layer_idx: int, n_layer: int,
                        mode: str, device: str) -> torch.Tensor:
    """Causal mask with layer-dependent attention window."""
    progress = (layer_idx + 1) / n_layer
    base = 8
    if mode == "linear":
        window = max(base, int(progress * T))
    elif mode == "quadratic":
        window = max(base, int(base + progress ** 2 * (T - base)))
    elif mode == "step":
        window = T // 4 if layer_idx < n_layer // 2 else T
    elif mode.startswith("power_"):
        exp = float(mode.split("_")[1])
        window = max(base, int(base + progress ** exp * (T - base)))
    elif mode.startswith("sigmoid_"):
        mid = float(mode.split("_")[1])
        steepness = 10.0
        s = 1.0 / (1.0 + math.exp(-steepness * (progress - mid)))
        window = max(base, int(base + s * (T - base)))
    elif mode == "logarithmic":
        window = max(base, int(base + math.log(1 + progress * (math.e - 1)) * (T - base)))
    elif mode == "exponential":
        window = max(base, int(base + (math.exp(progress * 3) - 1) / (math.e**3 - 1) * (T - base)))
    elif mode == "fibonacci":
        fibs = [base, base * 2]
        for _ in range(2, n_layer):
            fibs.append(min(fibs[-1] + fibs[-2], T))
        window = fibs[min(layer_idx, len(fibs) - 1)]
    else:
        return None
    window = min(window, T)
    # Build causal + windowed mask
    rows = torch.arange(T, device=device).unsqueeze(1)
    cols = torch.arange(T, device=device).unsqueeze(0)
    mask = (cols <= rows) & (cols >= rows - window + 1)
    return mask.float()  # (T, T)

# ---------------------------------------------------------------------------
# CA Modulation Channel
# ---------------------------------------------------------------------------
class CAChannel(nn.Module):
    """Lightweight parallel CA channel using depthwise conv1d."""
    def __init__(self, n_embd, ca_dim=32):
        super().__init__()
        self.proj_in = nn.Linear(n_embd, ca_dim, bias=False)
        self.conv = nn.Conv1d(ca_dim, ca_dim, kernel_size=3, padding=1,
                              groups=ca_dim, bias=False)
        self.proj_out = nn.Linear(ca_dim, n_embd, bias=False)
        nn.init.zeros_(self.proj_out.weight)
        self.ca_dim = ca_dim

    def forward(self, x):
        # x: (B, T, C) -> CA state -> modulation signal
        h = self.proj_in(x)                           # (B, T, ca_dim)
        h = h + 0.1 * self.conv(h.transpose(1, 2)).transpose(1, 2)  # CA step
        return self.proj_out(F.gelu(h))               # (B, T, C)

class MultiTimescaleCA(nn.Module):
    """Multiple CA channels at different timescales."""
    def __init__(self, n_embd, ca_dim=32):
        super().__init__()
        self.proj_in = nn.Linear(n_embd, ca_dim, bias=False)
        self.fast = nn.Conv1d(ca_dim // 2, ca_dim // 2, 3, padding=1,
                              groups=ca_dim // 2, bias=False)
        self.slow = nn.Conv1d(ca_dim // 2, ca_dim // 2, 7, padding=3,
                              groups=ca_dim // 2, bias=False)
        self.proj_out = nn.Linear(ca_dim, n_embd, bias=False)
        nn.init.zeros_(self.proj_out.weight)

    def forward(self, x, layer_idx=0):
        h = self.proj_in(x)
        fast, slow = h.chunk(2, dim=-1)
        fast = fast + 0.1 * self.fast(fast.transpose(1, 2)).transpose(1, 2)
        if layer_idx % 2 == 0:
            slow = slow + 0.05 * self.slow(slow.transpose(1, 2)).transpose(1, 2)
        return self.proj_out(F.gelu(torch.cat([fast, slow], dim=-1)))

# ---------------------------------------------------------------------------
# CA-based FFN replacement
# ---------------------------------------------------------------------------
class CAFFN(nn.Module):
    """Replace FFN with iterative local CA processing."""
    def __init__(self, n_embd, hidden_mult=4, ca_steps=4):
        super().__init__()
        h = hidden_mult * n_embd
        self.proj_in = nn.Linear(n_embd, h, bias=False)
        self.proj_out = nn.Linear(h, n_embd, bias=False)
        self.ca_conv = nn.Conv1d(h, h, kernel_size=3, padding=1, groups=h, bias=False)
        self.gate = nn.Linear(n_embd, n_embd, bias=False)
        self.ca_steps = ca_steps

    def forward(self, x):
        h = F.relu(self.proj_in(x)).square()
        ht = h.transpose(1, 2)  # (B, H, T) for conv1d
        for _ in range(self.ca_steps):
            ht = ht + 0.1 * self.ca_conv(ht)
        h = ht.transpose(1, 2)
        return self.proj_out(h) * torch.sigmoid(self.gate(x))

# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, n_layer, layer_idx, arch_cfg):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.layer_idx = layer_idx
        self.c_q = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.ve_gate = (
            nn.Linear(min(12, n_embd), n_kv_head, bias=False)
            if has_ve(layer_idx, n_layer) else None
        )
        self._ve_ch = min(12, n_embd)
        # Attention bias from CA patterns
        if arch_cfg.get("attn_bias") == "per_layer":
            bias = grid_ca_develop((MAX_SEQ_LEN, MAX_SEQ_LEN), n_steps=64,
                                   seed=["center","diagonal","distributed","random"][layer_idx % 4],
                                   target_std=0.05)
            bias = torch.tril(bias)
            self.register_buffer("ca_bias", bias.unsqueeze(0).unsqueeze(0))  # (1,1,T,T)
        elif arch_cfg.get("attn_bias") == "per_head":
            biases = []
            for h in range(n_head):
                b = grid_ca_develop((MAX_SEQ_LEN, MAX_SEQ_LEN), n_steps=32,
                                    seed=["center","diagonal","distributed","random"][(layer_idx*n_head+h)%4],
                                    target_std=0.03)
                biases.append(torch.tril(b))
            self.register_buffer("ca_bias", torch.stack(biases).unsqueeze(0))  # (1,H,T,T)
        else:
            self.ca_bias = None
        # Attention evolution
        self._evolve = arch_cfg.get("attn_evolve", False)
        if self._evolve and layer_idx > 0:
            self.refine_conv = nn.Conv2d(n_head, n_head, kernel_size=3, padding=1,
                                         groups=n_head, bias=False)
            self.refine_alpha = nn.Parameter(torch.tensor(0.05))
        else:
            self.refine_conv = None

    def forward(self, x, ve, cos, sin, mask=None, prev_attn=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None and self.ve_gate is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self._ve_ch]))
            v = v + gate.unsqueeze(-1) * ve
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = rms_norm(q) * 1.2, rms_norm(k) * 1.2
        if self.n_kv_head < self.n_head:
            r = self.n_head // self.n_kv_head
            k = k.repeat_interleave(r, dim=2)
            v = v.repeat_interleave(r, dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Compute attention scores
        use_manual = (mask is not None or self.ca_bias is not None
                      or self._evolve or prev_attn is not None)
        if use_manual:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = q @ k.transpose(-2, -1) * scale
            # Apply CA bias
            if self.ca_bias is not None:
                att = att + self.ca_bias[:, :, :T, :T]
            # Apply attention evolution
            if self.refine_conv is not None and prev_attn is not None:
                refined = self.refine_conv(prev_attn[:, :, :T, :T])
                att = att + self.refine_alpha * refined
            # Apply causal + window mask
            if mask is not None:
                att = att.masked_fill(mask[:T, :T] == 0, float("-inf"))
            else:
                cmask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
                att = att.masked_fill(~cmask, float("-inf"))
            attn_weights = F.softmax(att, dim=-1)
            y = attn_weights @ v
            attn_out = attn_weights.detach()  # for evolution / metrics
        else:
            try:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            except RuntimeError:
                scale = 1.0 / math.sqrt(self.head_dim)
                att = q @ k.transpose(-2, -1) * scale
                cmask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
                att = att.masked_fill(~cmask, float("-inf"))
                y = F.softmax(att, dim=-1) @ v
            attn_out = None

        out = self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))
        return out, attn_out

# ---------------------------------------------------------------------------
# MLP (standard)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())

# ---------------------------------------------------------------------------
# Block (with CA modulation hooks)
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, n_layer, layer_idx, arch_cfg):
        super().__init__()
        self.attn = Attention(n_embd, n_head, n_kv_head, n_layer, layer_idx, arch_cfg)
        self.layer_idx = layer_idx
        if arch_cfg.get("ca_ffn"):
            self.mlp = CAFFN(n_embd, ca_steps=arch_cfg.get("ca_ffn_steps", 4))
        else:
            self.mlp = MLP(n_embd)
        # CA modulation
        ca_mode = arch_cfg.get("ca_mod")
        if ca_mode == "multiscale":
            self.ca_ch = MultiTimescaleCA(n_embd)
        elif ca_mode in ("attn", "both", "additive"):
            self.ca_ch = CAChannel(n_embd)
        else:
            self.ca_ch = None
        self.ca_mode = ca_mode
        # Vitality
        if arch_cfg.get("vitality"):
            self.vitality_net = nn.Sequential(
                nn.Linear(n_embd, 32, bias=False), nn.GELU(),
                nn.Linear(32, 1, bias=False), nn.Sigmoid())
        else:
            self.vitality_net = None
        # Dev dropout
        self.dev_dropout = arch_cfg.get("dev_dropout", False)

    def forward(self, x, ve, cos, sin, mask=None, prev_attn=None,
                step=0, total_steps=1):
        # Dev dropout: skip block probabilistically, decreasing over training
        if self.training and self.dev_dropout:
            p = 0.1 * max(0.0, 1.0 - step / max(total_steps * 0.5, 1))
            if torch.rand(1).item() < p:
                return x, prev_attn

        attn_out, attn_weights = self.attn(rms_norm(x), ve, cos, sin, mask, prev_attn)

        # CA modulation on attention output
        if self.ca_ch is not None and self.ca_mode in ("attn", "both", "multiscale"):
            if self.ca_mode == "multiscale":
                ca_signal = self.ca_ch(x, layer_idx=self.layer_idx)
            else:
                ca_signal = self.ca_ch(x)
            attn_out = attn_out * (1 + torch.tanh(ca_signal) * 0.1)
        elif self.ca_ch is not None and self.ca_mode == "additive":
            attn_out = attn_out + self.ca_ch(x) * 0.1

        x = x + attn_out

        mlp_out = self.mlp(rms_norm(x))
        if self.ca_ch is not None and self.ca_mode == "both":
            ca_signal = self.ca_ch(x)
            mlp_out = mlp_out * (1 + torch.tanh(ca_signal) * 0.1)

        # Token vitality
        if self.vitality_net is not None:
            vitality = self.vitality_net(x).squeeze(-1)  # (B, T)
            # Neighbor smoothing
            v_smooth = F.avg_pool1d(vitality.unsqueeze(1), 5, 1, 2).squeeze(1)
            vitality = 0.7 * vitality + 0.3 * v_smooth
            vitality = torch.where(vitality > 0.3, vitality, vitality * 0.1)
            mlp_out = mlp_out * vitality.unsqueeze(-1)

        x = x + mlp_out
        return x, attn_weights

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_kv_head,
                 n_embd, arch_cfg=None):
        super().__init__()
        self.arch_cfg = arch_cfg or {}
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        head_dim = n_embd // n_head
        kv_dim = n_kv_head * head_dim

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, n_kv_head, n_layer, i, self.arch_cfg)
            for i in range(n_layer)
        ])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(n_layer))
        smear_ch = min(24, n_embd)
        self.smear_gate = nn.Linear(smear_ch, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self._smear_ch = smear_ch
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(vocab_size, kv_dim)
            for i in range(n_layer) if has_ve(i, n_layer)
        })
        ch = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (100000.0 ** (ch / head_dim))
        t = torch.arange(block_size, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, :, None, :], persistent=False)

        # Cross-layer CA state
        if self.arch_cfg.get("cross_ca"):
            ca_dim = 32
            self.ca_proj_in = nn.Linear(n_embd, ca_dim, bias=False)
            self.ca_rule = nn.Sequential(nn.Linear(ca_dim, ca_dim, bias=False), nn.Tanh())
            self.ca_proj_out = nn.Linear(ca_dim, n_embd, bias=False)
            nn.init.zeros_(self.ca_proj_out.weight)

        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        n = self.n_embd
        s = 3**0.5 * n**-0.5
        nn.init.normal_(self.wte.weight, 0, 0.8)
        nn.init.normal_(self.lm_head.weight, 0, 0.001)
        for b in self.blocks:
            nn.init.uniform_(b.attn.c_q.weight, -s, s)
            nn.init.uniform_(b.attn.c_k.weight, -s, s)
            nn.init.uniform_(b.attn.c_v.weight, -s, s)
            nn.init.zeros_(b.attn.c_proj.weight)
            if hasattr(b.mlp, 'c_fc'):
                nn.init.uniform_(b.mlp.c_fc.weight, -s * 0.4, s * 0.4)
                nn.init.zeros_(b.mlp.c_proj.weight)
            elif hasattr(b.mlp, 'proj_in'):
                nn.init.uniform_(b.mlp.proj_in.weight, -s * 0.4, s * 0.4)
                nn.init.zeros_(b.mlp.proj_out.weight)
            if b.attn.ve_gate is not None:
                nn.init.uniform_(b.attn.ve_gate.weight, 0, 0.02)
        for i in range(self.n_layer):
            self.resid_lambdas.data[i] = 1.15 - 0.10 * i / max(self.n_layer - 1, 1)
            self.x0_lambdas.data[i] = 0.20 - 0.15 * i / max(self.n_layer - 1, 1)
        for ve in self.value_embeds.values():
            nn.init.uniform_(ve.weight, -s, s)

    def forward(self, idx, targets=None, step=0, total_steps=1):
        B, T = idx.size()
        cos, sin = self.cos[:, :T], self.sin[:, :T]
        x = rms_norm(self.wte(idx))
        if T > 1:
            gate = self.smear_lambda * torch.sigmoid(
                self.smear_gate(x[:, 1:, :self._smear_ch]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        x0 = x
        mid = self.n_layer // 2
        x_mid = None
        prev_attn = None

        # Pre-compute window masks
        win_mode = self.arch_cfg.get("window")
        masks = [compute_window_mask(T, i, self.n_layer, win_mode, x.device)
                 if win_mode else None for i in range(self.n_layer)]

        # Cross-layer CA
        ca_state = self.ca_proj_in(x) if self.arch_cfg.get("cross_ca") else None

        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x, prev_attn = block(x, ve, cos, sin, mask=masks[i],
                                  prev_attn=prev_attn, step=step,
                                  total_steps=total_steps)
            if ca_state is not None:
                ca_state = ca_state + 0.1 * self.ca_rule(ca_state)
            if i == mid:
                x_mid = x

        if x_mid is not None:
            x = x - self.backout_lambda * x_mid
        if ca_state is not None:
            x = x + self.ca_proj_out(ca_state)

        logits = self.lm_head(rms_norm(x)).float()
        logits = 15 * torch.tanh(logits / 15)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_ca_parameters(self):
        """Count parameters belonging to CA-specific modules."""
        ca_params = 0
        for block in self.blocks:
            if block.ca_ch is not None:
                ca_params += sum(p.numel() for p in block.ca_ch.parameters())
            if block.vitality_net is not None:
                ca_params += sum(p.numel() for p in block.vitality_net.parameters())
            if hasattr(block.attn, 'refine_conv') and block.attn.refine_conv is not None:
                ca_params += sum(p.numel() for p in block.attn.refine_conv.parameters())
            if isinstance(block.mlp, CAFFN):
                ca_params += sum(p.numel() for p in block.mlp.ca_conv.parameters())
                ca_params += sum(p.numel() for p in block.mlp.gate.parameters())
        if hasattr(self, 'ca_proj_in'):
            ca_params += sum(p.numel() for p in self.ca_proj_in.parameters())
            ca_params += sum(p.numel() for p in self.ca_rule.parameters())
            ca_params += sum(p.numel() for p in self.ca_proj_out.parameters())
        return ca_params

# ---------------------------------------------------------------------------
# Sleep consolidation
# ---------------------------------------------------------------------------
def sleep_step(model, rule="homeostatic", alpha=0.001):
    """Offline CA weight consolidation (like biological sleep)."""
    with torch.no_grad():
        for name, W in model.named_parameters():
            if W.dim() < 2 or any(s in name for s in ("wte", "lm_head", "ve_gate")):
                continue
            W_2d = W.data.view(W.shape[0], -1)
            if rule == "homeostatic":
                row_std = W_2d.std(dim=1, keepdim=True)
                target = 0.02
                scale = target / (row_std + 1e-8)
                W_2d.mul_((1 - alpha) + alpha * scale.clamp(0.5, 2.0))
            elif rule == "competition":
                row_max = W_2d.abs().max(dim=1, keepdim=True).values
                strong = (W_2d.abs() > 0.5 * row_max).float()
                W_2d.add_(alpha * W_2d * strong - alpha * 0.5 * W_2d * (1 - strong))

# ---------------------------------------------------------------------------
# Universal Circuit Pre-Wiring
# ---------------------------------------------------------------------------
def apply_universal_init(model, mode: str):
    """Pre-wire known universal circuits from interpretability research."""
    n_layer = model.n_layer
    n_embd = model.n_embd
    n_head = model.blocks[0].attn.n_head
    head_dim = model.blocks[0].attn.head_dim

    with torch.no_grad():
        if mode in ("induction", "all"):
            _init_induction_heads(model, n_embd, n_head, head_dim)
        if mode in ("layer_roles", "all"):
            _init_layer_roles(model, n_layer, n_embd)
        if mode in ("diverse_heads", "all"):
            _init_diverse_heads(model, n_head, head_dim)

def _init_induction_heads(model, n_embd, n_head, head_dim):
    """Pre-wire induction circuit: prev-token head in layer 0, induction head in layer 1."""
    if model.n_layer < 2:
        return
    scale = 0.05
    # Layer 0, head 0: "previous token" head
    # V/O as near-identity: pass content through to next layer
    b0 = model.blocks[0]
    v_w = b0.attn.c_v.weight.data
    o_w = b0.attn.c_proj.weight.data
    # Set head 0's V to extract and O to project back (near-identity for that head)
    for i in range(head_dim):
        if i < n_embd:
            v_w[i, i] = scale
            o_w[i, i] = scale

    # Layer 1, head 0: "induction" head — content matching + copy
    b1 = model.blocks[1]
    q_w = b1.attn.c_q.weight.data
    k_w = b1.attn.c_k.weight.data
    v_w = b1.attn.c_v.weight.data
    o_w = b1.attn.c_proj.weight.data
    # Q/K: near-identity in head-0 subspace = content matching
    for i in range(head_dim):
        if i < n_embd:
            q_w[i, i] = scale
            k_w[i, i] = scale
    # V/O: copy mechanism (near-identity)
    for i in range(head_dim):
        if i < n_embd:
            v_w[i, i] = scale
            o_w[i, i] = scale

def _init_layer_roles(model, n_layer, n_embd):
    """Initialize layers with role-appropriate structure.
    Early: local/diagonal. Middle: distributed. Late: output-focused."""
    for idx, block in enumerate(model.blocks):
        progress = idx / max(n_layer - 1, 1)
        for name, W in block.named_parameters():
            if W.dim() < 2 or min(W.shape) < 4:
                continue
            if "ve_gate" in name:
                continue
            rows, cols = W.shape
            if progress < 0.33:
                # Early: near-diagonal — local processing bias
                diag = torch.zeros_like(W)
                bw = max(1, min(rows, cols) // 3)
                for i in range(min(rows, cols)):
                    lo = max(0, i - bw)
                    hi = min(cols, i + bw + 1)
                    diag[i % rows, lo:hi] = torch.randn(hi - lo, device=W.device) * 0.01
                W.data = W.data * 0.5 + diag * 0.5
            elif progress < 0.67:
                # Middle: boost block-diagonal for modularity
                nb = min(4, min(rows, cols))
                bh, bw_ = rows // nb, cols // nb
                for b in range(nb):
                    r0, r1 = b * bh, min((b + 1) * bh, rows)
                    c0, c1 = b * bw_, min((b + 1) * bw_, cols)
                    W.data[r0:r1, c0:c1] *= 1.3
            # Late layers: keep default xavier+CA init

def _init_diverse_heads(model, n_head, head_dim):
    """Diversify heads within each layer: positional, content, local, general."""
    for block in model.blocks:
        q_w = block.attn.c_q.weight.data
        k_w = block.attn.c_k.weight.data
        dev = q_w.device
        for h in range(n_head):
            s = h * head_dim
            e = (h + 1) * head_dim
            role = h % 4
            if role == 0:
                # Positional head: diagonal Q/K
                for i in range(head_dim):
                    idx = (s + i) % q_w.shape[1]
                    q_w[s + i, idx] += 0.03
                    k_w[s + i, idx] += 0.03
            elif role == 1:
                # Content head: add random structure for content matching
                q_w[s:e, :] += torch.randn_like(q_w[s:e, :]) * 0.01
                k_w[s:e, :] += torch.randn_like(k_w[s:e, :]) * 0.01
            elif role == 2:
                # Local head: banded structure
                for i in range(min(head_dim, 4)):  # limit iterations
                    idx = (s + i) % q_w.shape[1]
                    bw = min(8, q_w.shape[1])
                    lo = max(0, idx - bw)
                    hi = min(q_w.shape[1], idx + bw)
                    q_w[s + i, lo:hi] += torch.randn(hi - lo, device=dev) * 0.01
                    k_w[s + i, lo:hi] += torch.randn(hi - lo, device=dev) * 0.01
            # role == 3: general head, keep default init

# ---------------------------------------------------------------------------
# Induction Score Measurement
# ---------------------------------------------------------------------------
def measure_induction_score(model, device, n_samples=50, seq_len=128):
    """Measure induction head behavior: given [A,B,...,A], predict B.
    Returns score in [0,1] where 1 = perfect induction."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_samples):
            # Create sequence with repeated bigram: random prefix + [A,B] + random middle + [A]
            prefix_len = torch.randint(10, 40, (1,)).item()
            mid_len = torch.randint(10, 40, (1,)).item()
            A = torch.randint(1, 256, (1,)).item()
            B = torch.randint(1, 256, (1,)).item()
            # Build sequence
            seq = torch.randint(1, 256, (prefix_len,)).tolist()
            seq.extend([A, B])
            seq.extend(torch.randint(1, 256, (mid_len,)).tolist())
            seq.append(A)  # second occurrence of A
            if len(seq) > seq_len:
                seq = seq[:seq_len]
            ids = torch.tensor([seq], dtype=torch.long, device=device)
            logits, _ = model(ids)
            # Check prediction at position after second A
            pred_pos = len(seq) - 1  # position of second A
            probs = F.softmax(logits[0, pred_pos, :], dim=-1)
            if probs[B].item() > 1.0 / 256:  # better than random
                correct += probs[B].item()
            total += 1
    model.train()
    return correct / max(total, 1)

# ---------------------------------------------------------------------------
# Track J: Embryogenic CA — Activity-Dependent Development
# ---------------------------------------------------------------------------
def _is_embryo_target(name: str, W: torch.Tensor) -> bool:
    if W.dim() < 2:
        return False
    if any(s in name for s in ("wte", "lm_head", "ve_gate", "ca_", "vitality")):
        return False
    if min(W.shape) < 8:
        return False
    return True

def embryo_ca_step(model, grad_dict: dict, rule: str, step: int,
                    total_steps: int, crit_frac: float = 0.2,
                    base_alpha: float = 0.005):
    """Activity-dependent CA step during critical period.
    Returns (n_active, n_closed, mean_alpha) for diagnostics."""
    critical_end = int(total_steps * crit_frac)
    if step >= critical_end:
        return 0, 0, 0.0
    time_factor = 1.0 - step / critical_end
    n_active = 0
    n_closed = 0
    alpha_sum = 0.0
    alpha_count = 0

    with torch.no_grad():
        for name, W in model.named_parameters():
            if not _is_embryo_target(name, W):
                continue
            grad = grad_dict.get(name)
            if grad is None:
                continue

            # Per-weight adaptive alpha based on gradient stability
            grad_mag = grad.abs()
            instability = grad_mag / (grad_mag.mean() + 1e-8)
            instability = instability.clamp(0, 2.0)
            alpha = base_alpha * time_factor * instability

            # Count open/closed weights
            open_mask = (alpha > 0.0005).float()
            n_active += open_mask.sum().item()
            n_closed += (1 - open_mask).sum().item()
            alpha_sum += alpha.mean().item()
            alpha_count += 1

            # Compute CA delta based on rule
            if rule == "strengthen":
                delta = grad_mag * W.data.sign() * 0.01
            elif rule == "hebbian":
                w_mag = W.data.abs()
                w_strength = w_mag / (w_mag.mean() + 1e-8)
                g_strength = grad_mag / (grad_mag.mean() + 1e-8)
                delta = (w_strength * g_strength - 1.0) * W.data * 0.005
            elif rule == "targeted":
                # Different rules for different weight types
                if any(k in name for k in ("c_q", "c_k")):
                    # Hebbian for Q/K (reinforce attention patterns)
                    w_mag = W.data.abs()
                    w_s = w_mag / (w_mag.mean() + 1e-8)
                    g_s = grad_mag / (grad_mag.mean() + 1e-8)
                    delta = (w_s * g_s - 1.0) * W.data * 0.008
                elif any(k in name for k in ("c_v", "c_proj")):
                    # Strengthen for V/O (reinforce value pathways)
                    delta = grad_mag * W.data.sign() * 0.005
                else:
                    continue  # skip FFN weights
            elif rule == "gradalign":
                # Gradient-aligned: push in gradient direction, scaled by
                # row-wise consistency (where a row's gradients agree)
                g_flat = grad.view(W.shape[0], -1)
                row_std = g_flat.std(dim=1, keepdim=True)
                row_mean = g_flat.abs().mean(dim=1, keepdim=True)
                consistency = 1.0 / (row_std / (row_mean + 1e-8) + 1.0)
                # Expand consistency to match W shape
                if W.dim() == 2:
                    cons = consistency.expand_as(W)
                else:
                    cons = consistency.view(-1, *([1] * (W.dim() - 1))).expand_as(W)
                delta = grad.sign() * cons * 0.01
            else:
                continue

            W.data.add_(alpha * delta)

    mean_alpha = alpha_sum / max(alpha_count, 1)
    return int(n_active), int(n_closed), mean_alpha

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(time_budget: float | None = None, seed: int | None = None,
          arch: str = "baseline", quiet: bool = False, lr_override: float | None = None):
    if time_budget is None:
        time_budget = TIME_BUDGET
    arch_cfg = ARCHS.get(arch, {})
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    lr = lr_override if lr_override else LR

    if not quiet:
        print(f"device: {DEVICE}")
        print(f"arch: {arch} {arch_cfg}")
        print(f"time_budget: {time_budget:.0f}s ({time_budget/60:.1f} min)")
        print(f"lr: {lr:.1e}  seed: {seed}")

    train_data = load_data("train")
    val_data = load_data("val")
    block_size = MAX_SEQ_LEN

    model = GPT(VOCAB_SIZE, block_size, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS,
                arch_cfg=arch_cfg).to(DEVICE)
    total_params = model.count_parameters()
    ca_params = model.count_ca_parameters()
    if not quiet:
        print(f"params: {total_params:,} (ca: {ca_params:,})")

    # Apply block-CA init (Round 2 best)
    with torch.no_grad():
        from ca_rules import block_diagonal_init
        for name, p in model.named_parameters():
            if p.dim() >= 2 and not any(s in name for s in ("wte","lm_head","ve_gate")):
                if min(p.shape) >= 8:
                    nn.init.xavier_uniform_(p)
                    ca = block_diagonal_init(p.shape, n_blocks=min(4, min(p.shape)),
                                            target_std=p.std().item() * 0.05)
                    p.data.add_(ca.to(p.device))

    # Apply universal circuit pre-wiring (after standard init)
    universal_mode = arch_cfg.get("universal")
    if universal_mode:
        apply_universal_init(model, universal_mode)
        if not quiet:
            print(f"universal_init: {universal_mode}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    model.train()
    step = 0
    max_steps = 100_000
    min_lr = lr / 10
    t0 = time.time()
    est_total = int(time_budget / 0.12)  # rough estimate for depth 4

    curve_interval = max(50, int(time_budget / 60 / 10))
    curve_time_min = 30.0
    last_curve = 0.0
    loss_curve = []
    ca_time_total = 0.0

    while True:
        elapsed = time.time() - t0
        if elapsed >= time_budget:
            break
        x, y = get_batch(train_data, BATCH_SIZE, block_size, DEVICE)
        cur_lr = get_lr(step, WARMUP, max_steps, lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr
        _, loss = model(x, y, step=step, total_steps=est_total)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Capture gradients for embryogenic CA BEFORE optimizer step
        embryo_rule = arch_cfg.get("embryo")
        embryo_grad_dict = {}
        if embryo_rule:
            embryo_freq = arch_cfg.get("embryo_freq", 10)
            embryo_crit = arch_cfg.get("embryo_crit", 0.2)
            if step % embryo_freq == 0 and step < est_total * embryo_crit:
                for name, p in model.named_parameters():
                    if p.grad is not None and _is_embryo_target(name, p):
                        embryo_grad_dict[name] = p.grad.clone()

        optimizer.step()

        # Embryogenic CA step (activity-dependent, during critical period)
        if embryo_grad_dict:
            t_ca = time.time()
            n_act, n_cls, m_alpha = embryo_ca_step(
                model, embryo_grad_dict, embryo_rule, step, est_total,
                crit_frac=embryo_crit)
            ca_time_total += time.time() - t_ca

        # Sleep consolidation
        if arch_cfg.get("sleep"):
            interval = arch_cfg.get("sleep_interval", 500)
            if step > 0 and step % interval == 0:
                t_ca = time.time()
                rule = arch_cfg.get("sleep_rule", "homeostatic")
                for _ in range(arch_cfg.get("sleep_steps", 10)):
                    sleep_step(model, rule=rule, alpha=0.001)
                ca_time_total += time.time() - t_ca

        # Eval checkpoint
        elapsed = time.time() - t0
        should_record = (step > 0 and step % curve_interval == 0
                         and (elapsed - last_curve) >= curve_time_min)
        if should_record or step == 0:
            vbpb = evaluate_val_bpb(model, val_data, BATCH_SIZE, block_size, DEVICE)
            # Measure induction score at key checkpoints
            ind_score = None
            if universal_mode and step in (0, 100, 500, 2000, 5000, 10000):
                ind_score = measure_induction_score(model, DEVICE)
            loss_curve.append((step, round(elapsed, 1), round(vbpb, 4)))
            last_curve = elapsed
            if not quiet:
                extra = f" ind:{ind_score:.3f}" if ind_score is not None else ""
                print(f"step:{step:5d} loss:{loss.item():.4f} val_bpb:{vbpb:.4f} "
                      f"lr:{cur_lr:.2e} {elapsed:.0f}s{extra}")
        elif not quiet and step % 500 == 0:
            print(f"step:{step:5d} loss:{loss.item():.4f} lr:{cur_lr:.2e} {elapsed:.0f}s")
        step += 1

    val_bpb = evaluate_val_bpb(model, val_data, BATCH_SIZE, block_size, DEVICE)
    total_time = time.time() - t0
    loss_curve.append((step, round(total_time, 1), round(val_bpb, 4)))
    ca_overhead = ca_time_total / total_time * 100 if total_time > 0 else 0

    # Final induction score
    final_ind_score = measure_induction_score(model, DEVICE) if universal_mode else None

    if not quiet:
        print(f"\nval_bpb: {val_bpb:.4f}")
        print(f"total_steps: {step}")
        print(f"params: {total_params} (ca: {ca_params})")
        print(f"ca_overhead_pct: {ca_overhead:.1f}")
        print(f"wall_time_s: {total_time:.1f}")
        if final_ind_score is not None:
            print(f"induction_score: {final_ind_score:.4f}")

    result = {
        "val_bpb": round(val_bpb, 4),
        "final_train_loss": round(loss.item(), 4),
        "total_steps": step,
        "params": total_params,
        "ca_params": ca_params,
        "ca_overhead_pct": round(ca_overhead, 1),
        "wall_time_s": round(total_time, 1),
        "arch": arch,
        "seed": seed,
        "lr": lr,
        "time_budget_s": time_budget,
        "loss_curve": loss_curve,
        "induction_score": round(final_ind_score, 4) if final_ind_score is not None else None,
    }

    # Save checkpoint for quality evaluation
    result["_model"] = model  # keep model in memory for live eval
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"model_{arch}_{seed}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "arch": arch,
        "seed": seed,
        "val_bpb": val_bpb,
        "lr": lr,
        "total_steps": step,
    }, ckpt_path)
    if not quiet:
        print(f"checkpoint_saved: {ckpt_path}")

    result_json = {k: v for k, v in result.items() if k != "_model"}
    print(f"RESULT_JSON: {json.dumps(result_json)}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroGen Round 4")
    parser.add_argument("--arch", type=str, default="baseline",
                        choices=list(ARCHS.keys()), help="Architecture variant")
    parser.add_argument("--minutes", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Override LR")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    budget = args.minutes * 60 if args.minutes else None
    train(time_budget=budget, seed=args.seed, arch=args.arch,
          quiet=args.quiet, lr_override=args.lr)
