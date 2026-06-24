"""Layer-dependent attention window schedules — one source of truth.

The developmental attention-window mechanism (the core NeuroGen result) was
duplicated in three places with divergent constants and feature sets:

  * train_r4.py    compute_window_mask   base=8,  all modes, float mask
  * train_125m.py  compute_window_mask   base=16, power_ only, bool mask
  * train_125m.py  _compute_window_size  base=16, power_ only, int (flash-attn)

That meant a "window_quadratic" run was silently impossible at 125M, so the
cross-scale claim compared non-identical operators. This module centralizes the
window math so every caller uses the same schedule, modulo the `base` floor
(which scales with context length) and the mask dtype.

`compute_window_size` is the single schedule definition; the two mask builders
and the flash-attn integer path all derive from it.
"""
import math

import torch

# Minimum attention window ("base" floor). It scales with context length:
# 8 for the ~256-token 3.4M model, 16 for the ~1024-token 125M model.
DEFAULT_BASE_SMALL = 8
DEFAULT_BASE_LARGE = 16


def compute_window_size(layer_idx: int, n_layer: int, seq_len: int,
                        mode: str, base: int = DEFAULT_BASE_SMALL):
    """Attention window width (in tokens) for one layer.

    Returns an int in [base, seq_len], or None for full attention. `mode`
    selects the depth->window schedule; None / "none" / "baseline" and any
    unrecognized mode return None (full attention).
    """
    if mode in (None, "none", "baseline"):
        return None
    progress = (layer_idx + 1) / n_layer
    T = seq_len
    if mode.startswith("list:"):
        # Explicit per-layer windows, e.g. "list:8,256,256,256". Used for
        # per-layer ablations (which layer's locality carries the effect). No
        # base floor — the listed widths are taken verbatim.
        widths = [int(w) for w in mode[len("list:"):].split(",")]
        window = widths[layer_idx] if layer_idx < len(widths) else T
        return min(window, T)
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
        window = max(base, int(base + (math.exp(progress * 3) - 1) / (math.e ** 3 - 1) * (T - base)))
    elif mode == "fibonacci":
        fibs = [base, base * 2]
        for _ in range(2, n_layer):
            fibs.append(min(fibs[-1] + fibs[-2], T))
        window = fibs[min(layer_idx, len(fibs) - 1)]
    else:
        return None
    return min(window, T)


def compute_window_mask(T: int, layer_idx: int, n_layer: int, mode: str, device,
                        base: int = DEFAULT_BASE_SMALL, dtype=torch.float32):
    """Causal + windowed attention mask of shape (T, T), or None for full attention.

    dtype=torch.float32 reproduces train_r4.py's float mask; dtype=torch.bool
    reproduces train_125m.py's boolean mask. A valid windowed mode always
    returns a mask (a full causal mask when the window reaches T), matching the
    original behavior.
    """
    window = compute_window_size(layer_idx, n_layer, T, mode, base=base)
    if window is None:
        return None
    rows = torch.arange(T, device=device).unsqueeze(1)
    cols = torch.arange(T, device=device).unsqueeze(0)
    mask = (cols <= rows) & (cols >= rows - window + 1)
    return mask if dtype == torch.bool else mask.to(dtype)


def compute_sliding_window(layer_idx: int, n_layer: int, seq_len: int,
                           mode: str, base: int = DEFAULT_BASE_LARGE):
    """Integer window for flash-attn's native sliding window, or None.

    Returns None when the window covers the whole sequence (full attention is
    cheaper without a window), preserving train_125m.py's original contract.
    """
    window = compute_window_size(layer_idx, n_layer, seq_len, mode, base=base)
    if window is None or window >= seq_len:
        return None
    return window
