"""Tests for the shared attention-window schedule (windows.py).

These lock the window math so the refactor that unified three duplicated
implementations (train_r4.py, train_125m.py x2) cannot silently drift. They run
on CPU and need no data download.
"""
import math

import torch
import pytest

import windows

R4_MODES = [
    "linear", "quadratic", "step",
    "power_0.5", "power_2.0", "power_4.0", "power_5.0",
    "sigmoid_0.3", "sigmoid_0.5", "sigmoid_0.7",
    "logarithmic", "exponential", "fibonacci",
]


# --------------------------------------------------------------------------
# Frozen copies of the ORIGINAL implementations (pre-refactor), used as a
# golden reference. If windows.py ever diverges from these, the test fails.
# --------------------------------------------------------------------------
def legacy_r4_mask(T, layer_idx, n_layer, mode):
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
        s = 1.0 / (1.0 + math.exp(-10.0 * (progress - mid)))
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
    window = min(window, T)
    rows = torch.arange(T).unsqueeze(1)
    cols = torch.arange(T).unsqueeze(0)
    return ((cols <= rows) & (cols >= rows - window + 1)).float()


def legacy_125m_mask(T, layer_idx, n_layer, mode):
    if mode == "none":
        return None
    progress = (layer_idx + 1) / n_layer
    base = 16
    if mode.startswith("power_"):
        exp = float(mode.split("_")[1])
        window = max(base, int(base + progress ** exp * (T - base)))
    else:
        return None
    window = min(window, T)
    rows = torch.arange(T).unsqueeze(1)
    cols = torch.arange(T).unsqueeze(0)
    return (cols <= rows) & (cols >= rows - window + 1)


def legacy_125m_size(layer_idx, n_layer, seq_len, mode):
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


# --------------------------------------------------------------------------
# Behavior tests
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", R4_MODES)
def test_mask_is_causal(mode):
    T, n_layer = 64, 4
    for li in range(n_layer):
        m = windows.compute_window_mask(T, li, n_layer, mode, "cpu", base=8)
        assert m is not None
        # No position may attend strictly into the future.
        future = torch.triu(torch.ones(T, T), diagonal=1).bool()
        assert m[future].sum().item() == 0, f"{mode} L{li} attends to the future"


@pytest.mark.parametrize("mode", R4_MODES)
def test_mask_width_matches_window(mode):
    T, n_layer = 128, 4
    for li in range(n_layer):
        w = windows.compute_window_size(li, n_layer, T, mode, base=8)
        m = windows.compute_window_mask(T, li, n_layer, mode, "cpu", base=8)
        # The last (fully-seen) row attends to exactly min(window, T) keys.
        assert int(m[T - 1].sum().item()) == min(w, T)
        # An early row i attends to min(window, i+1) keys.
        i = min(w + 5, T - 1)
        assert int(m[i].sum().item()) == min(w, i + 1)


def test_last_layer_power_is_full_causal():
    T, n_layer = 256, 4
    m = windows.compute_window_mask(T, n_layer - 1, n_layer, "power_4.0", "cpu", base=8)
    full_causal = torch.tril(torch.ones(T, T))
    assert torch.equal(m, full_causal)
    assert windows.compute_window_size(n_layer - 1, n_layer, T, "power_4.0", base=8) == T


def test_base_floor_respected():
    T, n_layer = 256, 4
    # Steep schedule keeps layer 0 at the base floor.
    assert windows.compute_window_size(0, n_layer, T, "power_4.0", base=8) == 8
    assert windows.compute_window_size(0, n_layer, T, "power_4.0", base=16) == 16


@pytest.mark.parametrize("mode", [None, "none", "baseline", "totally_unknown"])
def test_full_attention_modes_return_none(mode):
    assert windows.compute_window_size(0, 4, 256, mode, base=8) is None
    assert windows.compute_window_mask(256, 0, 4, mode, "cpu", base=8) is None


def test_sliding_window_none_when_full():
    T, n_layer = 1024, 12
    # Last layer of a power schedule reaches full attention -> None for flash-attn.
    assert windows.compute_sliding_window(n_layer - 1, n_layer, T, "power_4.0", base=16) is None
    # Early layer is windowed -> a concrete int < T.
    w0 = windows.compute_sliding_window(0, n_layer, T, "power_4.0", base=16)
    assert isinstance(w0, int) and 0 < w0 < T


# --------------------------------------------------------------------------
# Golden tests: shared module must reproduce the legacy implementations.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", R4_MODES)
def test_golden_r4_float_mask(mode):
    for T, n_layer in [(256, 4), (64, 4), (128, 6)]:
        for li in range(n_layer):
            a = legacy_r4_mask(T, li, n_layer, mode)
            b = windows.compute_window_mask(T, li, n_layer, mode, "cpu", base=8, dtype=torch.float32)
            assert b.dtype == torch.float32
            assert torch.equal(a, b), f"r4 mismatch {mode} T={T} L{li}"


@pytest.mark.parametrize("mode", ["power_2.0", "power_4.0", "power_8.0", "power_12.0", "none"])
def test_golden_125m_bool_mask_and_size(mode):
    for T, n_layer in [(1024, 12), (512, 12)]:
        for li in range(n_layer):
            a = legacy_125m_mask(T, li, n_layer, mode)
            b = windows.compute_window_mask(T, li, n_layer, mode, "cpu", base=16, dtype=torch.bool)
            if a is None:
                assert b is None
            else:
                assert b.dtype == torch.bool
                assert torch.equal(a, b), f"125m mask mismatch {mode} T={T} L{li}"
            sa = legacy_125m_size(li, n_layer, T, mode)
            sb = windows.compute_sliding_window(li, n_layer, T, mode, base=16)
            assert sa == sb, f"125m size mismatch {mode} T={T} L{li}: {sa} != {sb}"
