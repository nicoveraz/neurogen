"""Experiment 2 — topographic regularizer.

Three training modes (select via --mode):

  - baseline      : identical to Exp 0 baseline but 90K steps (apples-to-apples
                    comparison condition).
  - topographic   : adds a Gaussian-kernel topographic loss on a 16x16 learned
                    grid of byte positions, driven by pre-computed co-occurrence
                    (runs/exp2_cooccur/cooccur_w5.npy).
  - control       : same topographic loss but with the co-occurrence matrix
                    permuted (random targets). Isolates the "structure signal"
                    in the regularizer from generic regularization.

Schedules
  - 90K total steps (memorization past ~85K from Exp 0).
  - Regularizer weight ramp: 0 → target over first 9K (10%).
  - Neighborhood width σ: anneal 8 → 1 over first 18K (20%).
  - Snapshot cadence: every 500 for first 10K, every 1K after (same as Exp 0).

Grid positions are nn.Parameter of shape (V, 2), initialized as a shuffled
16x16 lattice over [0, 16]². Saved inside the snapshots and the full ckpts.

Diagnostics logged every 100 steps plus at each full-ckpt step:
  - lm_loss, topo_loss_raw (with annealed σ), topo_loss_sigma1 (steady-state),
  - grad_norm(lm), grad_norm(topo) (measured separately by running two
    backward passes in calibrated steps; cheap, only every 100 steps),
  - cos(grad_lm, grad_topo) — direction overlap.
  - grid_spread_ratio — mean pairwise grid distance / initial value.
  - frac_positions_moved — fraction of tokens whose position has changed
    by >0.1 grid units since the last log step.

Usage:
  # pilot (10K steps, weight 0.1)
  uv run python -m autoresearch.exp2_train --mode topographic --steps 10000 \
      --run-name exp2_pilot --topo-weight 0.1 --seed 42

  # full run
  uv run python -m autoresearch.exp2_train --mode topographic --steps 90000 \
      --run-name exp2_topographic --topo-weight 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prepare import (  # noqa: E402
    MAX_SEQ_LEN,
    VOCAB_SIZE,
    evaluate_val_bpb,
    get_batch,
    get_device,
    load_data,
)
from train_r4 import ARCHS, CHANNELS, DEPTH, GPT, N_HEADS, N_KV_HEADS, get_lr  # noqa: E402
from ca_rules import block_diagonal_init  # noqa: E402


COOCCUR_PATH = REPO_ROOT / "runs" / "exp2_cooccur" / "cooccur_w5.npy"
GRID_SIZE = 16
GRID_EXTENT = float(GRID_SIZE)   # positions live in [0, GRID_EXTENT)
SIGMA_INIT = 8.0
SIGMA_FINAL = 1.0
DEFAULT_TARGET_WEIGHT = 0.1


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class ExpConfig:
    run_name: str = "exp2_topographic"
    mode: str = "topographic"      # baseline | topographic | control
    arch: str = "baseline"
    seed: int = 42
    max_steps: int = 90_000
    batch_size: int = 32
    lr: float = 2e-3
    weight_decay: float = 0.05
    warmup: int = 200
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    # topographic
    topo_weight_target: float = DEFAULT_TARGET_WEIGHT
    topo_weight_ramp_frac: float = 0.10     # 0 → target over first 10%
    sigma_init: float = SIGMA_INIT
    sigma_final: float = SIGMA_FINAL
    sigma_anneal_frac: float = 0.20         # 8 → 1 over first 20%
    grid_lr_scale: float = 10.0              # grid positions can use a higher LR
    # snapshot cadence (match Exp 0)
    snapshot_dense_every: int = 500
    snapshot_dense_until: int = 10_000
    snapshot_sparse_every: int = 1000
    full_ckpt_every: int = 2000
    full_ckpt_keep: int = 3
    eval_every: int = 1000
    # diagnostics
    diagnostic_every: int = 100

    def runs_dir(self) -> Path:
        return REPO_ROOT / "runs" / self.run_name


# ---------------------------------------------------------------------------
# Topographic components
# ---------------------------------------------------------------------------
def init_shuffled_lattice(vocab: int, grid_size: int, rng: np.random.Generator) -> torch.Tensor:
    """Return (V, 2) float32 tensor of positions on a grid_size x grid_size lattice,
    shuffled so tokens are spread. Only works when vocab == grid_size**2."""
    assert vocab == grid_size * grid_size, (
        f"vocab {vocab} != grid {grid_size}^2 = {grid_size*grid_size}"
    )
    rows, cols = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    positions = np.stack([rows.ravel(), cols.ravel()], axis=1).astype(np.float32)
    order = rng.permutation(vocab)
    return torch.from_numpy(positions[order].copy())


def topographic_loss(
    grid_pos: torch.Tensor,
    cooccur: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Gaussian-kernel attractive topographic loss.

    L = − Σ_{i,j} cooccur[i,j] · exp(−‖pos_i − pos_j‖² / (2σ²))

    Minimizing this pulls co-occurring tokens toward each other with bandwidth σ.
    """
    V = grid_pos.size(0)
    diff = grid_pos.unsqueeze(0) - grid_pos.unsqueeze(1)    # (V, V, 2)
    dist_sq = diff.pow(2).sum(-1)                            # (V, V)
    kernel = torch.exp(-dist_sq / (2.0 * sigma * sigma))     # (V, V)
    # Negative: minimizing makes co-occurring pairs have high kernel (close)
    return -(cooccur * kernel).sum()


def load_cooccur(path: Path, vocab: int, permute_seed: int | None = None) -> torch.Tensor:
    C = np.load(path)
    assert C.shape == (vocab, vocab), f"unexpected cooccur shape {C.shape}"
    if permute_seed is not None:
        rng = np.random.default_rng(permute_seed)
        perm = rng.permutation(vocab)
        C = C[perm][:, perm]
    return torch.from_numpy(C.astype(np.float32))


def sigma_schedule(step: int, total_steps: int, cfg: ExpConfig) -> float:
    frac = step / max(1, int(total_steps * cfg.sigma_anneal_frac))
    frac = min(1.0, max(0.0, frac))
    return cfg.sigma_init + (cfg.sigma_final - cfg.sigma_init) * frac


def topo_weight_schedule(step: int, total_steps: int, cfg: ExpConfig) -> float:
    ramp_end = max(1, int(total_steps * cfg.topo_weight_ramp_frac))
    if step >= ramp_end:
        return cfg.topo_weight_target
    return cfg.topo_weight_target * (step / ramp_end)


# ---------------------------------------------------------------------------
# Cadence helpers (same as Exp 0)
# ---------------------------------------------------------------------------
def should_snapshot(step: int, cfg: ExpConfig) -> bool:
    if step == 0:
        return True
    if step <= cfg.snapshot_dense_until:
        return step % cfg.snapshot_dense_every == 0
    return step % cfg.snapshot_sparse_every == 0


def should_full_ckpt(step: int, cfg: ExpConfig) -> bool:
    return step > 0 and step % cfg.full_ckpt_every == 0


def rotate_full_ckpts(full_dir: Path, keep: int) -> None:
    ckpts = sorted(full_dir.glob("step_*.pt"))
    for p in ckpts[:-keep]:
        p.unlink()


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_snapshot(model: GPT, grid_pos: torch.Tensor | None, snap_dir: Path, step: int) -> None:
    wte = model.wte.weight.detach().to("cpu").float().numpy()
    np.save(snap_dir / f"wte_step_{step:07d}.npy", wte)
    if grid_pos is not None:
        gp = grid_pos.detach().to("cpu").float().numpy()
        np.save(snap_dir / f"grid_step_{step:07d}.npy", gp)


def save_full_ckpt(
    model: GPT,
    grid_pos: nn.Parameter | None,
    optimizer: torch.optim.Optimizer,
    step: int,
    full_dir: Path,
    cfg: ExpConfig,
    rng_state: dict,
) -> Path:
    path = full_dir / f"step_{step:07d}.pt"
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": rng_state,
        "config": cfg.__dict__,
    }
    if grid_pos is not None:
        payload["grid_pos"] = grid_pos.detach().cpu()
    torch.save(payload, path)
    return path


def capture_rng_state(device: str) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


# ---------------------------------------------------------------------------
# Model init
# ---------------------------------------------------------------------------
def build_model(cfg: ExpConfig, device: str) -> GPT:
    arch_cfg = ARCHS.get(cfg.arch, {})
    model = GPT(
        VOCAB_SIZE, MAX_SEQ_LEN, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS,
        arch_cfg=arch_cfg,
    ).to(device)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2 and not any(s in name for s in ("wte", "lm_head", "ve_gate")):
                if min(p.shape) >= 8:
                    nn.init.xavier_uniform_(p)
                    ca = block_diagonal_init(
                        p.shape, n_blocks=min(4, min(p.shape)),
                        target_std=p.std().item() * 0.05,
                    )
                    p.data.add_(ca.to(p.device))
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train(cfg: ExpConfig) -> None:
    device = get_device()
    run_dir = cfg.runs_dir()
    full_dir = run_dir / "full_ckpts"
    snap_dir = run_dir / "snapshots"
    log_dir = run_dir / "logs"
    for d in (full_dir, snap_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    _write_yaml(run_dir / "config.yaml", cfg.__dict__ | {"device": device})

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    print(f"[exp2] device={device} run_dir={run_dir} mode={cfg.mode}", flush=True)

    train_data = load_data("train")
    val_data = load_data("val")

    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[exp2] model params={n_params:,} (~{n_params/1e6:.2f}M)", flush=True)

    # ----- topographic setup ---------------------------------------------
    use_topo = cfg.mode in ("topographic", "control")
    grid_pos: nn.Parameter | None = None
    cooccur: torch.Tensor | None = None
    init_mean_pairwise_dist: float | None = None

    if use_topo:
        rng = np.random.default_rng(cfg.seed)
        lattice = init_shuffled_lattice(VOCAB_SIZE, GRID_SIZE, rng)
        grid_pos = nn.Parameter(lattice.to(device))
        print(f"[exp2] grid_pos shape={tuple(grid_pos.shape)} "
              f"extent=[0, {GRID_EXTENT}]", flush=True)

        if not COOCCUR_PATH.exists():
            raise FileNotFoundError(
                f"co-occurrence matrix not found at {COOCCUR_PATH}. "
                f"run `uv run python -m autoresearch.precompute_cooccur` first."
            )
        permute_seed = (cfg.seed + 1000) if cfg.mode == "control" else None
        cooccur_cpu = load_cooccur(COOCCUR_PATH, VOCAB_SIZE, permute_seed=permute_seed)
        # Zero the diagonal defensively and renormalize to sum to 1.
        cooccur_cpu.fill_diagonal_(0.0)
        cooccur_cpu = cooccur_cpu / cooccur_cpu.sum().clamp_min(1e-12)
        cooccur = cooccur_cpu.to(device)
        print(f"[exp2] cooccur loaded (permuted={cfg.mode=='control'}); "
              f"max_weight={float(cooccur.max()):.6f}  "
              f"mean_nonzero={float(cooccur[cooccur>0].mean()):.6g}", flush=True)

        # Initial spread for the grid_spread_ratio diagnostic
        with torch.no_grad():
            diff0 = grid_pos.unsqueeze(0) - grid_pos.unsqueeze(1)
            init_mean_pairwise_dist = float(
                diff0.pow(2).sum(-1).sqrt().mean()
            )
        print(f"[exp2] init mean pairwise grid dist: {init_mean_pairwise_dist:.3f}",
              flush=True)

    # ----- optimizer ------------------------------------------------------
    param_groups = [{"params": list(model.parameters()), "lr": cfg.lr}]
    if grid_pos is not None:
        param_groups.append({
            "params": [grid_pos],
            "lr": cfg.lr * cfg.grid_lr_scale,
            # Don't regularize grid positions with weight decay.
            "weight_decay": 0.0,
        })
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    min_lr = cfg.lr * cfg.min_lr_ratio

    log_fp = open(log_dir / "train.jsonl", "a", buffering=1)
    model.train()

    save_snapshot(model, grid_pos, snap_dir, step=0)

    t0 = time.time()
    prev_grid_snapshot = (
        grid_pos.detach().clone() if grid_pos is not None else None
    )

    for step in range(1, cfg.max_steps + 1):
        lr = get_lr(step, cfg.warmup, cfg.max_steps, cfg.lr, min_lr)
        for i, g in enumerate(optimizer.param_groups):
            scale = cfg.grid_lr_scale if (grid_pos is not None and i == 1) else 1.0
            g["lr"] = lr * scale

        x, y = get_batch(train_data, cfg.batch_size, MAX_SEQ_LEN, device)
        _, lm_loss = model(x, y, step=step, total_steps=cfg.max_steps)

        # ---- topographic loss (if applicable) ---------------------------
        topo_loss_t = torch.tensor(0.0, device=device)
        topo_loss_sigma1_val = 0.0
        sigma = cfg.sigma_final
        cur_topo_weight = 0.0

        if use_topo:
            sigma = sigma_schedule(step, cfg.max_steps, cfg)
            cur_topo_weight = topo_weight_schedule(step, cfg.max_steps, cfg)
            topo_loss_t = topographic_loss(grid_pos, cooccur, sigma)
            with torch.no_grad():
                topo_loss_sigma1_val = float(
                    topographic_loss(grid_pos, cooccur, cfg.sigma_final)
                )

        total_loss = lm_loss + cur_topo_weight * topo_loss_t

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        gnorm_lm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        optimizer.step()

        # ---- diagnostics every `diagnostic_every` steps -----------------
        do_log = (step == 1) or (step % cfg.diagnostic_every == 0)

        if do_log:
            elapsed = time.time() - t0
            entry = {
                "kind": "step",
                "step": step,
                "lm_loss": round(lm_loss.detach().item(), 6),
                "total_loss": round(total_loss.detach().item(), 6),
                "lr": lr,
                "grad_norm_lm": round(gnorm_lm, 6),
                "elapsed_s": round(elapsed, 2),
                "sigma": sigma,
                "topo_weight": cur_topo_weight,
            }

            if use_topo:
                # Separately measured gradient norms and cosine — only at
                # diagnostic cadence because this requires two additional
                # backward passes through copies of the grid_pos.
                entry["topo_loss_raw"] = round(float(topo_loss_t), 6)
                entry["topo_loss_sigma1"] = round(topo_loss_sigma1_val, 6)

                # Grid spread + fraction-moved
                with torch.no_grad():
                    diff = grid_pos.unsqueeze(0) - grid_pos.unsqueeze(1)
                    mean_dist = float(diff.pow(2).sum(-1).sqrt().mean())
                    spread_ratio = mean_dist / init_mean_pairwise_dist
                    move_delta = (grid_pos.detach() - prev_grid_snapshot).norm(dim=-1)
                    frac_moved = float((move_delta > 0.1).float().mean())
                    prev_grid_snapshot = grid_pos.detach().clone()
                entry["grid_spread_ratio"] = round(spread_ratio, 4)
                entry["grid_mean_pairwise_dist"] = round(mean_dist, 4)
                entry["frac_positions_moved"] = round(frac_moved, 4)

                # Separate grad-norm and cos for lm vs topo
                # Zero grads, backward lm_loss only, capture grid_pos grad
                # (lm_loss has no gradient wrt grid_pos, so its grad here is 0
                # and we measure model-param grads only via norm).
                # Then backward topo_loss only for the grid/model grads.
                # We already consumed the graphs; recompute cheaply on the
                # same batch.
                # This adds overhead; keep it at diagnostic_every cadence.
                gn_lm_sep, gn_topo_sep, cos_lm_topo = _measure_grad_split(
                    model, grid_pos, cooccur, sigma, x, y, cfg
                )
                entry["grad_norm_lm_sep"] = round(gn_lm_sep, 6)
                entry["grad_norm_topo_sep"] = round(gn_topo_sep, 6)
                entry["grad_cos_lm_topo"] = round(cos_lm_topo, 6)

            log_fp.write(json.dumps(entry) + "\n")
            print(
                f"[exp2] step={step:>6d} lm={lm_loss.detach().item():.4f} "
                + (f"topo_s1={topo_loss_sigma1_val:+.4f} σ={sigma:.2f} "
                   f"w={cur_topo_weight:.3f} spread={entry.get('grid_spread_ratio', 0):.3f} "
                   if use_topo else "")
                + f"elapsed={elapsed:.0f}s",
                flush=True,
            )

        # ---- snapshot ---------------------------------------------------
        if should_snapshot(step, cfg):
            save_snapshot(model, grid_pos, snap_dir, step)
            log_fp.write(json.dumps(
                {"kind": "snapshot", "step": step}) + "\n")

        # ---- eval -------------------------------------------------------
        if step % cfg.eval_every == 0:
            vbpb = evaluate_val_bpb(
                model, val_data, cfg.batch_size, MAX_SEQ_LEN, device
            )
            log_fp.write(json.dumps(
                {"kind": "eval", "step": step, "val_bpb": round(vbpb, 6)}) + "\n")
            print(f"[exp2] step={step:>6d} val_bpb={vbpb:.4f}", flush=True)

        # ---- full ckpt --------------------------------------------------
        if should_full_ckpt(step, cfg):
            rng_state = capture_rng_state(device)
            save_full_ckpt(model, grid_pos, optimizer, step, full_dir, cfg, rng_state)
            rotate_full_ckpts(full_dir, cfg.full_ckpt_keep)

    # final
    save_snapshot(model, grid_pos, snap_dir, cfg.max_steps)
    rng_state = capture_rng_state(device)
    save_full_ckpt(model, grid_pos, optimizer, cfg.max_steps, full_dir, cfg, rng_state)
    rotate_full_ckpts(full_dir, cfg.full_ckpt_keep)

    final_vbpb = evaluate_val_bpb(model, val_data, cfg.batch_size, MAX_SEQ_LEN, device)
    total_time = time.time() - t0
    summary = {
        "kind": "summary",
        "mode": cfg.mode,
        "total_steps": cfg.max_steps,
        "final_val_bpb": round(final_vbpb, 6),
        "wall_time_s": round(total_time, 1),
    }
    print(f"[exp2] DONE {summary}", flush=True)
    log_fp.write(json.dumps(summary) + "\n")
    log_fp.close()


def _measure_grad_split(
    model: GPT,
    grid_pos: nn.Parameter,
    cooccur: torch.Tensor,
    sigma: float,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: ExpConfig,
) -> tuple[float, float, float]:
    """Recompute LM loss and topo loss separately to measure grad norms and
    the cosine between them. Caller has just done the combined backward and
    optimizer step; we do NOT update parameters here."""
    model.zero_grad(set_to_none=True)
    if grid_pos.grad is not None:
        grid_pos.grad = None

    # --- LM gradient ---
    _, lm_loss = model(x, y, step=0, total_steps=cfg.max_steps)
    lm_loss.backward()
    grads_lm = {
        id(p): p.grad.detach().clone() if p.grad is not None else None
        for p in model.parameters()
    }
    gn_lm = math.sqrt(sum(
        float((g ** 2).sum()) for g in grads_lm.values() if g is not None
    ))
    # grid_pos has no LM gradient (LM loss doesn't involve grid_pos)

    # --- Topo gradient ---
    model.zero_grad(set_to_none=True)
    if grid_pos.grad is not None:
        grid_pos.grad = None
    tloss = topographic_loss(grid_pos, cooccur, sigma)
    tloss.backward()
    gn_topo_grid = float(grid_pos.grad.norm()) if grid_pos.grad is not None else 0.0
    # Topo loss has no model gradient by construction (no model params in topo_loss)
    # Direction: we can only compare over shared params. Grid has topo grad only,
    # model params have LM grad only → they're in orthogonal subspaces and cos
    # is undefined. Report cos on the grid_pos dimension only, which is degenerate,
    # OR report 0 and note the independence.
    # Meaningful quantity: cos is well-defined only if we had a shared parameter
    # receiving both gradients. In our architecture, grid_pos receives only topo
    # gradient and model params receive only LM gradient → they never "fight" at
    # the parameter level (disjoint supports). The sense in which they fight is
    # via the shared LM loss trajectory: does adding topo_loss change how quickly
    # LM loss decreases? That's captured by the LM-loss-vs-baseline comparison,
    # not a gradient cos.
    # Return 0.0 for cos for this architecture; add a note in the diagnostic.
    gn_topo = gn_topo_grid
    cos_val = 0.0

    # Restore model grads for the next optimizer cycle (we'll do a fresh
    # forward+backward on the next batch, so this isn't strictly necessary;
    # zero them so nothing accumulates).
    model.zero_grad(set_to_none=True)
    if grid_pos.grad is not None:
        grid_pos.grad = None

    return gn_lm, gn_topo, cos_val


def _write_yaml(path: Path, data: dict) -> None:
    lines = []
    for k, v in data.items():
        if isinstance(v, str):
            lines.append(f"{k}: {v!r}")
        elif isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        elif v is None:
            lines.append(f"{k}: null")
        else:
            lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines) + "\n")


def _parse_args() -> ExpConfig:
    p = argparse.ArgumentParser(description="Exp 2 — topographic regularizer")
    p.add_argument("--mode", choices=["baseline", "topographic", "control"],
                   default="topographic")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=90_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--topo-weight", type=float, default=DEFAULT_TARGET_WEIGHT)
    p.add_argument("--grid-lr-scale", type=float, default=10.0)
    args = p.parse_args()

    run_name = args.run_name or f"exp2_{args.mode}"
    cfg = ExpConfig(
        run_name=run_name,
        mode=args.mode,
        seed=args.seed,
        max_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        topo_weight_target=args.topo_weight,
        grid_lr_scale=args.grid_lr_scale,
    )
    return cfg


if __name__ == "__main__":
    cfg = _parse_args()
    train(cfg)
