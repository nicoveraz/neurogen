"""Experiment 0 — Substrate check.

Baseline NeuroGen training with the checkpoint cadence that downstream
experiments require:

  - Full checkpoints (model + optimizer + RNG + step): every 2K steps, rotate
    keep-last-3, for resumability.
  - Trajectory snapshots (token embedding matrix only, .npy): every 500 steps
    for the first 10K steps, then every 1000 steps. Kept forever.

Output layout:

  runs/exp0_baseline/
    config.yaml
    logs/train.jsonl
    full_ckpts/step_XXXXXX.pt       # rotated
    snapshots/wte_step_XXXXXX.npy   # all kept

Usage:
  uv run python -m autoresearch.exp0_train --steps 100000 --seed 42
  uv run python -m autoresearch.exp0_train --steps 200 --smoke    # smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Make the repo root importable so we can reuse train_r4's model definition.
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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class ExpConfig:
    run_name: str = "exp0_baseline"
    arch: str = "baseline"
    seed: int = 42
    max_steps: int = 100_000
    batch_size: int = 32
    lr: float = 2e-3
    weight_decay: float = 0.05
    warmup: int = 200
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    # checkpoint cadence
    snapshot_dense_every: int = 500     # first snapshot_dense_until steps
    snapshot_dense_until: int = 10_000
    snapshot_sparse_every: int = 1000   # after that
    full_ckpt_every: int = 2000
    full_ckpt_keep: int = 3
    # eval cadence
    eval_every: int = 1000

    def runs_dir(self) -> Path:
        return REPO_ROOT / "runs" / self.run_name


# ---------------------------------------------------------------------------
# Cadence helpers
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
def save_snapshot(model: GPT, snap_dir: Path, step: int) -> Path:
    wte = model.wte.weight.detach().to("cpu").float().numpy()
    path = snap_dir / f"wte_step_{step:07d}.npy"
    np.save(path, wte)
    return path


def save_full_ckpt(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    full_dir: Path,
    cfg: ExpConfig,
    rng_state: dict,
) -> Path:
    path = full_dir / f"step_{step:07d}.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": rng_state,
            "config": cfg.__dict__,
        },
        path,
    )
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
# Model init (mirrors train_r4.train())
# ---------------------------------------------------------------------------
def build_model(cfg: ExpConfig, device: str) -> GPT:
    arch_cfg = ARCHS.get(cfg.arch, {})
    model = GPT(
        VOCAB_SIZE,
        MAX_SEQ_LEN,
        DEPTH,
        N_HEADS,
        N_KV_HEADS,
        CHANNELS,
        arch_cfg=arch_cfg,
    ).to(device)

    # Block-diagonal CA init (matches train_r4 default)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2 and not any(s in name for s in ("wte", "lm_head", "ve_gate")):
                if min(p.shape) >= 8:
                    nn.init.xavier_uniform_(p)
                    ca = block_diagonal_init(
                        p.shape,
                        n_blocks=min(4, min(p.shape)),
                        target_std=p.std().item() * 0.05,
                    )
                    p.data.add_(ca.to(p.device))
    return model


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(cfg: ExpConfig) -> None:
    device = get_device()
    run_dir = cfg.runs_dir()
    full_dir = run_dir / "full_ckpts"
    snap_dir = run_dir / "snapshots"
    log_dir = run_dir / "logs"
    for d in (full_dir, snap_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Write config.yaml (hand-rolled; avoid adding a yaml dep)
    _write_yaml(run_dir / "config.yaml", cfg.__dict__ | {"device": device})

    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    print(f"[exp0] device={device} run_dir={run_dir}", flush=True)

    train_data = load_data("train")
    val_data = load_data("val")
    print(
        f"[exp0] train_tokens={len(train_data):,}  val_tokens={len(val_data):,}",
        flush=True,
    )

    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[exp0] params={n_params:,} (~{n_params/1e6:.2f}M)", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    min_lr = cfg.lr * cfg.min_lr_ratio

    log_fp = open(log_dir / "train.jsonl", "a", buffering=1)
    model.train()

    # Save step-0 snapshot (initialization state).
    save_snapshot(model, snap_dir, step=0)

    t0 = time.time()
    last_log_loss = None

    for step in range(1, cfg.max_steps + 1):
        lr = get_lr(step, cfg.warmup, cfg.max_steps, cfg.lr, min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, cfg.batch_size, MAX_SEQ_LEN, device)
        _, loss = model(x, y, step=step, total_steps=cfg.max_steps)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        optimizer.step()

        last_log_loss = loss.item()

        # Periodic cheap log line
        if step % 100 == 0 or step == 1:
            elapsed = time.time() - t0
            print(
                f"[exp0] step={step:>7d} loss={last_log_loss:.4f} "
                f"lr={lr:.2e} grad={gnorm:.3f} elapsed={elapsed:.0f}s",
                flush=True,
            )
            log_fp.write(
                json.dumps(
                    {
                        "kind": "step",
                        "step": step,
                        "loss": round(last_log_loss, 6),
                        "lr": lr,
                        "grad_norm": round(gnorm, 6),
                        "elapsed_s": round(elapsed, 2),
                    }
                )
                + "\n"
            )

        # Snapshot token embeddings
        if should_snapshot(step, cfg):
            snap_path = save_snapshot(model, snap_dir, step)
            log_fp.write(
                json.dumps(
                    {"kind": "snapshot", "step": step, "path": str(snap_path.name)}
                )
                + "\n"
            )

        # Validation
        if step % cfg.eval_every == 0:
            vbpb = evaluate_val_bpb(
                model, val_data, cfg.batch_size, MAX_SEQ_LEN, device
            )
            print(f"[exp0] step={step:>7d} val_bpb={vbpb:.4f}", flush=True)
            log_fp.write(
                json.dumps({"kind": "eval", "step": step, "val_bpb": round(vbpb, 6)})
                + "\n"
            )

        # Full checkpoint (rotated)
        if should_full_ckpt(step, cfg):
            rng_state = capture_rng_state(device)
            ckpt_path = save_full_ckpt(model, optimizer, step, full_dir, cfg, rng_state)
            rotate_full_ckpts(full_dir, cfg.full_ckpt_keep)
            log_fp.write(
                json.dumps(
                    {"kind": "full_ckpt", "step": step, "path": str(ckpt_path.name)}
                )
                + "\n"
            )

    # Final snapshot + ckpt regardless of cadence
    save_snapshot(model, snap_dir, step=cfg.max_steps)
    rng_state = capture_rng_state(device)
    save_full_ckpt(model, optimizer, cfg.max_steps, full_dir, cfg, rng_state)
    rotate_full_ckpts(full_dir, cfg.full_ckpt_keep)

    final_vbpb = evaluate_val_bpb(model, val_data, cfg.batch_size, MAX_SEQ_LEN, device)
    total_time = time.time() - t0
    summary = {
        "kind": "summary",
        "total_steps": cfg.max_steps,
        "final_val_bpb": round(final_vbpb, 6),
        "wall_time_s": round(total_time, 1),
        "params": n_params,
    }
    print(f"[exp0] DONE {summary}", flush=True)
    log_fp.write(json.dumps(summary) + "\n")
    log_fp.close()


# ---------------------------------------------------------------------------
# Minimal YAML writer (dep-free)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> ExpConfig:
    p = argparse.ArgumentParser(description="Exp 0 — substrate check")
    p.add_argument("--run-name", default="exp0_baseline")
    p.add_argument("--arch", default="baseline", choices=list(ARCHS.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 200 steps, snapshots every 50, override run name")
    args = p.parse_args()

    cfg = ExpConfig(
        run_name=args.run_name,
        arch=args.arch,
        seed=args.seed,
        max_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    if args.smoke:
        cfg.run_name = args.run_name if args.run_name != "exp0_baseline" else "exp0_smoke"
        cfg.max_steps = min(args.steps, 200) if args.steps != 100_000 else 200
        cfg.snapshot_dense_every = 50
        cfg.snapshot_dense_until = 200
        cfg.snapshot_sparse_every = 50
        cfg.full_ckpt_every = 100
        cfg.eval_every = 100
    return cfg


if __name__ == "__main__":
    cfg = _parse_args()
    train(cfg)
