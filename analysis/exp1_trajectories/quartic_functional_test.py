"""Functional test — does quartic's emergent co-occurrence organization
actually help predictions on positions where co-occurrence is informative?

Test design:
  1. From training co-occurrence (window=5), derive an empirical next-byte
     distribution P(next | prev). For each of 256 possible prev bytes,
     compute H_bigram(prev) = H[P(next | prev)] (Shannon entropy in bits).
     Low H → prev strongly constrains next (e.g., 'q' → 'u' dominant).
     High H → prev weakly constrains next (e.g., ' ' → many letters).
  2. Load final checkpoints of baseline and quartic models.
  3. Run both on 500 batches of val data (~4M val tokens) with teacher
     forcing; compute per-token NLL.
  4. For each val position, record (prev_byte, per_token_NLL_baseline,
     per_token_NLL_quartic). Bucket by H_bigram(prev).
  5. Report per-bucket mean NLL for each model, and the delta.

Prediction: if quartic's co-occurrence-organized embeddings functionally
help next-byte prediction (not just correlate with it), the advantage
(NLL_baseline − NLL_quartic) should be LARGER in low-entropy buckets.
Uniform advantage across buckets → topographic-like structure doesn't
functionally matter, it's just correlated with what the model already
learned.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prepare import (  # noqa: E402
    MAX_SEQ_LEN, VOCAB_SIZE, get_batch, get_device, load_data,
)
from train_r4 import ARCHS, CHANNELS, DEPTH, GPT, N_HEADS, N_KV_HEADS  # noqa: E402

COOCCUR_PATH = REPO_ROOT / "runs" / "exp2_cooccur" / "cooccur_w5.npy"
OUT_DIR = REPO_ROOT / "analysis" / "exp1_trajectories"
REPORT_DIR = REPO_ROOT / "reports" / "exp_quartic"


def load_model(ckpt_path: Path, arch: str, device: str) -> GPT:
    arch_cfg = ARCHS.get(arch, {})
    model = GPT(
        VOCAB_SIZE, MAX_SEQ_LEN, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS,
        arch_cfg=arch_cfg,
    ).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict") or ck["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def compute_bigram_entropy(cooccur_raw: np.ndarray) -> np.ndarray:
    """For each prev byte i, compute H[P(next | i)] in bits.
    Uses the log-normalized cooccur matrix as the basis for P(next|prev).
    """
    # Per-row normalize to get P(next | prev)
    row_sum = cooccur_raw.sum(axis=1, keepdims=True)
    row_sum_safe = np.where(row_sum > 0, row_sum, 1)
    P = cooccur_raw / row_sum_safe
    # Entropy per row (in bits)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -(P * np.log2(np.where(P > 0, P, 1))).sum(axis=1)
    H[row_sum.squeeze() == 0] = 0.0
    return H


@torch.no_grad()
def eval_model_per_token(
    model: GPT, val_data: torch.Tensor, n_batches: int,
    batch_size: int, block_size: int, device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (prev_bytes, nll) flat arrays, one per token."""
    all_prev = []
    all_nll = []
    for _ in range(n_batches):
        x, y = get_batch(val_data, batch_size, block_size, device)
        logits, _ = model(x)
        logp = F.log_softmax(logits.float(), dim=-1)  # (B, T, V)
        # Per-token NLL = -logp[y]. But we want "natural log" for nats.
        nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # (B, T), nats
        # prev_byte for position t is x[:, t]. For t=0, there's no prev
        # within this batch, but we're conditioning on x[:, t] to predict y[:, t]
        # = x[:, t+1] (teacher forcing). So "prev_byte" = x[:, t].
        prev = x  # (B, T)
        all_prev.append(prev.flatten().cpu().numpy())
        all_nll.append(nll.flatten().cpu().numpy())
    return np.concatenate(all_prev), np.concatenate(all_nll)


def main() -> None:
    device = get_device()
    print(f"device: {device}")

    cooccur_raw = np.load(REPO_ROOT / "runs" / "exp2_cooccur" / "cooccur_w5_raw.npy")
    H_bigram = compute_bigram_entropy(cooccur_raw)
    print(f"Bigram entropy range: {H_bigram.min():.3f} – {H_bigram.max():.3f} bits "
          f"(median {np.median(H_bigram):.3f})")

    val_data = load_data("val")
    print(f"val tokens: {len(val_data):,}")

    baseline_ckpt = REPO_ROOT / "runs" / "exp0_baseline" / "full_ckpts" / "step_0100000.pt"
    quartic_ckpt = REPO_ROOT / "runs" / "exp_quartic" / "full_ckpts" / "step_0100000.pt"
    print("Loading baseline...")
    m_base = load_model(baseline_ckpt, "baseline", device)
    print("Loading quartic...")
    m_q = load_model(quartic_ckpt, "window_power_4.0", device)

    # Reset RNG so both models see identical batches
    batch_size = 32
    n_batches = 200  # ~1.6M tokens — enough for reliable per-bucket averages
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"\nEvaluating baseline on {n_batches} batches...")
    prev_b, nll_b = eval_model_per_token(m_base, val_data, n_batches, batch_size,
                                          MAX_SEQ_LEN, device)
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"Evaluating quartic on {n_batches} batches...")
    prev_q, nll_q = eval_model_per_token(m_q, val_data, n_batches, batch_size,
                                          MAX_SEQ_LEN, device)

    # Sanity: same batches
    assert np.array_equal(prev_b, prev_q), "batches differ between models"

    prev_bytes = prev_b
    nll_base = nll_b
    nll_quart = nll_q
    n_tokens = len(prev_bytes)
    print(f"Total tokens evaluated: {n_tokens:,}")
    print(f"Mean NLL: baseline={nll_base.mean():.4f}  quartic={nll_quart.mean():.4f}  "
          f"bpb delta={(nll_quart.mean()-nll_base.mean())/math.log(2):+.4f}")

    # Bucket by bigram entropy of prev byte
    H_per_tok = H_bigram[prev_bytes]
    # Drop tokens where H is 0 (prev byte never seen in training)
    mask = H_per_tok > 0
    H_per_tok = H_per_tok[mask]
    nll_base = nll_base[mask]
    nll_quart = nll_quart[mask]
    prev_bytes = prev_bytes[mask]
    print(f"  after filtering prev-byte-absent-in-train: {len(H_per_tok):,} tokens")

    # Quantile buckets
    n_buckets = 5
    qs = np.quantile(H_per_tok, np.linspace(0, 1, n_buckets + 1))
    print(f"\nBucket entropy boundaries (bits): {[round(q, 3) for q in qs]}")
    print(f"\n{'bucket':<8} {'H range':<16} {'n':>9} "
          f"{'nll_base':>9} {'nll_quart':>10} {'delta(q-b)':>11} "
          f"{'delta(bpb)':>11}")
    print("-" * 80)

    results = []
    for k in range(n_buckets):
        lo, hi = qs[k], qs[k + 1]
        m = (H_per_tok >= lo) & (H_per_tok < hi if k < n_buckets - 1 else H_per_tok <= hi)
        if m.sum() == 0:
            continue
        nb = nll_base[m].mean()
        nq = nll_quart[m].mean()
        delta = nq - nb
        delta_bpb = delta / math.log(2)
        print(f"{k:<8} {lo:5.2f}–{hi:5.2f}   {m.sum():>9d} "
              f"{nb:>9.4f} {nq:>10.4f} {delta:>+11.4f} {delta_bpb:>+11.4f}")
        results.append({
            "bucket": k,
            "entropy_range_bits": [float(lo), float(hi)],
            "n_tokens": int(m.sum()),
            "nll_baseline": float(nb),
            "nll_quartic": float(nq),
            "delta_nats": float(delta),
            "delta_bpb": float(delta_bpb),
        })

    # Also compute: per-prev-byte mean NLL difference. Which specific bytes
    # does quartic help on?
    print(f"\n=== Per-prev-byte NLL delta (top 15 bytes where quartic helps most) ===")
    per_byte = {}
    for b in range(256):
        m = prev_bytes == b
        if m.sum() < 100:
            continue
        nb = nll_base[m].mean()
        nq = nll_quart[m].mean()
        per_byte[b] = {
            "n": int(m.sum()),
            "nll_base": float(nb),
            "nll_quart": float(nq),
            "delta": float(nq - nb),
            "H_bigram": float(H_bigram[b]),
        }
    sorted_help = sorted(per_byte.items(), key=lambda kv: kv[1]["delta"])
    for b, d in sorted_help[:15]:
        bc = repr(chr(b)) if 32 <= b < 127 else f"\\x{b:02x}"
        print(f"  prev={bc:<6} n={d['n']:>6d}  nll_b={d['nll_base']:.3f}  "
              f"nll_q={d['nll_quart']:.3f}  Δ={d['delta']:+.4f}  "
              f"H_bigram={d['H_bigram']:.2f}")

    print(f"\n=== Per-prev-byte NLL delta (top 10 bytes where quartic is WORSE) ===")
    for b, d in sorted_help[-10:][::-1]:
        bc = repr(chr(b)) if 32 <= b < 127 else f"\\x{b:02x}"
        print(f"  prev={bc:<6} n={d['n']:>6d}  nll_b={d['nll_base']:.3f}  "
              f"nll_q={d['nll_quart']:.3f}  Δ={d['delta']:+.4f}  "
              f"H_bigram={d['H_bigram']:.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "quartic_functional_test.json").write_text(json.dumps({
        "n_tokens_evaluated": int(n_tokens),
        "mean_nll_baseline": float(nll_base.mean()),
        "mean_nll_quartic": float(nll_quart.mean()),
        "buckets": results,
        "per_prev_byte": {chr(b) if 32<=b<127 else f"\\x{b:02x}": v for b, v in per_byte.items()},
    }, indent=2))
    print(f"\nSaved → {OUT_DIR/'quartic_functional_test.json'}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        bucket_mid = [(r["entropy_range_bits"][0] + r["entropy_range_bits"][1]) / 2
                      for r in results]
        nll_b = [r["nll_baseline"] for r in results]
        nll_q = [r["nll_quartic"] for r in results]
        ax1.plot(bucket_mid, nll_b, marker="o", label="baseline", color="C0")
        ax1.plot(bucket_mid, nll_q, marker="s", label="quartic", color="C3")
        ax1.set_xlabel("bigram entropy of prev byte (bits)")
        ax1.set_ylabel("mean NLL (nats)")
        ax1.set_title("NLL by bigram-predictability bucket")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        deltas = [r["delta_bpb"] for r in results]
        ax2.bar(range(len(results)), deltas, color=["C3" if d < 0 else "C0" for d in deltas])
        ax2.axhline(0, color="black", lw=0.7)
        ax2.set_xlabel("bucket (0 = most constrained prev, 4 = least)")
        ax2.set_ylabel("Δ bpb (quartic − baseline)")
        ax2.set_title("Per-bucket advantage (negative = quartic better)")
        ax2.set_xticks(range(len(results)))
        ax2.set_xticklabels([f"{r['entropy_range_bits'][0]:.1f}–{r['entropy_range_bits'][1]:.1f}"
                             for r in results], rotation=30)
        ax2.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            "If the emergent organization is functionally useful, quartic's advantage\n"
            "should concentrate in low-entropy (left) buckets where co-occurrence is informative.",
            fontsize=10, y=1.05,
        )
        fig.tight_layout()
        out = REPORT_DIR / "functional_test.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"Plot → {out}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
