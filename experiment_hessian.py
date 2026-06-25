"""Experiment: loss-landscape sharpness of windowed vs full-attention models.

Mechanism question. Experiment 6 (gradient direction stability) is a weak proxy
for landscape flatness. This measures the Hessian curvature directly on the
committed matched-seed checkpoints:

  * top eigenvalue lambda_max  (power iteration on Hessian-vector products)
  * trace Tr(H)                (Hutchinson estimator with Rademacher probes)
  * mean curvature Tr(H)/n_params

Both are the standard estimators (cf. PyHessian, Yao et al. 2020): a
Hessian-vector product via double backprop, power iteration for the top
eigenvalue, Hutchinson for the trace. We compute on the model's OWN loss (the
window mask is active inside forward for the quartic model), on a fixed
validation batch shared within each matched pair.

Falsifiable framing: flatter == smaller lambda_max / trace for quartic. Zhai et
al. 2023 predict the opposite (lower attention entropy -> sharper), so a flatter
quartic minimum would be a genuine surprise and a sharper one cleanly refutes
the "windows smooth the landscape" hypothesis (consistent with Exp 6).

Run on CPU (small model; avoids MPS double-backward gaps). ~a few minutes.

Usage: uv run python experiment_hessian.py
"""
import argparse
import json
import os

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from prepare import load_data, get_batch, VOCAB_SIZE, MAX_SEQ_LEN
from train_r4 import GPT, DEPTH, CHANNELS, N_HEADS, N_KV_HEADS

# The baseline path uses F.scaled_dot_product_attention, whose CPU flash kernel
# has no double-backward. Force the math backend so the Hessian (2nd-order) works.
_SDPA = [SDPBackend.MATH]

DEVICE = "cpu"  # double-backward is fragile on MPS; the 3.4M model is tiny on CPU
RESULTS_DIR = "gradient_results"

def pair_for_seed(seed):
    """Matched baseline-vs-quartic checkpoint pair for a seed (same init)."""
    return {
        "baseline": (f"checkpoints/model_baseline_{seed}.pt", {}),
        "quartic": (f"checkpoints/model_window_power_4.0_{seed}.pt", {"window": "power_4.0"}),
    }


def load_model(ckpt_path, arch_cfg):
    model = GPT(VOCAB_SIZE, MAX_SEQ_LEN, DEPTH, N_HEADS, N_KV_HEADS, CHANNELS, arch_cfg=arch_cfg)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model.to(DEVICE).eval()


def make_hvp(model, x, y):
    """Return (Hv, flat_grad, n_params): Hv(vec) gives the Hessian-vector product.

    One forward + one create_graph backward builds the graph; each Hv reuses it.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    with sdpa_kernel(_SDPA):
        _, loss = model(x, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])

    def Hv(vec):
        gv = (flat_grad * vec).sum()
        hv = torch.autograd.grad(gv, params, retain_graph=True)
        return torch.cat([h.reshape(-1) for h in hv]).detach()

    return Hv, flat_grad.detach(), n_params


def top_eigenvalue(Hv, n_params, iters=20, gen=None):
    """Largest-magnitude Hessian eigenvalue via power iteration (Rayleigh quotient)."""
    v = torch.randn(n_params, generator=gen)
    v /= v.norm()
    eig = 0.0
    for _ in range(iters):
        hv = Hv(v)
        eig = torch.dot(v, hv).item()  # Rayleigh quotient (signed)
        nrm = hv.norm()
        if nrm < 1e-12:
            break
        v = hv / nrm
    return eig


def hutchinson_trace(Hv, n_params, n_probes=12, gen=None):
    """Tr(H) via Hutchinson: E[z^T H z] with Rademacher z in {-1,+1}."""
    ests = []
    for _ in range(n_probes):
        z = torch.randint(0, 2, (n_params,), generator=gen).float() * 2 - 1
        ests.append(torch.dot(z, Hv(z)).item())
    t = torch.tensor(ests)
    return t.mean().item(), t.std(unbiased=True).item()


def _mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = (sum((v - m) ** 2 for v in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, sd


def measure_model(model, batches, power_iters, n_probes):
    """Per-batch lambda_max and trace for one model over a fixed set of batches."""
    lams, traces, gnorms, n_params = [], [], [], 0
    for x, y in batches:
        Hv, flat_grad, n_params = make_hvp(model, x, y)
        gen = torch.Generator().manual_seed(2024)
        lams.append(top_eigenvalue(Hv, n_params, iters=power_iters, gen=gen))
        gen = torch.Generator().manual_seed(7)
        tr, _ = hutchinson_trace(Hv, n_params, n_probes=n_probes, gen=gen)
        traces.append(tr)
        gnorms.append(flat_grad.norm().item())
        del Hv, flat_grad
    return lams, traces, gnorms, n_params


def analyze_pair(seed, batch_size=8, block_size=MAX_SEQ_LEN, power_iters=20,
                 n_probes=12, n_batches=5):
    cfg = pair_for_seed(seed)
    if not all(os.path.exists(p) for p, _ in cfg.values()):
        print(f"  seed {seed}: missing checkpoint(s), skipping")
        return None
    val_data = load_data("val")
    # A fixed set of batches, SHARED by both arms of the pair (matched data), so
    # the per-batch ratio q/b controls for batch-to-batch Hessian noise.
    torch.manual_seed(1000 + seed)
    batches = [get_batch(val_data, batch_size, block_size, DEVICE) for _ in range(n_batches)]

    out = {}
    per_model = {}
    for name, (ckpt, arch_cfg) in cfg.items():
        model = load_model(ckpt, arch_cfg)
        lams, traces, gnorms, n_params = measure_model(model, batches, power_iters, n_probes)
        per_model[name] = (lams, traces)
        lam_m, lam_sd = _mean_sd(lams)
        tr_m, tr_sd = _mean_sd(traces)
        out[name] = {
            "lambda_max_mean": lam_m, "lambda_max_sd": lam_sd,
            "trace_mean": tr_m, "trace_sd": tr_sd,
            "mean_curvature": tr_m / n_params,
            "grad_norm_mean": _mean_sd(gnorms)[0],
            "n_params": n_params, "n_batches": n_batches,
            "lambda_max_per_batch": lams, "trace_per_batch": traces,
        }
        print(f"  seed {seed} {name:9s}: lambda_max={lam_m:7.3f}±{lam_sd:.3f}  "
              f"trace={tr_m:8.1f}±{tr_sd:.1f}  mean_curv={tr_m/n_params:.3e}")
        del model
    # Per-batch matched ratios (q/b on the SAME batch), then mean±sd across batches.
    (bl_lam, bl_tr), (q_lam, q_tr) = per_model["baseline"], per_model["quartic"]
    lam_ratios = [q / b for q, b in zip(q_lam, bl_lam)]
    tr_ratios = [q / b for q, b in zip(q_tr, bl_tr)]
    lr_m, lr_sd = _mean_sd(lam_ratios)
    trr_m, trr_sd = _mean_sd(tr_ratios)
    out["delta"] = {
        "lambda_max_ratio_mean": lr_m, "lambda_max_ratio_sd": lr_sd,
        "trace_ratio_mean": trr_m, "trace_ratio_sd": trr_sd,
        "n_batches_quartic_flatter_trace": sum(1 for r in tr_ratios if r < 1),
        "n_batches_quartic_flatter_lambda": sum(1 for r in lam_ratios if r < 1),
    }
    print(f"  -> ratio q/b across {n_batches} batches: lambda_max {lr_m:.3f}±{lr_sd:.3f} "
          f"({out['delta']['n_batches_quartic_flatter_lambda']}/{n_batches} flatter), "
          f"trace {trr_m:.3f}±{trr_sd:.3f} "
          f"({out['delta']['n_batches_quartic_flatter_trace']}/{n_batches} flatter)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 137])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--power-iters", type=int, default=20)
    ap.add_argument("--probes", type=int, default=12)
    ap.add_argument("--n-batches", type=int, default=5)
    args = ap.parse_args()

    print("=" * 78)
    print("  EXPERIMENT: Loss-landscape sharpness (Hessian) — windowed vs full attn")
    print("=" * 78)
    print("  Flatter quartic => ratio q/b < 1. Zhai 2023 predicts sharper (ratio > 1).")

    results = {}
    for seed in args.seeds:
        print(f"\nMatched pair, seed {seed} ({args.n_batches} batches):")
        r = analyze_pair(seed, batch_size=args.batch_size, power_iters=args.power_iters,
                         n_probes=args.probes, n_batches=args.n_batches)
        if r:
            results[seed] = r

    if results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "hessian_sharpness.json")
        with open(path, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        print(f"\nSaved {path}")

        # Cross-seed summary
        print("\nSummary (quartic vs baseline, mean ratio q/b across batches; <1 = flatter):")
        for seed, r in results.items():
            d = r["delta"]
            print(f"  seed {seed}: lambda_max x{d['lambda_max_ratio_mean']:.3f}±{d['lambda_max_ratio_sd']:.3f}, "
                  f"trace x{d['trace_ratio_mean']:.3f}±{d['trace_ratio_sd']:.3f}")


if __name__ == "__main__":
    main()
