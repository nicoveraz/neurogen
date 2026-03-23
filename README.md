# NeuroGen

**Developmental constraints improve transformer training.**

An [autoresearch](https://github.com/karpathy/autoresearch) project that discovered layer-wise attention window growth — forcing early layers to attend locally before opening to global attention — produces statistically significant improvements in transformer language models.

## Key Finding

Quartic attention window growth (`window_power_4.0`) improves converged val_bpb by **1.5%** over standard full attention (p=0.001, Cohen's d=2.05, 5 seeds, 20k steps).

```
Layer windows at depth 4:  [8, 10, 65, 256]
- Layers 0-2: restricted to local context (8-65 tokens)
- Layer 3: full attention (256 tokens)
```

The advantage is a **constant offset** — it persists from early training through convergence, meaning the constraint produces a genuinely better solution, not just faster convergence.

![Learning Curves](charts/learning_curves.svg)

![Final Performance](charts/final_performance.svg)

![Window Schedule](charts/window_schedule.svg)

```
Validation results (20,000 steps, 5 seeds each):

config                  mean bpb   std      vs baseline   p-value   Cohen's d
baseline                0.9002     0.0075   —             —         —
window_power_4.0        0.8866     0.0056   +1.5%         0.001     2.05
window_quadratic        0.8911     0.0048   +1.0%         0.022     1.45
window_quad_induction   0.8899     0.0041   +1.1%         0.007     1.69

All 5 seeds of every window variant beat the baseline mean.
Throughput is identical across all architectures (4.8 steps/sec).
```

## How It Works

A standard transformer uses full attention at every layer — each token can attend to all previous tokens from layer 1. This is like a brain where every neuron connects to every other neuron from birth.

Real brains develop differently: local receptive fields form first, then global connectivity builds on top. NeuroGen embeds this developmental principle into the transformer architecture by restricting each layer's attention window based on depth:

```python
def compute_window(layer_idx, n_layers, seq_len, exponent=4.0):
    progress = (layer_idx + 1) / n_layers
    return int(8 + progress ** exponent * (seq_len - 8))
```

The window function was found through systematic search across power functions (exponents 0.5-5.0), sigmoid curves, logarithmic, exponential, and Fibonacci schedules. The optimal exponent is 3-4 at depth 4 — the model wants early layers extremely local.

## Research Journey

This project ran 100+ autonomous experiments across 4 rounds:

- **Round 1** (50 experiments): CA weight initialization gives ~0.8% improvement. Live CA fails on MPS due to overhead.
- **Round 2** (40 experiments): CA init advantage holds at 30min training (constant offset, not head start).
- **Round 4** (68 experiments): Tested 26 architecture variants including CA modulation channels, embryogenic CA, universal circuit pre-wiring, token vitality, sleep consolidation. Most failed. Developmental attention windows emerged as the clear winner.
- **Validation** (20 experiments): Confirmed at 20k steps with 5 seeds. Statistically significant. Throughput-neutral.

### What Didn't Work
- CA modulation channels (model collapse)
- Token vitality / cell death dynamics (model collapse)
- Sleep consolidation (overhead outweighed benefit)
- Pre-wiring known circuits alone (induction heads, layer roles — gradient descent prefers organic discovery)
- Live CA during training (any per-step overhead hurts at small scale)
- Embryogenic activity-dependent CA (marginal gains, high overhead)

### What Did Work
- **Developmental attention windows** (quartic growth, +1.5%)
- **Combining constraints with scaffolds** (window + induction pre-wiring, +1.1%)
- **Block-diagonal CA init** (+0.6% at 10min, constant offset)

## Quick Start

```bash
# Install
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Download data
uv run prepare.py

# Train baseline
uv run train_r4.py --arch baseline --minutes 40 --seed 42

# Train with developmental windows (best result)
uv run train_r4.py --arch window_power_4.0 --minutes 40 --seed 42

# Validation run (step-budget, full convergence)
uv run validate.py --arch window_power_4.0 --steps 20000 --seed 42

# Quality evaluation
uv run evaluate_quality.py --live --arch baseline window_power_4.0 --seed 42 --minutes 40
```

## Project Structure

```
prepare.py          — data prep + evaluation (fixed)
train.py            — Round 1-2 training script (depth 2)
train_r4.py         — Round 4 training with 26 architecture variants
validate.py         — step-budget convergence runs with diagnostics
evaluate_quality.py — generation quality metrics
ca_rules.py         — CA rule library
program.md          — autoresearch instructions
results.tsv         — experiment log
validation_results/ — convergence run data (JSON, 20k steps × 5 seeds)
outputs/            — experiment logs
```

## Hardware

Designed for Apple Silicon (MPS). Also works on CUDA and CPU. Default: depth 4, channels 256, ~3.4M params, ~4.8 steps/sec on M1 Pro.

## References

- [nanochat](https://github.com/karpathy/nanochat) / [autoresearch](https://github.com/karpathy/autoresearch) — Karpathy. Training harness and experiment loop.
- [HyperNCA](https://arxiv.org/abs/2204.11674) — Najarro & Risi, 2022. NCA growing RL policy weights.
- [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) — Mordvintsev et al., 2020.
- Olsson et al., 2022 — In-context learning and induction heads.

## License

MIT
