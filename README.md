# NeuroGen 🧬🧠

**Can a cellular automaton grow better neural network weights than random initialization?**

An [autoresearch](https://github.com/karpathy/autoresearch) project: an AI agent autonomously experiments with cellular automata rules for transformer weight initialization and live training dynamics. Built on the [nanochat](https://github.com/karpathy/nanochat) pattern, adapted for Apple Silicon.

## The Idea

The brain doesn't learn from random initial conditions. Genetic programs grow structured architecture *before* learning begins. NeuroGen tests whether the same principle — small local rules producing structured weights — can improve transformer training.

```
Tiny CA Rule Set (genome)  →  Iterate N steps  →  Structured Weights  →  Better Training?
    ~1K params                                      ~10M params
```

Three research axes:
1. **CA Initialization** — replace random init with CA-developed weights
2. **Live CA** — CA rules running alongside gradient descent during training
3. **Meta-learned genomes** — evolve the CA rules through the autoresearch loop

## How It Works

Following Karpathy's autoresearch pattern: the human writes `program.md`, the AI agent modifies `train.py`, runs 2-minute experiments, keeps improvements, discards failures, and repeats.

```
program.md  →  agent reads instructions
                   ↓
              agent modifies train.py (adds CA rule)
                   ↓
              uv run train.py (2 min experiment)
                   ↓
              val_bpb improved?
              ├── yes → git commit, new baseline
              └── no  → git reset, try something else
```

## Quick Start

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download data and prepare shards
uv run prepare.py

# Run baseline training (~2 min)
uv run train.py

# Start autoresearch (point your agent at program.md)
```

## Benchmarking & Evaluation

NeuroGen includes three evaluation tools beyond the autoresearch loop:

```bash
# Establish xavier baseline (run once per config)
uv run benchmark.py --baseline --seeds 5

# Multi-seed statistical comparison of init methods
uv run benchmark.py --compare "default,xavier,grid_ca,modular_ca" --seeds 5

# Add output quality metrics (repetition, diversity, self-perplexity)
uv run benchmark.py --compare "default,xavier" --seeds 3 --quality

# Evaluate generation quality with sample outputs
uv run evaluate_quality.py --methods "default,xavier,grid_ca"

# Quality convergence over training (the money plot)
uv run evaluate_quality.py --quality-over-time --methods "default,grid_ca"
```

**Benchmark reports** include: val_bpb with std, vs-baseline percentage, paired t-tests, init diagnostics (head diversity, block-diagonal ratio, layer similarity), FLOPs-matched comparison, and optional quality metrics. All saved to `outputs/`.

## Project Structure

```
program.md           — agent research instructions (human writes this)
prepare.py           — data prep + evaluation (fixed, do not modify)
train.py             — model + CA + training loop (agent modifies this)
ca_rules.py          — CA rule library (agent can import and modify)
benchmark.py         — multi-seed statistical benchmarks
evaluate_quality.py  — output quality evaluation (repetition, diversity, coherence)
evaluate_core.py     — DCLM CORE evaluation stub (Level 2, optional)
analysis.ipynb       — results visualization
results.tsv          — experiment log
NEUROGEN.md          — full research reference
outputs/             — benchmark reports, quality reports, CSVs
```

## Hardware

Designed for MacBook Pro M1 Pro (MPS backend). Also works on CUDA GPUs and CPU. Default config: depth 4, ~3.4M params, ~2 min per experiment, ~30 experiments/hour on M1 Pro.

## References & Prior Art

NeuroGen builds on and extends several lines of research. See `NEUROGEN.md` for detailed comparison of how this project differs from each.

- [HyperNCA](https://arxiv.org/abs/2204.11674) — Najarro & Risi, 2022. **Most directly related.** NCA growing RL policy weights via CMA-ES. NeuroGen extends this to transformers, adds live CA, and uses functional neuroscience principles.
- [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) — Mordvintsev et al., 2020. Foundational NCA architecture.
- [Weight Agnostic Neural Networks](https://arxiv.org/abs/1906.04358) — Gaier & Ha, 2019. Architecture encodes inductive bias without weight training.
- [HyperNetworks](https://arxiv.org/abs/1609.09106) — Ha et al., 2017. Small networks generating weights for larger networks.
- [nanochat](https://github.com/karpathy/nanochat) / [autoresearch](https://github.com/karpathy/autoresearch) — Karpathy, 2025-2026. Training harness and autonomous experiment loop.

## License

MIT
