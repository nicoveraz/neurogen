# NeuroGen 🧬🧠

**Can a cellular automaton grow better neural network weights than random initialization?**

The brain doesn't start from random noise — genetic programs grow structured architecture *before* learning begins. NeuroGen tests whether the same principle can improve transformer training.

## The Idea

```
Tiny CA Rule Set (genome)  ──▶  Iterate N steps  ──▶  Structured Weight Matrices  ──▶  Faster/Better Training?
    ~10K params                                          ~10M params
```

A small cellular automaton acts as a "developmental program" that grows weight matrices for a GPT-style transformer. If the CA produces useful structure (modularity, spectral properties, locality), the model should start closer to a good solution and train more efficiently.

## Quick Start

```bash
# Install
pip install -e .

# Train baseline (standard random init)
python scripts/train.py --init xavier_normal --steps 5000

# Train with CA initialization
python scripts/train.py --init grid_ca --steps 5000

# Run full baseline comparison
python scripts/run_phase.py --phase 1

# Run everything
python scripts/run_all.py
```

## Project Structure

- `neurogen/model/` — Minimal GPT (Karpathy-style)
- `neurogen/ca/` — Cellular automata weight development engine (5 variants)
- `neurogen/baselines/` — Standard initialization methods for comparison
- `neurogen/training/` — Training loop + meta-learning
- `neurogen/analysis/` — Weight analysis and visualization
- `research/` — Auto-research engine (YAML-driven experiments, auto-reports)
- `scripts/` — CLI tools

## Research Phases

| Phase | Question | Status |
|-------|----------|--------|
| 1 | How do standard inits compare on microGPT? | 🔲 |
| 2 | Can CAs produce valid weight tensors? | 🔲 |
| 3 | Does CA *structure* alone help (random CA vs random init)? | 🔲 |
| 4 | Do hand-designed CA priors help? | 🔲 |
| 5 | Can we meta-learn the CA genome? | 🔲 |
| 6 | Does co-evolution (CA active during training) help? | 🔲 |
| 7 | Ablations: CA steps, genome size, neighborhoods | 🔲 |
| 8 | Does a small-model CA transfer to larger models? | 🔲 |

## Testing & Benchmarks

```bash
# Run fast tests (<10s)
pytest tests/ -v -m "not slow"

# Run all tests
pytest tests/ -v

# Quick benchmark suite (~5 min CPU)
python scripts/run_benchmark.py --suite quick

# Standard benchmark suite (~2-4h GPU)
python scripts/run_benchmark.py --suite standard

# Single benchmark
python scripts/run_benchmark.py --benchmark bm2
```

8 benchmark protocols (BM1-BM8) measure init quality, convergence speed, compute efficiency, weight structure, training dynamics, generation quality, meta-learning effectiveness, and scale transfer. See [NEUROGEN_TESTING.md](NEUROGEN_TESTING.md) for full specifications.

## Full Specification

See [NEUROGEN_PROJECT.md](NEUROGEN_PROJECT.md) for the complete research plan, architecture details, and implementation guide. See [NEUROGEN_EXPLORATION.md](NEUROGEN_EXPLORATION.md) for the three-stage CA search strategy, and [NEUROGEN_TESTING.md](NEUROGEN_TESTING.md) for test and benchmark specs.

## License

MIT
