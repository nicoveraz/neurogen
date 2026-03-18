# NeuroGen

**Can a cellular automaton grow better neural network weights than random initialization?**

The brain doesn't start from random noise — genetic programs grow structured architecture *before* learning begins. NeuroGen tests whether the same principle can improve transformer training.

## The Idea

```
Tiny CA Rule Set (genome)  -->  Iterate N steps  -->  Structured Weight Matrices  -->  Faster/Better Training?
    ~10K params                                          ~10M params
```

A small cellular automaton acts as a "developmental program" that grows weight matrices for a GPT-style transformer. If the CA produces useful structure (modularity, spectral properties, locality), the model should start closer to a good solution and train more efficiently.

## Quick Start

```bash
# Install
pip install -e .

# Prepare data (downloads Shakespeare)
python prepare.py

# Run a training experiment (2-minute time budget)
python train.py

# Run library-based training
python scripts/train.py --init xavier_normal --steps 5000

# Run tests
pytest tests/ -v -m "not slow"
```

## Autoresearch

NeuroGen uses a [Karpathy-style autoresearch](https://github.com/karpathy/autoresearch) approach: an AI agent autonomously conducts experiments by modifying `train.py`, running it, and iterating based on results.

```
prepare.py      Fixed evaluation harness (DO NOT MODIFY)
train.py        Model + init + training loop (AI modifies this)
program.md      Instructions for the AI researcher
results.tsv     Experiment log (append-only)
```

The AI researcher reads `program.md`, modifies `train.py`, runs experiments (2 min each), logs results, and uses git keep/discard to track progress. See [NEUROGEN_AUTORESEARCH.md](NEUROGEN_AUTORESEARCH.md) for details.

## Project Structure

- `prepare.py` / `train.py` / `program.md` — Autoresearch harness
- `neurogen/model/` — Minimal GPT (Karpathy-style)
- `neurogen/ca/` — Cellular automata weight development engine (5 variants)
- `neurogen/ca/live/` — Live CA: CA operating during training alongside gradient descent
- `neurogen/baselines/` — Standard initialization methods for comparison
- `neurogen/training/` — Training loop, live CA trainer, meta-learning
- `neurogen/analysis/` — Weight analysis and visualization
- `scripts/` — CLI tools

## Research Phases

| Phase | Question | Status |
|-------|----------|--------|
| 1 | How do standard inits compare? | In progress |
| 2 | Can simple CAs produce useful weight initializations? | Planned |
| 3 | Do structured CAs (block-diagonal, spectral) help more? | Planned |
| 4 | Does live CA during training improve over init-only? | Planned |
| 5 | Can we meta-learn the CA rule? | Planned |

## Testing

```bash
# Run fast tests (<10s)
pytest tests/ -v -m "not slow"

# Run all tests (200+)
pytest tests/ -v
```

## Documentation

- [NEUROGEN_PROJECT.md](NEUROGEN_PROJECT.md) — Full research plan, architecture, and implementation guide
- [NEUROGEN_AUTORESEARCH.md](NEUROGEN_AUTORESEARCH.md) — Autoresearch system specification
- [NEUROGEN_LIVE_CA.md](NEUROGEN_LIVE_CA.md) — Live CA specification
- [NEUROGEN_EXPLORATION.md](NEUROGEN_EXPLORATION.md) — CA search strategy
- [NEUROGEN_TESTING.md](NEUROGEN_TESTING.md) — Test and benchmark specs

## License

MIT
