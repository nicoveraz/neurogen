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

# Download data and train tokenizer
uv run prepare.py

# Run baseline training
uv run train.py
```

## Running Autoresearch

The `.claude/settings.json` pre-approves all safe operations (file edits, git, python/uv runs) so Claude Code won't ask permission during the experiment loop. Dangerous operations (rm -rf, sudo, curl) remain blocked.

**Option A — Settings file (recommended, already included):**
```bash
# Just start Claude Code in the repo — settings.json handles permissions
claude
# Then say: "Read program.md and start experimenting."
```

**Option B — Full YOLO mode (faster, use in a container or disposable env):**
```bash
claude --dangerously-skip-permissions
# Then say: "Read program.md and start experimenting."
```

**Option C — Headless overnight run:**
```bash
claude --dangerously-skip-permissions \
  -p "Read program.md. Run the full autoresearch loop until you've completed 50 experiments or exhausted all phases." \
  --max-turns 200
```

Commit before starting: `git add -A && git commit -m "checkpoint before autoresearch"` so you can always revert.

## Project Structure

```
program.md      — agent research instructions (human writes this)
prepare.py      — data prep + evaluation (fixed, do not modify)
train.py        — model + CA + training loop (agent modifies this)
ca_rules.py     — CA rule library (agent can import and modify)
results.tsv     — experiment log
analysis.ipynb  — results visualization
NEUROGEN.md     — full research reference
```

## Hardware

Designed for MacBook Pro M1 Pro (MPS backend). Also works on CUDA GPUs and CPU. Default config: depth 4, ~2 min per experiment, ~30 experiments/hour on M1 Pro.

## References & Prior Art

NeuroGen builds on and extends several lines of research. See `NEUROGEN.md` for detailed comparison of how this project differs from each.

- [HyperNCA](https://arxiv.org/abs/2204.11674) — Najarro & Risi, 2022. **Most directly related.** NCA growing RL policy weights via CMA-ES. NeuroGen extends this to transformers, adds live CA, and uses functional neuroscience principles.
- [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) — Mordvintsev et al., 2020. Foundational NCA architecture.
- [Weight Agnostic Neural Networks](https://arxiv.org/abs/1906.04358) — Gaier & Ha, 2019. Architecture encodes inductive bias without weight training.
- [HyperNetworks](https://arxiv.org/abs/1609.09106) — Ha et al., 2017. Small networks generating weights for larger networks.
- [nanochat](https://github.com/karpathy/nanochat) / [autoresearch](https://github.com/karpathy/autoresearch) — Karpathy, 2025-2026. Training harness and autonomous experiment loop.

## License

MIT
