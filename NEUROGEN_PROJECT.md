# NeuroGen: Cellular Automata Developmental Initialization for Language Models

## Project Vision

The mammalian brain does not begin learning from a blank slate. Genetic developmental programs — small rule sets — grow structured neural architecture *before* experience-driven learning begins. Broca's area, Wernicke's area, the arcuate fasciculus: these are not learned, they are *grown* by local iterative rules operating on a cellular substrate. Learning then refines this scaffold.

Current LLMs (transformers) do the opposite: uniform architecture, random initialization, and rely entirely on gradient descent to discover any internal specialization. This works but is almost certainly suboptimal.

**NeuroGen** tests a concrete hypothesis: a small cellular automaton (CA), acting as a "genomic developmental program," can generate structured weight initializations for a transformer that improve training efficiency, final performance, or both — compared to standard random initialization.

---

## Core Hypothesis

> A learned or hand-designed cellular automaton rule set, applied iteratively to a seed tensor, can produce weight matrices with structural priors (modularity, locality, spectral properties) that place the network in a more favorable region of the loss landscape than random initialization, reducing compute needed to reach a target loss.

### Sub-hypotheses

1. **H1 — Structure**: CA-grown weights exhibit measurable structure (block-diagonality, spectral clustering, low-rank regions) absent in random init.
2. **H2 — Convergence**: CA-initialized models reach a reference loss in fewer training steps.
3. **H3 — Compression**: The CA rule set is orders of magnitude smaller than the weight matrices it generates (genomic compression).
4. **H4 — Co-evolution**: Allowing the CA to continue shaping weights *during* training (not just at init) produces further improvements.
5. **H5 — Transferability**: A CA rule learned on a small model/dataset transfers beneficially to a larger model/dataset.
6. **H6 — Live dual-process**: A CA operating at every training step alongside gradient descent produces training dynamics qualitatively different from (and better than) either process alone. The CA-gradient alignment shifts from low (independent agendas) to high (cooperative) over training.

---

## Architecture Overview

The project has two layers: a reusable library (`neurogen/`) and a self-contained autoresearch harness inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

```
┌──────────────────────────────────────────────────────┐
│               AUTORESEARCH HARNESS                    │
│         (AI researcher modifies train.py)             │
│                                                      │
│  prepare.py ──── Fixed data/eval harness             │
│  train.py ────── Model + init + training (modifiable)│
│  program.md ──── Agent instructions                  │
│  results.tsv ─── Experiment log (append-only)        │
│  git ─────────── Keep/discard workflow               │
└──────────────────────────────────────────────────────┘
        │                    │
        ▼                    ▼
┌───────────────┐   ┌───────────────────┐
│   MicroGPT    │   │  CA Weight Engine │
│  (Karpathy)   │   │  (NeuroGen Core)  │
└───────────────┘   └───────────────────┘

The neurogen/ package is the tested, modular library.
The autoresearch harness is self-contained — train.py
does NOT import from neurogen/. Validated ideas get
ported back into the library.
```

---

## Component Specifications

### 1. MicroGPT (`model/gpt.py`)

A minimal GPT implementation following Andrej Karpathy's nanoGPT / microGPT style. Must be clean, readable, and self-contained.

**Requirements:**
- Single-file transformer implementation
- Configurable: `n_layer`, `n_head`, `n_embd`, `block_size`, `vocab_size`, `dropout`
- Causal self-attention with flash attention option
- Standard components: token embeddings, positional embeddings, transformer blocks (LN → MHA → LN → FFN), final LN + LM head
- Weight tying between token embedding and LM head
- `model.get_weight_tensors()` method that returns a dict of all trainable weight matrices (excluding biases and LayerNorm params) — this is the interface the CA engine targets
- `model.set_weight_tensors(dict)` method to inject CA-generated weights
- `model.count_parameters()` utility
- Training loop extracted as a separate function, not inside the model class

**Default config (tiny, for fast iteration):**
```python
config = GPTConfig(
    block_size=256,
    vocab_size=0,       # set by dataset
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,
)
```

**Datasets to support:**
- Shakespeare (char-level) — primary, for fast iteration
- TinyStories — secondary, for word/BPE level validation
- OpenWebText subset — stretch goal

---

### 2. Cellular Automata Weight Engine (`neurogen/ca_engine.py`)

The core innovation. A system that uses CA-like rules to *grow* weight matrices from small seeds.

#### 2.1 Conceptual Model

```
Seed Tensor (small)
       │
       ▼
┌──────────────┐
│  CA Rule Set  │ ──── applied iteratively N steps
│  (local ops)  │
└──────────────┘
       │
       ▼
Developed Weight Matrix (full size)
       │
       ▼
  Injected into GPT layer
```

#### 2.2 CA Rule Variants to Implement

Each variant defines a different "developmental program":

**Variant A — Classic Grid CA (`grid_ca`)**
- Treat each weight matrix as a 2D grid of cells
- Each cell's value is updated based on its neighborhood (Moore or Von Neumann)
- Rule: `w[i,j] = f(neighborhood_mean, neighborhood_std, current_value, step)`
- `f` is a small MLP (the "genome") shared across all cells
- Seed: small center region initialized, rest zeros
- Iterate for T steps → full matrix

**Variant B — Neural Cellular Automata (`neural_ca`)**
- Based on Mordvintsev et al. (Growing Neural Cellular Automata)
- State: each cell has a hidden state vector (not just a scalar)
- Perception: Sobel-like filters to sense neighbors
- Update: small MLP processes perceived state → state delta
- Stochastic update mask (not all cells update each step)
- Final projection from hidden state to scalar weight value
- The MLP parameters ARE the genome

**Variant C — Spectral CA (`spectral_ca`)**
- Operate in frequency domain
- CA rules generate Fourier coefficients
- Inverse FFT produces the weight matrix
- Hypothesis: useful structure is easier to express spectrally

**Variant D — Growth from Topology (`topo_ca`)**
- Define a connectivity graph (which neurons connect to which)
- CA rules grow edge weights on this graph
- The graph structure itself encodes priors (e.g., local connectivity, hierarchical modules)
- Maps to transformer weights via adjacency → weight matrix

**Variant E — Reaction-Diffusion (`reaction_diffusion`)**
- Two coupled "chemical" fields (activator-inhibitor)
- Classic Turing pattern formation
- Produces naturally modular, periodic structures
- Parameters: diffusion rates, reaction rates, feed/kill rates

#### 2.3 Genome Representation

For learned CA variants (A, B), the "genome" is a small parameter set:

```python
class CAGenome(nn.Module):
    """
    The developmental program. Small enough to be meta-learned.
    Typical size: 1K-50K parameters (vs millions in the target GPT).
    """
    def __init__(self, hidden_dim=64, n_channels=16):
        # Perception filters
        # Update MLP
        # Projection to weight value

    def develop(self, seed, target_shape, n_steps=64):
        """Run the CA for n_steps, return developed weight matrix."""
        ...
        return weight_matrix
```

#### 2.4 Interface

```python
class CAWeightEngine:
    def __init__(self, variant: str, genome_config: dict):
        ...

    def develop_weights(self, model: GPT) -> dict:
        """Generate all weight matrices for the model."""
        ...

    def genome_size(self) -> int:
        """Total parameters in the developmental program."""
        ...

    def compression_ratio(self, model: GPT) -> float:
        """Ratio: model_params / genome_params."""
        ...
```

---

### 3. Initialization Baselines (`neurogen/baselines.py`)

For rigorous comparison, implement these standard initialization strategies:

| Name | Method |
|------|--------|
| `xavier_uniform` | Glorot uniform |
| `xavier_normal` | Glorot normal |
| `kaiming_uniform` | He uniform (ReLU-aware) |
| `kaiming_normal` | He normal |
| `orthogonal` | Orthogonal matrices |
| `sparse` | Sparse initialization |
| `fixup` | Fixup initialization (residual-aware) |
| `mimetic` | Mimetic initialization (recent, 2023) |
| `spectral_delta` | Identity-like with spectral scaling |

Each baseline must conform to the same interface: `def initialize(model: GPT) -> dict`

---

### 4. Auto-Research System

NeuroGen uses a Karpathy-style autoresearch approach where an AI agent is the researcher. See `NEUROGEN_AUTORESEARCH.md` for the full specification.

#### 4.1 Core Files

| File | Role | Modifiable? |
|------|------|-------------|
| `prepare.py` | Fixed evaluation harness — data loading, eval, device detection | No |
| `train.py` | Self-contained model + init + training loop | Yes (AI modifies this) |
| `program.md` | Instructions for the AI researcher | Rarely |
| `results.tsv` | Tab-separated experiment log | Append-only |

#### 4.2 The Research Loop

The AI researcher iterates:
1. Read `results.tsv` to understand what's been tried
2. Form a hypothesis about what to try next
3. Modify `train.py` to implement the idea
4. Commit with a descriptive message
5. Run `python train.py` (fixed 2-minute budget)
6. Append results to `results.tsv`
7. Keep (tag) or discard (revert) based on val_loss
8. Repeat

#### 4.3 Research Phases

The AI researcher follows a phased agenda defined in `program.md`:

| Phase | Question | Approach |
|-------|----------|----------|
| 1 | How do standard inits compare? | Sweep xavier, kaiming, orthogonal, etc. |
| 2 | Can simple CAs produce useful weights? | Elementary CA, GoL-style, continuous CA |
| 3 | Do structured CAs help more? | Block-diagonal, low-rank, spectral, multi-scale |
| 4 | Does live CA during training help? | CA alongside gradient descent with decaying α |
| 5 | Can we meta-learn the CA rule? | Evolve rules using training loss as fitness |

#### 4.4 Constraints

- **Time budget**: 2 minutes per experiment (fixed in `prepare.py`)
- **Single file**: All code in `train.py`, no external imports from `neurogen/`
- **Fixed task**: Character-level Shakespeare, fixed eval protocol
- **Self-contained**: Every experiment is reproducible from its git commit

#### 4.5 Legacy Research Infrastructure

The `neurogen/` package also contains a full research infrastructure (`research/` directory) with YAML-driven experiment runner, report generator, and analysis tools. These are available for more controlled experiments but the primary research workflow is the autoresearch harness described above.

---

### 5. Project Structure

```
neurogen/
├── README.md                    # Project overview and quick start
├── NEUROGEN_PROJECT.md          # This file — full specification
├── NEUROGEN_AUTORESEARCH.md     # Autoresearch system specification
├── pyproject.toml               # Project metadata and dependencies
│
├── prepare.py                   # AUTORESEARCH: Fixed eval harness (DO NOT MODIFY)
├── train.py                     # AUTORESEARCH: Modifiable model + training
├── program.md                   # AUTORESEARCH: AI researcher instructions
├── results.tsv                  # AUTORESEARCH: Experiment log (append-only)
│
├── neurogen/                    # Core library (reusable, tested, modular)
│   ├── __init__.py
│   ├── config.py                # All configuration dataclasses
│   ├── model/
│   │   ├── __init__.py
│   │   ├── gpt.py               # MicroGPT implementation
│   │   └── components.py        # Attention, FFN, Block modules
│   ├── ca/
│   │   ├── __init__.py
│   │   ├── engine.py            # CAWeightEngine main class
│   │   ├── grid_ca.py           # Variant A
│   │   ├── neural_ca.py         # Variant B
│   │   ├── spectral_ca.py       # Variant C
│   │   ├── topo_ca.py           # Variant D
│   │   ├── reaction_diffusion.py # Variant E
│   │   ├── handcrafted.py       # Hand-designed rules
│   │   ├── genome.py            # CAGenome base class
│   │   └── live/                # Live CA (operates during training)
│   │       ├── __init__.py
│   │       ├── base.py          # LiveCA base class and interface
│   │       ├── local_norm.py    # Homeostatic weight normalization
│   │       ├── modularity.py    # Block-diagonal structure enforcer
│   │       ├── pruning.py       # Gradient-aware dynamic pruning
│   │       ├── competition.py   # Lateral inhibition / winner-take-all
│   │       ├── learned.py       # Learned CA rule (meta-optimizable genome)
│   │       ├── multi_timescale.py # Multiple CAs at different frequencies
│   │       ├── ca_optimizer.py  # CA as learned optimizer (replaces Adam)
│   │       └── alpha_schedule.py # Developmental influence schedules
│   ├── baselines/
│   │   ├── __init__.py
│   │   └── initializers.py      # All baseline init strategies
│   ├── data/
│   │   ├── __init__.py
│   │   └── shakespeare.py       # Shakespeare char-level dataset
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Training loop
│   │   ├── evaluator.py         # Eval and metrics collection
│   │   ├── live_ca_trainer.py   # Live CA training loop
│   │   └── meta_trainer.py      # Meta-learning outer loop
│   └── analysis/
│       ├── __init__.py
│       ├── weight_analysis.py   # Spectral, rank, sparsity analysis
│       └── plotting.py          # All matplotlib plotting functions
│
├── research/                    # Legacy research infrastructure
│   ├── __init__.py
│   ├── engine.py                # YAML experiment runner
│   ├── report.py                # Markdown report generator
│   └── ...
│
├── scripts/                     # CLI entry points
│   ├── train.py                 # Single training run (uses neurogen/)
│   ├── run_benchmark.py         # Benchmark runner (BM1-BM8)
│   └── ...
│
├── .github/workflows/test.yml   # CI: tests + linting
│
├── tests/                       # Test suite (200+ tests)
│   ├── conftest.py
│   ├── test_model.py
│   ├── test_ca_engine.py
│   ├── test_baselines.py
│   ├── test_live_ca.py
│   └── ...
│
├── outputs/                     # Generated outputs (gitignored)
└── data/                        # Data directory (gitignored)
```

---

## Dependencies

```
# Core
torch>=2.1.0
numpy>=1.24.0

# Data
tiktoken>=0.5.0         # BPE tokenization
datasets>=2.14.0        # HuggingFace datasets (TinyStories)

# Analysis & Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0

# Experiment Management
pyyaml>=6.0
tqdm>=4.65.0
tensorboard>=2.14.0     # optional, for live monitoring

# Meta-learning
cma>=3.3.0              # CMA-ES for evolution strategies

# Dev
pytest>=7.4.0
black>=23.0.0
ruff>=0.1.0
```

---

## Hardware Compatibility & Device Handling

### Primary Target: Apple Silicon (M1 Pro / M1 Max / M2 / M4)

This project is designed to run fully on a MacBook Pro with Apple Silicon. PyTorch supports Apple's Metal Performance Shaders (MPS) backend since v2.1.

#### Device Auto-Detection

All code must use a centralized device selection utility — never hardcode `"cuda"` or `"cpu"`:

```python
# neurogen/config.py
def get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
```

Every script, trainer, and benchmark must call `get_device()` and pass the device through config. The `--device` CLI flag should override auto-detection.

#### MPS Compatibility Notes

Some PyTorch operations are not yet supported on MPS. The codebase must handle these:

| Issue | Workaround |
|-------|-----------|
| `torch.linalg.svd` sometimes fails on MPS | Fall back to CPU for SVD-based analysis |
| `torch.histc` not supported | Use numpy for histogram operations |
| Some custom CUDA kernels (flash attention) | Use standard attention on MPS, flash on CUDA |
| `torch.compile` limited on MPS | Disable `torch.compile` when device is MPS |
| Inconsistent `float64` support | Use `float32` everywhere, cast only for analysis |
| `torch.multinomial` edge cases on MPS | Add CPU fallback for generation sampling |

Implement a helper for analysis functions that use unsupported ops (SVD, eigendecomposition) — always pull to CPU first:

```python
# neurogen/analysis/weight_analysis.py
def spectral_norm(weight: torch.Tensor) -> float:
    """Compute spectral norm, MPS-safe."""
    w = weight.detach().cpu().float()
    return torch.linalg.svdvals(w)[0].item()
```

#### Model Size Configs by Hardware

```python
# neurogen/config.py
HARDWARE_PROFILES = {
    "macbook_m1pro_16gb": {
        "description": "MacBook Pro M1 Pro, 16GB unified memory",
        "max_config": GPTConfig(
            n_layer=6, n_head=6, n_embd=384,
            block_size=256, dropout=0.2,
        ),
        "safe_batch_size": 32,
        "meta_learning_population": 10,
        "meta_learning_inner_steps": 300,
        "notes": "Default config runs fine. Reduce batch for BM7/BM8.",
    },
    "macbook_m1pro_32gb": {
        "description": "MacBook Pro M1 Pro, 32GB unified memory",
        "max_config": GPTConfig(
            n_layer=8, n_head=8, n_embd=512,
            block_size=512, dropout=0.2,
        ),
        "safe_batch_size": 64,
        "meta_learning_population": 20,
        "meta_learning_inner_steps": 500,
        "notes": "Can run medium configs and full meta-learning.",
    },
    "cpu_only": {
        "description": "Any machine, CPU-only fallback",
        "max_config": GPTConfig(
            n_layer=4, n_head=4, n_embd=128,
            block_size=128, dropout=0.2,
        ),
        "safe_batch_size": 16,
        "meta_learning_population": 5,
        "meta_learning_inner_steps": 100,
        "notes": "Slow but works. Use for CI and testing only.",
    },
}
```

#### Estimated Training Times (M1 Pro 16GB, MPS backend)

| Config | Params | 5K steps | BM2 full (3 seeds) | BM7 meta (100 gens) |
|--------|--------|----------|---------------------|----------------------|
| tiny (n_embd=64, 2L) | ~200K | ~1 min | ~3 min | ~30 min |
| small (n_embd=128, 4L) | ~1.5M | ~5 min | ~15 min | ~2 hr |
| default (n_embd=384, 6L) | ~10M | ~20 min | ~1 hr | ~8 hr |
| medium (n_embd=512, 8L) | ~25M | ~45 min | ~2.5 hr | ~20 hr |

These are rough estimates. Actual times depend on batch size and block_size.

#### CLI Hardware Flag

```bash
# Auto-detect (recommended)
python scripts/train.py --steps 5000

# Force specific device
python scripts/train.py --steps 5000 --device mps
python scripts/train.py --steps 5000 --device cpu

# Use hardware profile (auto-adjusts batch size and config limits)
python scripts/train.py --steps 5000 --hardware macbook_m1pro_16gb

# Quick check: verify MPS is working
python -c "from neurogen.config import get_device; print(get_device())"
```

---

## Implementation Priority & Claude Code Instructions

### Role: Research Engineer + ML Scientist

When building this project with Claude Code, follow this implementation order. Each step should produce working, tested code before moving to the next.

### Sprint 1: Foundation (get training working)

1. **`neurogen/config.py`** — All config dataclasses (GPTConfig, TrainConfig, CAConfig, ExperimentConfig). Use `dataclasses` with sensible defaults.

2. **`neurogen/model/gpt.py`** — MicroGPT implementation. Follow Karpathy's nanoGPT closely but ensure the `get_weight_tensors()` / `set_weight_tensors()` interface exists. Test: instantiate model, forward pass with random data, backward pass.

3. **`neurogen/data/shakespeare.py`** — Character-level Shakespeare dataset. Auto-downloads input.txt. Train/val split. Test: load data, create batches.

4. **`neurogen/training/trainer.py`** — Training loop. AdamW, cosine LR schedule, gradient clipping, eval loop. Logs all metrics. Test: train for 100 steps, loss decreases.

5. **`scripts/train.py`** — CLI that ties it all together. `python scripts/train.py --init xavier_normal --steps 5000`

**Checkpoint:** Can train a char-level Shakespeare model end-to-end and generate text.

### Sprint 2: Baselines & Analysis

6. **`neurogen/baselines/initializers.py`** — All baseline init strategies. Each returns a dict of weight tensors. Test: each produces correct shapes and finite values.

7. **`neurogen/analysis/weight_analysis.py`** — Weight statistics: spectral norms, effective rank, sparsity, Frobenius norms, singular value distributions. Test: analyze random weights.

8. **`neurogen/analysis/plotting.py`** — Plotting functions: loss curves, weight heatmaps, spectral plots, comparison charts.

**Checkpoint:** Can run baseline sweep and produce comparison plots.

### Sprint 3: CA Engine (core innovation)

9. **`neurogen/ca/genome.py`** — Base CAGenome class with the develop interface.

10. **`neurogen/ca/grid_ca.py`** — Variant A implementation. Start simple: 3×3 neighborhood, small MLP update rule. Test: develop a weight matrix, verify shape and values.

11. **`neurogen/ca/neural_ca.py`** — Variant B. More sophisticated, hidden state per cell. Test: development produces structured patterns.

12. **`neurogen/ca/reaction_diffusion.py`** — Variant E. Turing patterns. Test: produces visually structured weights.

13. **`neurogen/ca/handcrafted.py`** — Hand-designed rules (Phase 4): block-diagonal, low-rank+sparse, frequency-biased attention init.

14. **`neurogen/ca/engine.py`** — CAWeightEngine that dispatches to variants and handles the full model initialization pipeline.

**Checkpoint:** Can initialize GPT with CA-developed weights and train.

### Sprint 4: Auto-Research Engine

15. **`research/engine.py`** — YAML-driven experiment runner. Handles: config loading, model instantiation, init selection, training, metric collection, checkpointing.

16. **`research/registry.py`** — Tracks experiment status (pending, running, complete, failed), results paths, and enables resumption.

17. **`research/report.py`** — Auto-generates markdown reports with embedded plots and statistical tables.

18. **Experiment YAML files** for Phases 1-4.

**Checkpoint:** Can run `python scripts/run_phase.py --phase 1` and get a complete report.

### Sprint 5: Exploration & Meta-Learning

19. **`exploration/stage1_survey.py`** — Broad survey: random + heuristic configs across all 5 CA variants, evaluated cheaply on tiny model. See `NEUROGEN_EXPLORATION.md` for full protocol.

20. **`exploration/stage2_focused.py`** — Bayesian optimization (Optuna) over architectural hyperparameters for top variants from Stage 1.

21. **`neurogen/training/meta_trainer.py`** — CMA-ES genome optimization (Stage 3). Progressive evaluation, early stopping, warm-starting. See `NEUROGEN_EXPLORATION.md` for budget planning.

22. **`exploration/coevolution.py`** — Co-evolutionary search: CA active during training with tunable interval, magnitude, decay, and scope.

23. **Experiment YAML files** for Phases 5-8.

24. **`neurogen/ca/spectral_ca.py`** and **`neurogen/ca/topo_ca.py`** — Remaining CA variants.

**Checkpoint:** Full exploration pipeline runs Stage 1 → Stage 2 → Stage 3 and produces ranked results.

### Sprint 6: Live CA (dual-process training)

25. **`neurogen/ca/live/base.py`** — LiveCA base class with `step(W, grad_W) -> delta_W` interface. See `NEUROGEN_LIVE_CA.md` for full specification.

26. **`neurogen/ca/live/local_norm.py`**, **`modularity.py`**, **`pruning.py`**, **`competition.py`** — Four hand-designed live CA rules. Each must produce bounded, finite deltas.

27. **`neurogen/ca/live/alpha_schedule.py`** — Developmental influence schedules (exponential decay, cosine, phased, adaptive, cyclic).

28. **`neurogen/training/live_ca_trainer.py`** — Modified training loop integrating CA step after each gradient step. Logs CA-gradient alignment and contribution ratio.

29. **`neurogen/ca/live/learned.py`** — Learned CA rule with meta-optimizable genome. Same CMA-ES outer loop as Sprint 5 but optimizing the live rule.

30. **`neurogen/ca/live/multi_timescale.py`** — Multiple CA rules at different frequencies (fast/medium/slow).

**Checkpoint:** Can train with live CA and measure CA-gradient alignment. Fixed rules produce measurably different training dynamics than baseline.

### Sprint 7: Polish & Documentation

31. **Tests** — Full test suite for all modules including live CA.
32. **`README.md`** — Quick start, results summary, contributing guide.
33. **Notebooks** — Interactive exploration and visualization.
34. **CI** — GitHub Actions for tests and linting.

---

## Key Design Principles

1. **Reproducibility first.** Every experiment must be fully specified by its YAML config + a random seed. No implicit state.

2. **Minimal dependencies.** PyTorch + numpy for core. Everything else is optional.

3. **Readable over clever.** This is a research codebase. Prioritize clarity. Comment the *why*, not the *what*.

4. **Fail fast, log everything.** Assertions on shapes and value ranges. Comprehensive logging.

5. **Modular interfaces.** The CA engine and the GPT model communicate only through `get_weight_tensors()` / `set_weight_tensors()`. Any init strategy (baseline or CA) must conform to `def initialize(model) -> dict[str, Tensor]`.

6. **Small-scale first.** All default configs should run in <5 minutes on a CPU. Scale up via config changes, not code changes.

---

## Success Criteria

The project succeeds if ANY of the following are demonstrated:

1. **A CA-initialized model reaches baseline val_loss in ≥20% fewer steps** (averaged across seeds).
2. **A CA-initialized model achieves ≥5% lower final val_loss** at the same step count.
3. **The CA genome is ≥100x smaller** than the weights it generates while matching baseline performance.
4. **CA-developed weights show measurably different structure** (spectral, rank, modularity) that correlates with training dynamics.
5. **A CA rule learned on a small model transfers** to a larger model with positive effect.

Even negative results are valuable if rigorously documented — they constrain the hypothesis space for future work.

---

## References & Prior Art

- Karpathy, A. — nanoGPT: https://github.com/karpathy/nanoGPT
- Mordvintsev, A. et al. (2020) — Growing Neural Cellular Automata (Distill)
- Gaier, A. & Ha, D. (2019) — Weight Agnostic Neural Networks
- Najarro, E. & Risi, S. (2020) — Meta-Learning through Hebbian Plasticity in Random Networks
- Ha, D. et al. (2017) — HyperNetworks
- Stanley, K. & Miikkulainen, R. (2002) — NEAT: Evolving Neural Network Topologies
- Turing, A. (1952) — The Chemical Basis of Morphogenesis (reaction-diffusion)
- Frankle, J. & Carlin, M. (2019) — The Lottery Ticket Hypothesis
- Dauphin, Y. & Schoenholz, S. (2019) — MetaInit: Initializing learning by learning to initialize
- Zhu, C. et al. (2023) — Mimetic Initialization for Transformers

---

## Companion Documents

- **`NEUROGEN_TESTING.md`** — Complete test suite definitions (70+ test cases), 8 benchmark protocols (BM1-BM8), CI/CD workflows, sprint validation checklists, and benchmark runner configurations.
- **`NEUROGEN_EXPLORATION.md`** — CA configuration space exploration strategy: three-stage funnel (broad survey → Bayesian optimization → CMA-ES meta-learning), search space taxonomy, budget planning for M1 Pro, and co-evolutionary search protocol.
- **`NEUROGEN_LIVE_CA.md`** — Live CA specification: CA operating within training step-by-step alongside gradient descent. Five integration modes (additive, homeostatic, pruning, multi-timescale, CA-as-optimizer), five concrete CA rules, alpha schedules, per-layer scope configuration, CA-gradient alignment diagnostics, and biological motivation.
- **`NEUROGEN_AUTORESEARCH.md`** — Karpathy-style autoresearch system: AI-as-researcher with fixed eval harness (`prepare.py`), single modifiable file (`train.py`), agent instructions (`program.md`), and append-only results log (`results.tsv`). The LLM is the decision engine.
- **`program.md`** — Instructions for the AI researcher: 5-phase research agenda, experiment loop protocol, results format, and scientific guidelines.
- **`CLAUDE.md`** — Claude Code implementation instructions, coding conventions, and interface contracts.

## License

MIT

---

*This document is the single source of truth for the NeuroGen project. All implementation decisions should align with this spec. If reality diverges from the spec, update the spec first, then the code.*
