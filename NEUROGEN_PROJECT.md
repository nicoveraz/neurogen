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

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                  AUTO-RESEARCH LOOP                   │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │ Experiment│──▶│  Runner  │──▶│  Analysis &      │ │
│  │ Registry  │   │          │   │  Report Generator │ │
│  └──────────┘   └──────────┘   └──────────────────┘ │
│       ▲                              │               │
│       └──────────────────────────────┘               │
│              (next experiment)                        │
└──────────────────────────────────────────────────────┘
        │                    │
        ▼                    ▼
┌───────────────┐   ┌───────────────────┐
│   MicroGPT    │   │  CA Weight Engine │
│  (Karpathy)   │   │  (NeuroGen Core)  │
└───────────────┘   └───────────────────┘
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

### 4. Auto-Research Engine (`research/engine.py`)

The systematic experiment runner that automates the research loop.

#### 4.1 Experiment Definition

```yaml
# experiments/exp_001_baseline_sweep.yaml
name: "Baseline Initialization Sweep"
hypothesis: "Establish baseline convergence curves for all standard inits"
model:
  config: default_tiny
dataset: shakespeare_char
inits:
  - xavier_uniform
  - xavier_normal
  - kaiming_normal
  - orthogonal
training:
  max_steps: 5000
  eval_interval: 250
  lr: 3e-4
  batch_size: 64
metrics:
  - train_loss
  - val_loss
  - gradient_norm
  - weight_spectral_norm
  - convergence_step_to_target  # steps to reach val_loss < X
seeds: [42, 137, 256]  # statistical replicates
```

#### 4.2 Experiment Phases

The auto-research loop executes these phases in order:

**Phase 1: Baseline Characterization**
- Run all standard inits on Shakespeare char-level
- Record full loss curves, gradient stats, weight statistics over training
- Establish target loss thresholds (e.g., best val_loss at 5K steps)
- Output: baseline report with convergence curves

**Phase 2: CA Development Validation**
- Test each CA variant's ability to produce weight-shaped tensors
- Verify: correct shapes, finite values, reasonable magnitudes
- Measure: development time, genome size, compression ratio
- Analyze: weight statistics (mean, std, spectral properties, rank)
- Output: CA development report

**Phase 3: Random CA vs Random Init**
- Use untrained CA genomes (random parameters) to initialize GPT
- Compare training curves against standard random inits
- Question: does CA *structure* alone (even random) help?
- Output: structure-vs-random report

**Phase 4: Hand-Designed CA Rules**
- Implement hand-designed rules encoding known priors:
  - Block-diagonal (modular) structure
  - Low-rank + sparse composition
  - Attention heads with different frequency biases
  - FFN with gradually increasing receptive field across layers
- Compare against baselines
- Output: prior-knowledge report

**Phase 5: Meta-Learned CA (the big experiment)**
- Outer loop: optimize CA genome parameters
- Inner loop: train GPT from CA-developed weights, measure val_loss
- Meta-objective: minimize val_loss at step N (or area under curve)
- Methods:
  - Evolution strategies (CMA-ES) on genome params — gradient-free
  - MAML-style: differentiate through inner loop (expensive)
  - Reptile-style: approximate meta-gradient
- Output: meta-learning report

**Phase 6: Co-Evolution**
- CA continues to apply weight perturbations during training
- Test: CA applies every K steps with decaying magnitude
- Compare: init-only vs init+periodic vs continuous
- Output: co-evolution report

**Phase 7: Ablations & Analysis**
- Vary: CA steps, genome size, seed pattern, neighborhood size
- Visualize: developed weights (heatmaps, spectra, SVD)
- Attention pattern analysis: do CA-init models learn different patterns?
- Output: ablation report

**Phase 8: Scale Transfer**
- Train CA genome on tiny model
- Apply to larger model (2x, 4x params)
- Does the developmental program generalize?
- Output: transfer report

#### 4.3 Metrics & Logging

Every experiment logs:

```python
metrics = {
    # Training dynamics
    "train_loss": [],          # per step
    "val_loss": [],            # per eval interval
    "gradient_norm": [],       # per step
    "learning_rate": [],       # per step

    # Weight analysis (per eval interval)
    "weight_spectral_norms": {},    # per layer
    "weight_effective_rank": {},    # per layer
    "weight_sparsity": {},          # per layer
    "weight_frobenius_norm": {},    # per layer

    # Convergence
    "steps_to_target_loss": None,   # first step where val_loss < target
    "final_val_loss": None,
    "best_val_loss": None,

    # CA-specific
    "genome_size": None,
    "compression_ratio": None,
    "development_time_ms": None,

    # Compute
    "total_train_time_s": None,
    "peak_memory_mb": None,
}
```

#### 4.4 Report Generator (`research/report.py`)

After each phase, auto-generate a markdown report with:
- Experiment configuration summary
- Key results table
- Loss curve plots (matplotlib → saved as PNG, embedded in markdown)
- Statistical comparisons (mean ± std across seeds)
- Weight visualization panels
- Conclusions and next-step recommendations

---

### 5. Project Structure

```
neurogen/
├── README.md                    # Project overview and quick start
├── NEUROGEN_PROJECT.md          # This file — full specification
├── pyproject.toml               # Project metadata and dependencies
├── requirements.txt             # Pinned dependencies
│
├── neurogen/                    # Core library
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
│   │   ├── handcrafted.py       # Phase 4 hand-designed rules
│   │   └── genome.py            # CAGenome base class
│   ├── baselines/
│   │   ├── __init__.py
│   │   └── initializers.py      # All baseline init strategies
│   ├── data/
│   │   ├── __init__.py
│   │   ├── shakespeare.py       # Shakespeare char-level dataset
│   │   └── tinystories.py       # TinyStories dataset loader
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Training loop
│   │   ├── evaluator.py         # Eval and metrics collection
│   │   └── meta_trainer.py      # Meta-learning outer loop (Phase 5)
│   └── analysis/
│       ├── __init__.py
│       ├── weight_analysis.py   # Spectral, rank, sparsity analysis
│       ├── attention_analysis.py # Attention pattern visualization
│       └── plotting.py          # All matplotlib plotting functions
│
├── exploration/                 # CA search space exploration
│   ├── __init__.py
│   ├── stage1_survey.py         # Broad random+heuristic survey
│   ├── stage2_focused.py        # Bayesian optimization (Optuna)
│   ├── stage3_meta.py           # CMA-ES genome optimization
│   ├── coevolution.py           # Co-evolutionary search
│   ├── budget.py                # Compute budget estimation
│   └── visualization.py         # Exploration-specific plots
│
├── research/                    # Auto-research engine
│   ├── __init__.py
│   ├── engine.py                # Experiment runner
│   ├── registry.py              # Experiment registry and status
│   ├── report.py                # Markdown report generator
│   └── experiments/             # Experiment YAML definitions
│       ├── phase1_baselines.yaml
│       ├── phase2_ca_validation.yaml
│       ├── phase3_random_ca.yaml
│       ├── phase4_handcrafted.yaml
│       ├── phase5_meta_learning.yaml
│       ├── phase6_coevolution.yaml
│       ├── phase7_ablations.yaml
│       └── phase8_transfer.yaml
│
├── scripts/                     # CLI entry points
│   ├── train.py                 # Single training run
│   ├── develop_weights.py       # Run CA development standalone
│   ├── run_experiment.py        # Run a single experiment from YAML
│   ├── run_phase.py             # Run all experiments in a phase
│   ├── run_all.py               # Full auto-research pipeline
│   ├── run_benchmark.py         # Benchmark runner (BM1-BM8, suites)
│   ├── analyze_weights.py       # Weight analysis CLI
│   └── generate_report.py       # Report generation CLI
│
├── .github/                     # CI/CD
│   └── workflows/
│       └── test.yml             # Tests, linting, quick benchmarks
│
├── notebooks/                   # Jupyter notebooks for exploration
│   ├── 01_ca_visualization.ipynb
│   ├── 02_weight_comparison.ipynb
│   └── 03_results_explorer.ipynb
│
├── tests/                       # Test suite
│   ├── test_model.py
│   ├── test_ca_engine.py
│   ├── test_baselines.py
│   ├── test_trainer.py
│   └── test_research_engine.py
│
├── outputs/                     # Generated outputs (gitignored)
│   ├── checkpoints/
│   ├── logs/
│   ├── reports/
│   └── figures/
│
└── data/                        # Data directory (gitignored)
    └── shakespeare/
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

### Sprint 6: Polish & Documentation

22. **Tests** — Full test suite for all modules.
23. **`README.md`** — Quick start, results summary, contributing guide.
24. **Notebooks** — Interactive exploration and visualization.
25. **CI** — GitHub Actions for tests and linting.

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
- **`CLAUDE.md`** — Claude Code implementation instructions, coding conventions, and interface contracts.

## License

MIT

---

*This document is the single source of truth for the NeuroGen project. All implementation decisions should align with this spec. If reality diverges from the spec, update the spec first, then the code.*
