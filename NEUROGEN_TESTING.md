# NeuroGen: Testing & Benchmarking Specification

This document is a companion to `NEUROGEN_PROJECT.md`. It defines all test cases, benchmark suites, and evaluation protocols for the project.

---

## Part 1: Test Suite

### Philosophy

- Every module gets unit tests that run in **<10 seconds** on CPU
- Use tiny model configs: `n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256`
- Tests verify **correctness**, not performance (benchmarks handle that)
- Tests should catch shape mismatches, NaN/Inf propagation, and interface violations
- Run full suite with `pytest tests/ -v`
- Run fast subset with `pytest tests/ -v -m "not slow"`

---

### Test Definitions by Module

#### `tests/test_model.py` — MicroGPT

```
test_gpt_instantiation
    Create GPT with default tiny config
    Assert: model creates without error
    Assert: parameter count matches expected formula

test_gpt_forward_shape
    Forward pass with random input (batch=4, seq_len=32)
    Assert: logits shape == (batch, seq_len, vocab_size)
    Assert: loss is scalar when targets provided
    Assert: loss is None when targets omitted

test_gpt_backward
    Forward + backward pass
    Assert: all parameters have non-None gradients
    Assert: no gradient is NaN or Inf

test_gpt_generate
    Generate 50 tokens from a trained-for-10-steps model
    Assert: output shape == (1, prompt_len + 50)
    Assert: all token IDs in [0, vocab_size)

test_gpt_weight_interface_get
    Call model.get_weight_tensors()
    Assert: returns dict[str, Tensor]
    Assert: keys include attention and FFN weights for each layer
    Assert: no bias or LayerNorm params included
    Assert: all tensors require_grad

test_gpt_weight_interface_set
    Get weights, modify one tensor (fill with zeros), set back
    Assert: model's internal parameter is now zeros
    Assert: forward pass still runs (no shape mismatch)

test_gpt_weight_interface_roundtrip
    Get weights, set them back unchanged
    Assert: model output is identical before and after

test_gpt_weight_tying
    Assert: token embedding weight tensor is same object as LM head weight
    Assert: get_weight_tensors returns it only once (not duplicated)

test_gpt_configs
    Instantiate with several different configs (vary n_layer, n_head, n_embd)
    Assert: all create successfully
    Assert: n_embd must be divisible by n_head (should raise if not)

test_gpt_determinism
    Two forward passes with same input and same seed
    Assert: identical outputs
```

#### `tests/test_ca_engine.py` — CA Weight Engine

```
test_ca_engine_variants_registered
    Assert: CAWeightEngine.available_variants() includes all 5 variants
    Assert: ['grid_ca', 'neural_ca', 'spectral_ca', 'topo_ca', 'reaction_diffusion']

test_ca_develop_weights_shapes
    For each CA variant:
        engine = CAWeightEngine(variant, default_config)
        weights = engine.develop_weights(model)
        For each (key, tensor) in weights:
            Assert: shape matches model.get_weight_tensors()[key].shape
            Assert: dtype is float32

test_ca_develop_weights_finite
    For each CA variant:
        weights = engine.develop_weights(model)
        For each tensor in weights.values():
            Assert: no NaN values
            Assert: no Inf values

test_ca_develop_weights_magnitude
    For each CA variant:
        weights = engine.develop_weights(model)
        For each tensor in weights.values():
            Assert: tensor.std() > 1e-6  (not collapsed to constant)
            Assert: tensor.std() < 10.0  (not exploded)
            Assert: tensor.abs().max() < 100.0

test_ca_develop_deterministic
    Same seed → same weights
    For each CA variant:
        torch.manual_seed(42)
        w1 = engine.develop_weights(model)
        torch.manual_seed(42)
        w2 = engine.develop_weights(model)
        Assert: all tensors identical

test_ca_develop_different_seeds
    Different seeds → different weights
    torch.manual_seed(42); w1 = engine.develop_weights(model)
    torch.manual_seed(99); w2 = engine.develop_weights(model)
    Assert: at least one tensor differs

test_ca_genome_size
    For each CA variant:
        Assert: engine.genome_size() > 0
        Assert: engine.genome_size() < model.count_parameters() / 10
        Assert: engine.compression_ratio(model) > 10

test_ca_genome_gradient_flow
    For learnable variants (grid_ca, neural_ca):
        weights = engine.develop_weights(model)
        loss = sum(w.sum() for w in weights.values())
        loss.backward()
        Assert: genome parameters have non-None gradients

test_ca_step_count_affects_output
    Same seed, different n_steps → different weights
    w1 = engine.develop(seed, shape, n_steps=10)
    w2 = engine.develop(seed, shape, n_steps=50)
    Assert: w1 != w2

test_ca_target_shape_flexibility
    Assert: can develop weights for different target shapes
    Develop (64, 64), (128, 256), (384, 1536) with same genome
    Assert: all produce correct shapes
```

#### `tests/test_ca_variants.py` — Individual CA Variant Tests

```
# Grid CA
test_grid_ca_neighborhood
    Verify neighborhood extraction is correct (Moore 3x3)
    Known grid → known neighborhood values

test_grid_ca_seed_propagation
    Start with small center seed, run 5 steps
    Assert: non-zero region grows over steps

test_grid_ca_update_rule
    Manually compute expected output for a 5x5 grid, 1 step
    Assert: matches actual output

# Neural CA
test_neural_ca_hidden_state
    Assert: internal state has channel dimension
    Assert: state shape == (H, W, n_channels) during development

test_neural_ca_stochastic_update
    Run same state twice with stochastic mask
    Assert: results differ (stochastic) but are both valid

test_neural_ca_perception_filters
    Assert: Sobel-like filters are correctly shaped
    Assert: perception output has expected channel count

# Spectral CA
test_spectral_ca_frequency_domain
    Assert: intermediate state is in frequency domain
    Assert: final output is in spatial domain (real-valued)

test_spectral_ca_symmetry
    Assert: if rules are symmetric, output has corresponding symmetry

# Reaction-Diffusion
test_rd_turing_patterns
    Run for 200+ steps on 64x64 grid
    Assert: output has spatial structure (not uniform)
    Assert: autocorrelation shows periodicity

test_rd_parameter_sensitivity
    Vary feed/kill rates
    Assert: different parameters produce different pattern types
```

#### `tests/test_baselines.py` — Initialization Baselines

```
test_all_baselines_registered
    Assert: all 9 baseline methods are available

test_baseline_shapes
    For each baseline:
        weights = baseline.initialize(model)
        Assert: keys match model.get_weight_tensors().keys()
        Assert: all shapes match

test_baseline_finite
    For each baseline:
        Assert: no NaN, no Inf in any tensor

test_baseline_statistics
    For each baseline:
        For each tensor:
            Assert: mean is near 0 (within 0.1)
            Assert: std is in reasonable range [0.001, 1.0]

test_xavier_uniform_bounds
    Assert: values in [-bound, +bound] where bound = sqrt(6/(fan_in+fan_out))

test_kaiming_normal_variance
    Assert: variance ≈ 2/fan_in (within 10% for large tensors)

test_orthogonal_property
    For square weight matrices:
        Assert: W @ W.T ≈ I (within tolerance 1e-5)

test_baseline_interface_matches_ca
    Both return dict[str, Tensor] with same key format
    Assert: can use either interchangeably with model.set_weight_tensors()
```

#### `tests/test_trainer.py` — Training Loop

```
test_trainer_loss_decreases
    Train for 100 steps on Shakespeare
    Assert: step_100_loss < step_0_loss

test_trainer_gradient_clipping
    Set max_grad_norm = 0.5
    Assert: no gradient exceeds 0.5 after clipping

test_trainer_lr_schedule
    Cosine schedule with warmup
    Assert: LR starts at 0, warms up to max, decays to min
    Check specific step values

test_trainer_eval_metrics
    Run eval after 50 steps
    Assert: returns dict with 'val_loss'
    Assert: val_loss is finite scalar

test_trainer_checkpoint_save_load
    Train 50 steps, save checkpoint
    Load checkpoint into fresh model
    Assert: same val_loss
    Assert: optimizer state restored

test_trainer_deterministic
    Same seed → same loss at step 50
    Two independent runs
    Assert: losses match exactly

test_trainer_custom_init
    Train with CA init vs xavier init
    Assert: both produce decreasing loss (no crash)
    Assert: loss values differ (init matters)

test_trainer_metrics_logging
    Train 100 steps with eval_interval=25
    Assert: metrics dict has 4 eval points
    Assert: all expected keys present (train_loss, val_loss, etc.)

test_trainer_nan_detection
    Inject NaN weights manually
    Assert: trainer raises or logs NaN detection within 5 steps
```

#### `tests/test_data.py` — Datasets

```
test_shakespeare_download_and_load
    Load Shakespeare dataset
    Assert: train and val splits exist
    Assert: vocab_size > 0
    Assert: total chars > 1_000_000

test_shakespeare_batch_shape
    Get a batch (batch_size=4, block_size=32)
    Assert: x.shape == (4, 32)
    Assert: y.shape == (4, 32)
    Assert: y == x shifted by 1

test_shakespeare_encoding_roundtrip
    Encode then decode a known string
    Assert: roundtrip matches original

test_shakespeare_all_chars_covered
    Assert: every char in the dataset has a valid encoding
```

#### `tests/test_analysis.py` — Weight Analysis

```
test_spectral_norm
    Known matrix → known largest singular value
    Assert: matches torch.linalg.svdvals()[0]

test_effective_rank
    Full-rank random matrix → rank ≈ min(m,n)
    Rank-1 matrix → effective rank ≈ 1
    Assert: both within tolerance

test_sparsity_measure
    All-zeros matrix → sparsity = 1.0
    No-zeros matrix → sparsity = 0.0
    50% zeros → sparsity ≈ 0.5

test_frobenius_norm
    Known matrix → known Frobenius norm
    Assert: matches manual computation

test_weight_comparison_report
    Compare two sets of weights
    Assert: returns dict with all expected metrics
    Assert: includes per-layer and aggregate statistics
```

#### `tests/test_research_engine.py` — Auto-Research Engine

```
test_experiment_yaml_loading
    Load a phase 1 YAML file
    Assert: all required fields present
    Assert: config parses into ExperimentConfig

test_experiment_yaml_validation
    YAML missing required field → raises clear error
    YAML with invalid init name → raises clear error
    YAML with invalid model config → raises clear error

test_experiment_runner_single
    Run a minimal experiment (10 steps, 1 seed)
    Assert: completes without error
    Assert: results directory created
    Assert: metrics JSON written

test_experiment_registry
    Register an experiment, mark running, mark complete
    Assert: status transitions are correct
    Assert: can resume from failed state

test_report_generation
    Generate report from pre-computed results
    Assert: markdown file created
    Assert: contains expected sections (config, results, plots)
    Assert: PNG figures exist and are referenced

test_phase_runner
    Run phase 1 with minimal config (2 inits, 50 steps, 1 seed)
    Assert: all experiments complete
    Assert: phase report generated with comparison table
```

#### `tests/test_integration.py` — End-to-End Integration

```
test_e2e_baseline_training
    Full pipeline: data → model → xavier init → train 200 steps → eval → metrics
    Assert: val_loss < 4.0 (char-level, should beat random)

test_e2e_ca_training
    Full pipeline: data → model → grid_ca init → train 200 steps → eval → metrics
    Assert: val_loss is finite
    Assert: produces valid generated text

test_e2e_ca_vs_baseline_comparison
    Train both, collect metrics
    Assert: both produce valid metrics dicts
    Assert: comparison can be computed (loss difference, convergence step)

test_e2e_experiment_yaml
    Write a minimal YAML, run it through the research engine
    Assert: produces complete results and report
```

#### `tests/test_device.py` — Hardware & MPS Compatibility

```
test_device_autodetect
    device = get_device()
    Assert: returns one of ["cuda", "mps", "cpu"]
    Assert: torch can create a tensor on the detected device

test_mps_model_forward
    Skip if MPS not available
    model = GPT(tiny_config).to("mps")
    x = torch.randint(...).to("mps")
    logits, loss = model(x, y)
    Assert: logits.device.type == "mps"
    Assert: loss is finite

test_mps_model_backward
    Skip if MPS not available
    Forward + backward on MPS
    Assert: all gradients are finite (no MPS NaN bugs)

test_mps_training_loop
    Skip if MPS not available
    Train 50 steps on MPS
    Assert: loss decreases
    Assert: no device mismatch errors

test_mps_ca_development
    Skip if MPS not available
    For each CA variant:
        Develop weights on MPS
        Assert: all tensors on correct device
        Assert: all finite

test_mps_analysis_cpu_fallback
    Skip if MPS not available
    model on MPS → call spectral_norm()
    Assert: works without error (internally moves to CPU)
    Assert: returns finite float

test_mps_generation_sampling
    Skip if MPS not available
    Generate 50 tokens on MPS
    Assert: no multinomial errors
    Assert: valid token IDs

test_mps_checkpoint_save_load
    Skip if MPS not available
    Train on MPS, save checkpoint
    Load checkpoint (should work on any device)
    Assert: loads successfully on CPU

test_device_consistency
    Train 20 steps on CPU with seed 42
    If MPS available: train 20 steps on MPS with seed 42
    Assert: losses are close (within 1e-4) — may differ slightly due to float32 precision

test_hardware_profile_loading
    Load "macbook_m1pro_16gb" profile
    Assert: batch_size, model config are within expected ranges
    Assert: profile exists for "cpu_only"
```

#### `tests/test_live_ca.py` — Live CA Rules

```
test_live_ca_base_interface
    All live CA rules implement step(W, grad_W) -> delta_W
    Assert: output shape == input shape
    Assert: output is finite

test_live_ca_delta_bounded
    For each live CA rule:
        delta = rule.step(random_weights, random_grads)
        Assert: delta.abs().max() < 1.0  (bounded)
        Assert: delta.abs().mean() < 0.1  (small corrections)

test_local_norm_reduces_outliers
    Create weight matrix with one extreme outlier
    delta = LocalNormCA().step(W)
    Assert: delta at outlier position pushes toward local mean
    Assert: delta at normal positions is small

test_modularity_ca_structure
    W = random weights
    delta = ModularityCA(n_blocks=4).step(W)
    Assert: on-diagonal block deltas are positive (reinforce)
    Assert: off-diagonal block deltas are negative (suppress)

test_pruning_ca_needs_gradients
    delta = PruningCA().step(W, grad_W=None)
    Assert: raises or returns zero (needs gradient info)
    delta = PruningCA().step(W, grad_W=grad)
    Assert: returns non-zero delta

test_pruning_ca_suppresses_unimportant
    W with some near-zero weights + near-zero gradients
    delta = PruningCA().step(W, grad_W)
    Assert: delta pushes unimportant weights further toward zero

test_competition_ca_winner_take_all
    W with one clear local maximum
    delta = CompetitionCA().step(W)
    Assert: winner gets positive delta (strengthened)
    Assert: neighbors get negative delta (suppressed)

test_learned_ca_gradient_flow
    rule = LearnedCA()
    W = torch.randn(64, 64, requires_grad=True)
    delta = rule.step(W)
    delta.sum().backward()
    Assert: W.grad is not None
    Assert: rule.rule_net parameters have gradients

test_multi_timescale_frequency
    ca = MultiTimescaleCA(fast_interval=1, medium_interval=100, slow_interval=1000)
    Assert: at step 1, only fast CA fires
    Assert: at step 100, fast + medium fire
    Assert: at step 1000, all three fire

test_alpha_schedule_decay
    For each schedule type (exponential, cosine, phased, cyclic):
        alphas = [schedule.get_alpha(t) for t in range(5000)]
        Assert: all finite and non-negative
        Assert: exponential/cosine → monotonically non-increasing overall

test_alpha_schedule_adaptive
    schedule = AlphaSchedule(mode="adaptive")
    Assert: stagnating loss → higher alpha
    Assert: improving loss → lower alpha

test_live_ca_trainer_integration
    Train 100 steps with LiveCATrainer + LocalNormCA
    Assert: loss decreases
    Assert: ca_delta metrics are logged
    Assert: ca_gradient_alignment is logged

test_live_ca_trainer_no_nan
    Train 200 steps with each live CA rule
    Assert: no NaN in weights at any point
    Assert: no NaN in loss

test_live_ca_scope_config
    Configure different CA rules for different layers
    Train 50 steps
    Assert: attention weights get CompetitionCA deltas
    Assert: FFN weights get PruningCA deltas
    Assert: embedding weights get LocalNormCA deltas

test_ca_gradient_alignment_metric
    Known parallel vectors → alignment = 1.0
    Known anti-parallel → alignment = -1.0
    Known orthogonal → alignment ≈ 0.0
```

---

### Test Configuration

```python
# tests/conftest.py

import pytest
import torch
from neurogen.config import GPTConfig, TrainConfig, get_device

@pytest.fixture
def device():
    """Auto-detected device for tests."""
    return get_device()

@pytest.fixture
def tiny_config():
    """Ultra-small config for fast tests."""
    return GPTConfig(
        block_size=32,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
    )

@pytest.fixture
def tiny_train_config():
    """Minimal training config for tests."""
    return TrainConfig(
        max_steps=50,
        eval_interval=25,
        batch_size=4,
        lr=1e-3,
        grad_clip=1.0,
    )

@pytest.fixture
def tiny_model(tiny_config, device):
    """Pre-instantiated tiny model on best available device."""
    return GPT(tiny_config).to(device)

@pytest.fixture
def random_batch(tiny_config, device):
    """Random batch matching tiny config on best available device."""
    x = torch.randint(0, tiny_config.vocab_size, (4, tiny_config.block_size)).to(device)
    y = torch.randint(0, tiny_config.vocab_size, (4, tiny_config.block_size)).to(device)
    return x, y

def mps_available():
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

# Skip decorator for MPS-only tests
requires_mps = pytest.mark.skipif(not mps_available(), reason="MPS not available")
```

### Pytest Markers

```ini
# pyproject.toml additions
[tool.pytest.ini_options]
markers = [
    "slow: marks tests that take >30s (deselect with '-m not slow')",
    "gpu: marks tests requiring CUDA GPU",
    "mps: marks tests requiring Apple MPS backend",
    "integration: marks end-to-end integration tests",
    "ca_variant: marks tests for specific CA variant",
]
```

---

## Part 2: Benchmark Suite

### Purpose

Benchmarks answer the research questions. They are separate from tests — they measure *relative performance*, not correctness.

---

### Benchmark Definitions

#### BM1: Initialization Quality Benchmark

**Question:** How do initial weight properties differ across init strategies?

**Protocol:**
1. For each init strategy (9 baselines + 5 CA variants):
   - Initialize the default tiny model 10 times (different seeds)
   - Measure for each weight tensor:
     - Mean, std, min, max
     - Spectral norm (largest singular value)
     - Effective rank (Shannon entropy of normalized singular values)
     - Condition number
     - Sparsity (fraction of near-zero values, threshold=1e-4)
     - Frobenius norm
   - Measure for the overall model:
     - Initial train loss (before any training)
     - Initial gradient norm (single backward pass)
     - Initial val loss
2. Report: mean ± std across seeds for each metric

**Output:** `outputs/benchmarks/bm1_init_quality.md` with tables and violin plots

---

#### BM2: Convergence Speed Benchmark

**Question:** How fast does each init strategy reach target loss?

**Protocol:**
1. Define target losses: {4.0, 3.5, 3.0, 2.5, 2.0} (char-level Shakespeare)
2. For each init strategy:
   - Train for 5000 steps, 3 seeds
   - Record step at which each target is first reached (or "not reached")
   - Record full loss curve (every 10 steps)
3. Compute:
   - Steps-to-target for each threshold
   - Area under the loss curve (AUC) — lower is better
   - Final loss at step 5000
   - Loss variance across seeds at step 5000

**Output:** `outputs/benchmarks/bm2_convergence.md` with loss curve overlays and steps-to-target bar charts

---

#### BM3: Compute Efficiency Benchmark

**Question:** What is the total compute cost including initialization?

**Protocol:**
1. For each init strategy:
   - Measure wall-clock time for initialization (ms)
   - Measure wall-clock time for 1000 training steps (s)
   - Measure peak GPU/CPU memory during init
   - Measure peak GPU/CPU memory during training
   - Compute: total time to reach val_loss=3.0 (init time + train time)
2. For CA strategies additionally:
   - Genome parameter count
   - Compression ratio (model params / genome params)
   - CA development step count
   - Time per CA step

**Output:** `outputs/benchmarks/bm3_efficiency.md` with time and memory tables, Pareto plots (loss vs. compute)

---

#### BM4: Weight Structure Benchmark

**Question:** Do CA-initialized weights have measurably different structure?

**Protocol:**
1. For each init strategy, after initialization (before training):
   - Compute singular value spectrum for each weight matrix
   - Compute block-diagonal score (energy in block-diagonal vs off-diagonal)
   - Compute locality score (weight magnitude vs distance from diagonal)
   - Run PCA on flattened weight vectors — do CA inits cluster separately?
   - Compute mutual information between layers
2. Repeat after 500 and 5000 steps of training:
   - Same metrics as above
   - Additionally: CKA similarity between init and trained weights
   - Question: do CA inits retain their structure through training?

**Output:** `outputs/benchmarks/bm4_structure.md` with spectral plots, heatmaps, PCA scatter, structure-retention curves

---

#### BM5: Training Dynamics Benchmark

**Question:** Do CA inits produce different training dynamics?

**Protocol:**
1. For each init strategy, during 5000-step training:
   - Gradient norm per layer per step
   - Gradient signal-to-noise ratio
   - Weight update magnitude vs weight magnitude
   - Loss landscape sharpness (SAM-style: loss at w vs. loss at w + epsilon)
   - Attention entropy per head per layer (every 500 steps)
   - Attention pattern diversity (cosine similarity between heads)
2. Compute:
   - Gradient flow health: ratio of gradient norm at first layer vs last layer
   - Training stability: max gradient norm spike
   - Specialization speed: when do attention heads diverge in entropy?

**Output:** `outputs/benchmarks/bm5_dynamics.md` with per-layer gradient flow plots, attention entropy timelines, stability metrics

---

#### BM6: Generated Text Quality Benchmark

**Question:** Does init strategy affect qualitative output beyond loss?

**Protocol:**
1. For each init strategy, at steps {500, 1000, 2500, 5000}:
   - Generate 10 samples of 200 chars each, temperature=0.8
   - Measure:
     - Character-level perplexity on held-out validation
     - Unique n-gram ratio (n=3,4,5) — diversity measure
     - Longest coherent word sequence — crude quality proxy
     - Repetition rate (fraction of repeated 10-grams)
2. Optional (if compute allows):
   - Human evaluation: rate 5 samples per strategy on a 1-5 coherence scale

**Output:** `outputs/benchmarks/bm6_generation.md` with sample tables and diversity metrics

---

#### BM7: Meta-Learning Benchmark (Phase 5)

**Question:** Can the CA genome be optimized, and does it outperform random CA?

**Protocol:**
1. Meta-learn CA genome using CMA-ES (100 generations, population=20)
   - Inner loop: train GPT from CA init for 500 steps, measure val_loss
   - Outer loop: optimize genome to minimize inner val_loss
2. Compare:
   - Random CA genome init → train 5000 steps
   - Meta-learned CA genome → train 5000 steps
   - Best baseline → train 5000 steps
3. Track:
   - Meta-loss curve (outer loop)
   - Genome parameter evolution over generations
   - Best genome's developed weights vs random genome's weights

**Output:** `outputs/benchmarks/bm7_metalearning.md`

---

#### BM8: Scale Transfer Benchmark (Phase 8)

**Question:** Does a genome learned on a small model help a larger model?

**Protocol:**
1. Learn CA genome on tiny model (n_embd=64, n_layer=2)
2. Apply same genome to:
   - Small model (n_embd=128, n_layer=4)
   - Medium model (n_embd=256, n_layer=6)
   - Default model (n_embd=384, n_layer=6)
3. For each, compare against:
   - Xavier init (same larger model)
   - CA with random genome (same larger model)
4. Report convergence curves and final loss for each

**Output:** `outputs/benchmarks/bm8_transfer.md`

---

#### BM9: Live CA Training Dynamics Benchmark

**Question:** How does a live CA change training dynamics compared to standard training?

**Protocol:**
1. For each live CA rule (LocalNorm, Modularity, Pruning, Competition, Learned):
   - Train for 5000 steps with live CA active, 3 seeds
   - Record per-step:
     - CA delta magnitude (||Δw_ca||)
     - Gradient delta magnitude (||Δw_grad||)
     - CA contribution ratio: ||Δw_ca|| / (||Δw_ca|| + ||Δw_grad||)
     - CA-gradient alignment (cosine similarity)
   - Record per-eval-interval:
     - Weight structure scores (modularity, sparsity, spectral)
     - Attention head specialization (entropy diversity)
2. Compare against:
   - Standard training (no CA)
   - Init-only CA (same rule at init, then standard training)
3. Test each alpha schedule (exponential, cosine, phased, adaptive, cyclic)
   with the best-performing CA rule

**Output:** `outputs/benchmarks/bm9_live_ca_dynamics.md` with alignment curves, contribution ratio plots, structure evolution timelines

---

#### BM10: Live CA vs Init-Only CA Benchmark

**Question:** Is it better to use a CA only at initialization, or to keep it active during training?

**Protocol:**
1. For the top 3 CA variants from exploration Stage 2:
   - **Condition A:** CA at init only → standard training 5000 steps
   - **Condition B:** Random init → live CA during training 5000 steps
   - **Condition C:** CA at init → same CA live during training 5000 steps
   - **Condition D:** CA at init → different CA live during training 5000 steps
   - **Baseline:** Xavier init → standard training 5000 steps
2. 3 seeds per condition, default model
3. Compare: convergence speed, final loss, weight structure, generation quality

**Output:** `outputs/benchmarks/bm10_live_vs_init.md` with head-to-head loss curves and summary statistics

---

### Benchmark Configurations

```python
# neurogen/config.py — benchmark-specific configs

@dataclass
class BenchmarkConfig:
    """Configuration for running benchmarks."""
    # Model sizes to test
    model_sizes: dict = field(default_factory=lambda: {
        "tiny":    GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=32),
        "small":   GPTConfig(n_layer=4, n_head=4, n_embd=128, block_size=128),
        "medium":  GPTConfig(n_layer=6, n_head=6, n_embd=256, block_size=256),
        "default": GPTConfig(n_layer=6, n_head=6, n_embd=384, block_size=256),
    })

    # Training budgets
    quick_steps: int = 500       # for fast benchmarks
    standard_steps: int = 5000   # for full benchmarks
    extended_steps: int = 20000  # for convergence studies

    # Statistical replicates
    n_seeds: int = 3             # minimum for error bars
    n_seeds_full: int = 5        # for final reported results

    # Target val losses (char-level Shakespeare)
    target_losses: list = field(default_factory=lambda: [4.0, 3.5, 3.0, 2.5, 2.0])

    # Eval intervals
    eval_interval_quick: int = 50
    eval_interval_standard: int = 250
```

---

### Benchmark Runner

```
scripts/run_benchmark.py --benchmark bm1          # single benchmark
scripts/run_benchmark.py --benchmark all           # all benchmarks
scripts/run_benchmark.py --benchmark bm2 --quick   # fast version (fewer seeds, steps)
scripts/run_benchmark.py --suite convergence       # predefined suite
```

**Predefined suites:**
- `quick`: BM1 + BM2 with tiny model, 1 seed, 500 steps (~2 min M1 Pro MPS, ~5 min CPU)
- `standard`: BM1-BM6 with default model, 3 seeds, 5000 steps (~3 hr M1 Pro MPS)
- `full`: All BM1-BM8 with 5 seeds (~15-20 hr M1 Pro MPS)
- `convergence`: BM2 only, extended steps, 5 seeds (deep convergence study)

**Estimated benchmark times on MacBook Pro M1 Pro 16GB (MPS):**

| Benchmark | Quick (tiny, 1 seed) | Standard (default, 3 seeds) |
|-----------|---------------------|-----------------------------|
| BM1: Init Quality | <10 sec | ~2 min |
| BM2: Convergence | ~1 min | ~1 hr |
| BM3: Compute Efficiency | ~1 min | ~30 min |
| BM4: Weight Structure | ~30 sec | ~15 min |
| BM5: Training Dynamics | ~2 min | ~1.5 hr |
| BM6: Generation Quality | ~1 min | ~30 min |
| BM7: Meta-Learning | ~10 min | ~8 hr |
| BM8: Scale Transfer | ~5 min | ~3 hr |

---

### Benchmark Output Format

Each benchmark produces:
1. **Raw data:** `outputs/benchmarks/{bm_id}/raw/` — JSON/CSV files with all metrics
2. **Figures:** `outputs/benchmarks/{bm_id}/figures/` — PNG plots
3. **Report:** `outputs/benchmarks/{bm_id}/report.md` — Markdown summary with embedded figures
4. **Summary row:** appended to `outputs/benchmarks/summary.csv` — one-line per experiment for cross-benchmark analysis

---

## Part 3: CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v -m "not slow and not gpu" --tb=short

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install black ruff
      - run: black --check .
      - run: ruff check .

  benchmark-quick:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e .
      - run: python scripts/run_benchmark.py --suite quick
      - uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: outputs/benchmarks/
```

---

## Part 4: Validation Checklist

Before marking any sprint as complete, verify:

### Sprint 1 Checklist
- [ ] `pytest tests/test_model.py` — all pass
- [ ] `pytest tests/test_data.py` — all pass
- [ ] `pytest tests/test_trainer.py` — all pass
- [ ] `pytest tests/test_device.py` — all pass (MPS tests skipped if not on Mac)
- [ ] `python scripts/train.py --steps 100` — loss decreases, generates text
- [ ] `python scripts/train.py --steps 100 --device mps` — works on Apple Silicon
- [ ] Model parameter count matches hand-calculation
- [ ] `get_device()` correctly returns "mps" on Mac, "cuda" on GPU, "cpu" otherwise

### Sprint 2 Checklist
- [ ] `pytest tests/test_baselines.py` — all pass
- [ ] `pytest tests/test_analysis.py` — all pass
- [ ] `python scripts/run_benchmark.py --benchmark bm1 --quick` — produces report
- [ ] All 9 baselines produce valid, finite weights

### Sprint 3 Checklist
- [ ] `pytest tests/test_ca_engine.py` — all pass
- [ ] `pytest tests/test_ca_variants.py` — all pass
- [ ] All 5 CA variants produce correct shapes, finite values
- [ ] At least grid_ca and neural_ca have gradient flow to genome
- [ ] `python scripts/train.py --init grid_ca --steps 500` — loss decreases
- [ ] Weight heatmaps show visible structure (not noise)

### Sprint 4 Checklist
- [ ] `pytest tests/test_research_engine.py` — all pass
- [ ] Phase 1 experiment YAML validates and runs
- [ ] Auto-generated report contains tables and embedded figures
- [ ] `python scripts/run_phase.py --phase 1` completes end-to-end

### Sprint 5 Checklist
- [ ] `pytest tests/test_integration.py` — all pass
- [ ] Meta-learning outer loop shows decreasing meta-loss
- [ ] BM7 benchmark produces comparison report
- [ ] At least one CA variant shows measurable difference from baselines

### Sprint 6 Checklist
- [ ] `pytest tests/test_live_ca.py` — all pass
- [ ] All 5 live CA rules produce bounded, finite deltas
- [ ] LiveCATrainer trains without NaN for 1000+ steps with each rule
- [ ] CA-gradient alignment metric is logged and visualizable
- [ ] Multi-timescale CA fires at correct intervals
- [ ] Scope-differentiated CA applies correct rules to correct layers
- [ ] BM9 benchmark produces training dynamics report
- [ ] BM10 benchmark shows head-to-head live-vs-init comparison

### Sprint 7 Checklist
- [ ] `pytest tests/` — 100% pass
- [ ] `python scripts/run_benchmark.py --suite standard` — all benchmarks produce reports
- [ ] README has results summary with key figures
- [ ] GitHub Actions CI passes on clean clone
- [ ] All benchmark reports are coherent and cross-referenced

---

*This testing and benchmarking specification ensures that every claim made by NeuroGen is backed by reproducible, statistically sound evidence.*
