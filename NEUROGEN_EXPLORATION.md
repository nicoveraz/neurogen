# NeuroGen: CA Configuration Space Exploration Strategy

## The Search Space Problem

The space of possible CA configurations is hierarchical and mixed-type:

```
Level 0: CA Variant (discrete, 5 choices)
  │
  ├── grid_ca
  │     Level 1: Architecture (discrete/integer)
  │     │   neighborhood: {moore_3x3, moore_5x5, von_neumann, hexagonal}
  │     │   update_mlp_layers: {1, 2, 3}
  │     │   update_mlp_width: {32, 64, 128}
  │     │   n_development_steps: {16, 32, 64, 128, 256}
  │     │   seed_pattern: {center_block, diagonal, random_sparse, identity_like}
  │     │   boundary: {periodic, zero_pad, reflect}
  │     │
  │     Level 2: Continuous Parameters (the genome)
  │         MLP weights, biases → ~1K-50K floats
  │
  ├── neural_ca
  │     Level 1: Architecture
  │     │   n_channels: {8, 16, 32}
  │     │   perception: {sobel, laplacian, learned_3x3, learned_5x5}
  │     │   update_mlp_width: {64, 128, 256}
  │     │   stochastic_rate: {0.3, 0.5, 0.7, 1.0}
  │     │   n_development_steps: {32, 64, 128, 256}
  │     │   alive_threshold: {0.0, 0.1}  # cell death mechanism
  │     │
  │     Level 2: Continuous Parameters
  │         Perception filters + MLP weights → ~5K-100K floats
  │
  ├── spectral_ca
  │     Level 1: Architecture
  │     │   n_frequencies: {16, 32, 64, 128}
  │     │   ca_rule_in_frequency_domain: {true, false}
  │     │   n_development_steps: {8, 16, 32}
  │     │   frequency_init: {uniform, low_pass_bias, band_pass}
  │     │
  │     Level 2: Continuous Parameters
  │         Fourier coefficients + evolution rules → ~2K-20K floats
  │
  ├── reaction_diffusion
  │     Level 1: Architecture
  │     │   model: {gray_scott, fitzhugh_nagumo, brusselator, custom_2field}
  │     │   grid_resolution_factor: {1, 2, 4}  # develop at Nx then downsample
  │     │   n_development_steps: {100, 200, 500, 1000}
  │     │   dt: {0.1, 0.5, 1.0}
  │     │
  │     Level 2: Continuous Parameters
  │         Diffusion rates, reaction rates, feed/kill → ~10-100 floats
  │         (much smaller genome — but sensitive to exact values)
  │
  └── topo_ca
        Level 1: Architecture
        │   graph_type: {grid, small_world, scale_free, modular, hierarchical}
        │   edge_rule_mlp_width: {32, 64}
        │   n_development_steps: {16, 32, 64}
        │   initial_topology: {sparse_random, lattice, ring}
        │
        Level 2: Continuous Parameters
            Edge weight rules → ~1K-20K floats
```

**Total discrete configurations per variant:** ~100-500  
**Total continuous dimensions per configuration:** ~1K-100K  
**Brute force is impossible.** We need a structured strategy.

---

## Exploration Strategy: Three-Stage Funnel

```
Stage 1: BROAD SURVEY (cheap, many configs)
    ↓ select top performers
Stage 2: FOCUSED SEARCH (medium cost, promising regions)
    ↓ select best configs
Stage 3: DEEP OPTIMIZATION (expensive, few configs)
    ↓ final results
```

### Stage 1: Broad Survey — Random + Heuristic Sampling

**Goal:** Quickly identify which CA variants and structural hyperparameters show promise.

**Method:** For each CA variant, sample N configurations and evaluate cheaply.

**Protocol:**
1. Fix a tiny model (n_embd=64, n_layer=2) and short training budget (500 steps)
2. For each CA variant:
   a. Generate 20 random architectural configs (Level 1 params sampled uniformly)
   b. Add 5 hand-picked "informed" configs based on priors:
      - For grid_ca: small neighborhood + deep MLP; large neighborhood + shallow MLP
      - For neural_ca: high channels + few steps; low channels + many steps
      - For reaction_diffusion: known Turing pattern parameter regimes
      - etc.
   c. For each config, use 3 random genome seeds (Level 2)
   d. Develop weights → train 500 steps → record val_loss at step 500
3. Total runs: 5 variants × 25 configs × 3 seeds = 375 short training runs
4. On M1 Pro (tiny model, 500 steps ≈ 10s each): ~1 hour total

**Evaluation metric:** Rank by val_loss at step 500 (proxy for init quality)

**Output:** 
- Top 5 configs per variant (25 total)
- Identified dead zones (configs that produce NaN, collapsed weights, or divergent training)
- Variant-level ranking: which CA type is most promising overall?

```python
# research/exploration/stage1_survey.py

@dataclass
class SurveyConfig:
    n_random_configs_per_variant: int = 20
    n_heuristic_configs_per_variant: int = 5
    n_seeds: int = 3
    train_steps: int = 500
    model_size: str = "tiny"

def run_broad_survey(config: SurveyConfig) -> SurveyResults:
    """Stage 1: Evaluate many configs cheaply."""
    results = []
    for variant in CA_VARIANTS:
        configs = sample_random_configs(variant, config.n_random_configs_per_variant)
        configs += get_heuristic_configs(variant, config.n_heuristic_configs_per_variant)
        for ca_config in configs:
            for seed in range(config.n_seeds):
                val_loss = quick_evaluate(variant, ca_config, seed, config.train_steps)
                results.append(SurveyResult(variant, ca_config, seed, val_loss))
    return SurveyResults(results)
```

---

### Stage 2: Focused Search — Bayesian Optimization over Architecture

**Goal:** Find the best Level 1 (architecture) hyperparameters for the top-performing variants.

**Method:** Bayesian optimization with a surrogate model, operating on the discrete/integer architectural hyperparameters.

**Protocol:**
1. Take the top 2-3 variants from Stage 1
2. For each variant, run TPE (Tree-structured Parzen Estimators) or SMAC:
   a. Search space: all Level 1 params for that variant
   b. Objective: val_loss after 1500 steps (longer than Stage 1 for better signal)
   c. Budget: 50 evaluations per variant (each eval = 3 seeds, take mean)
   d. Use the small model config (n_embd=128, n_layer=4) for better signal
3. Total runs: 3 variants × 50 evals × 3 seeds = 450 medium training runs
4. On M1 Pro (small model, 1500 steps ≈ 90s each): ~11 hours total

**Why Bayesian optimization?**
- The architecture space is small enough (~5-7 dimensions per variant) for BO to work
- Each evaluation is expensive enough that random search wastes budget
- TPE handles mixed discrete/integer/categorical spaces natively

**Implementation option: use Optuna (lightweight, good TPE)**

```python
# research/exploration/stage2_focused.py
import optuna

def create_search_space(trial: optuna.Trial, variant: str) -> dict:
    """Define variant-specific search space."""
    if variant == "grid_ca":
        return {
            "neighborhood": trial.suggest_categorical("neighborhood",
                ["moore_3x3", "moore_5x5", "von_neumann"]),
            "mlp_layers": trial.suggest_int("mlp_layers", 1, 3),
            "mlp_width": trial.suggest_categorical("mlp_width", [32, 64, 128]),
            "n_steps": trial.suggest_int("n_steps", 16, 256, log=True),
            "seed_pattern": trial.suggest_categorical("seed_pattern",
                ["center_block", "diagonal", "random_sparse", "identity_like"]),
            "boundary": trial.suggest_categorical("boundary",
                ["periodic", "zero_pad"]),
        }
    elif variant == "neural_ca":
        return {
            "n_channels": trial.suggest_categorical("n_channels", [8, 16, 32]),
            "perception": trial.suggest_categorical("perception",
                ["sobel", "laplacian", "learned_3x3"]),
            "mlp_width": trial.suggest_categorical("mlp_width", [64, 128, 256]),
            "stochastic_rate": trial.suggest_float("stochastic_rate", 0.3, 1.0),
            "n_steps": trial.suggest_int("n_steps", 32, 256, log=True),
        }
    elif variant == "reaction_diffusion":
        return {
            "model": trial.suggest_categorical("model",
                ["gray_scott", "fitzhugh_nagumo", "brusselator"]),
            "resolution_factor": trial.suggest_int("resolution_factor", 1, 4),
            "n_steps": trial.suggest_int("n_steps", 100, 1000, log=True),
            "dt": trial.suggest_float("dt", 0.05, 1.0, log=True),
        }
    # ... etc

def objective(trial: optuna.Trial, variant: str) -> float:
    """Single evaluation: develop weights, train, return val_loss."""
    ca_config = create_search_space(trial, variant)
    losses = []
    for seed in [42, 137, 256]:
        engine = CAWeightEngine(variant, ca_config)
        model = GPT(small_config)
        weights = engine.develop_weights(model, seed=seed)
        model.set_weight_tensors(weights)
        val_loss = train_and_evaluate(model, steps=1500)
        losses.append(val_loss)
    return np.mean(losses)

def run_focused_search(top_variants: list[str], n_trials: int = 50):
    """Stage 2: Bayesian optimization over architecture."""
    results = {}
    for variant in top_variants:
        study = optuna.create_study(direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(
            lambda trial: objective(trial, variant),
            n_trials=n_trials,
        )
        results[variant] = {
            "best_params": study.best_params,
            "best_val_loss": study.best_value,
            "all_trials": study.trials,
        }
    return results
```

**What we learn:**
- Best architectural hyperparameters per variant
- Hyperparameter importance (which settings matter most?)
- Whether there's a clear winner variant or if multiple are competitive

---

### Stage 3: Deep Optimization — Meta-Learning the Genome

**Goal:** Optimize the continuous genome parameters (Level 2) for the best architecture(s) found in Stage 2.

**Method:** CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for gradient-free optimization of genome parameters, with optional gradient-based refinement.

**Why CMA-ES first?**
- The genome → develop → train → loss pipeline is not easily differentiable end-to-end
- CMA-ES handles non-smooth, noisy objectives well
- Genome size (~1K-50K params) is within CMA-ES's effective range
- No gradient computation needed through the development+training process

**Protocol:**
1. Take the top 1-3 configs (variant + architecture) from Stage 2
2. For each config, run CMA-ES on the genome parameters:
   a. Population size: 20 (lambda)
   b. Genome dimensionality: ~1K-50K (depends on variant)
   c. Objective: val_loss after 2000 steps on small model
   d. Generations: 100-200
   e. Evaluation: mean of 2 seeds per candidate (reduce noise)
3. Total inner training runs: 3 configs × 20 pop × 100 gens × 2 seeds = 12,000
   - But many are short (2000 steps on small model ≈ 2min on M1 Pro)
   - With 12,000 × 2min = 400 hours → needs parallelism or clever budgeting
4. **Budget reduction strategies:**
   a. Early stopping: kill candidates that diverge within 200 steps
   b. Progressive training: start with 500-step evals, increase to 2000 as search narrows
   c. Warm-starting: initialize CMA-ES mean from best Stage 2 genome
   d. Subspace search: PCA on genome params, optimize in reduced dimensions

```python
# neurogen/training/meta_trainer.py
import cma

@dataclass
class MetaTrainConfig:
    population_size: int = 20
    max_generations: int = 100
    inner_train_steps: int = 2000
    inner_seeds: int = 2
    sigma0: float = 0.1               # initial CMA-ES step size
    early_stop_loss: float = 10.0     # kill candidates with loss above this
    progressive_schedule: dict = field(default_factory=lambda: {
        0: 500,     # generations 0-29: 500 inner steps
        30: 1000,   # generations 30-59: 1000 inner steps  
        60: 2000,   # generations 60+: 2000 inner steps
    })

class CMAESMetaTrainer:
    """Optimize CA genome parameters using CMA-ES."""

    def __init__(self, variant: str, arch_config: dict, meta_config: MetaTrainConfig):
        self.variant = variant
        self.arch_config = arch_config
        self.meta_config = meta_config
        self.engine = CAWeightEngine(variant, arch_config)
        self.genome_dim = self.engine.genome_size()

    def evaluate_genome(self, genome_params: np.ndarray, generation: int) -> float:
        """Inner loop: set genome, develop weights, train, return val_loss."""
        self.engine.set_genome_params(genome_params)

        inner_steps = self._get_progressive_steps(generation)
        losses = []
        for seed in range(self.meta_config.inner_seeds):
            model = GPT(small_config).to(device)
            weights = self.engine.develop_weights(model, seed=seed)
            model.set_weight_tensors(weights)

            val_loss = train_and_evaluate(model, steps=inner_steps)

            if val_loss > self.meta_config.early_stop_loss:
                return val_loss  # early terminate bad candidates

            losses.append(val_loss)
        return np.mean(losses)

    def run(self) -> MetaTrainResults:
        """Outer loop: CMA-ES optimization of genome."""
        x0 = self.engine.get_genome_params()  # initial genome (random or warm-start)
        es = cma.CMAEvolutionStrategy(x0, self.meta_config.sigma0, {
            'popsize': self.meta_config.population_size,
            'maxiter': self.meta_config.max_generations,
            'seed': 42,
        })

        history = []
        for generation in range(self.meta_config.max_generations):
            solutions = es.ask()
            fitnesses = [
                self.evaluate_genome(s, generation) for s in solutions
            ]
            es.tell(solutions, fitnesses)

            history.append({
                'generation': generation,
                'best_fitness': min(fitnesses),
                'mean_fitness': np.mean(fitnesses),
                'std_fitness': np.std(fitnesses),
                'sigma': es.sigma,
            })

            if es.stop():
                break

        return MetaTrainResults(
            best_genome=es.result.xbest,
            best_fitness=es.result.fbest,
            history=history,
        )

    def _get_progressive_steps(self, generation: int) -> int:
        """More training steps as search progresses."""
        steps = 500
        for gen_threshold, step_count in sorted(self.meta_config.progressive_schedule.items()):
            if generation >= gen_threshold:
                steps = step_count
        return steps
```

#### Alternative: Gradient-Based Meta-Learning (when genome is small)

For reaction-diffusion CA (~10-100 params), direct gradient-based optimization is feasible:

```python
class DifferentiableMetaTrainer:
    """When we can differentiate through development + short training."""

    def meta_step(self):
        # 1. Get current genome parameters (requires_grad=True)
        genome_params = self.engine.get_genome_params_tensor()

        # 2. Develop weights (differentiable)
        weights = self.engine.develop_weights_differentiable(model)
        model.set_weight_tensors(weights)

        # 3. Short inner training (K steps of gradient descent)
        #    Using higher-order gradients (expensive but exact)
        for k in range(K_inner_steps):
            loss = model.forward_loss(batch)
            inner_grads = torch.autograd.grad(loss, model.parameters(),
                                               create_graph=True)
            # Manual SGD step (to keep computation graph)
            for p, g in zip(model.parameters(), inner_grads):
                p.data = p - inner_lr * g

        # 4. Evaluate on validation set
        val_loss = model.forward_loss(val_batch)

        # 5. Backpropagate through everything to genome
        val_loss.backward()

        # 6. Update genome
        genome_optimizer.step()
```

**When to use which:**
- CMA-ES: default, always works, genome size up to ~50K params
- Differentiable: only for small genomes (<1K params) with short inner loops (<50 steps)
- Reptile-style: middle ground — no second-order gradients, genome up to ~10K params

---

### Stage 3b: Co-Evolutionary Search (Phase 6)

After finding good static genomes, explore whether the CA should remain active during training:

```python
@dataclass
class CoEvolutionConfig:
    ca_apply_interval: int = 100     # apply CA perturbation every N steps
    ca_perturbation_scale: float = 0.01  # magnitude of CA influence
    ca_decay_rate: float = 0.95      # decay perturbation over training
    ca_scope: str = "all"            # "all", "attention_only", "ffn_only"

class CoEvolutionSearchSpace:
    """What to search in the co-evolution setting."""
    params = {
        "apply_interval": [50, 100, 200, 500],
        "perturbation_scale": [0.001, 0.005, 0.01, 0.05],
        "decay_rate": [0.9, 0.95, 0.99, 1.0],
        "scope": ["all", "attention_only", "ffn_only"],
        "mode": [
            "additive",        # w += ca_perturbation * scale
            "interpolate",     # w = (1-alpha)*w + alpha*ca_developed
            "regularize",      # loss += lambda * ||w - ca_developed||
        ],
    }
```

---

## Exploration Budget Planning (M1 Pro 16GB)

| Stage | Runs | Time per run | Total time | What we learn |
|-------|------|-------------|------------|---------------|
| 1: Broad Survey | 375 | ~10s (tiny, 500 steps) | ~1 hr | Which variants/configs are viable |
| 2: Focused Search | 450 | ~90s (small, 1500 steps) | ~11 hr | Best architecture per variant |
| 3: CMA-ES Meta | 12,000 | ~2 min (small, 2000 steps) | ~400 hr raw | Best genome params |
| 3 (with budget tricks) | ~2,000 | ~90s avg (progressive) | ~50 hr | Same, with early stopping |
| 3b: Co-Evolution | ~200 | ~5 min (small, 5000 steps) | ~17 hr | Whether co-evolution helps |

**Realistic total: ~80 hours of compute** for a thorough search on M1 Pro.

### Budget Reduction Strategies

1. **Progressive evaluation:** Start Stage 3 with 500-step inner evals, increase to 2000 as CMA-ES converges. Saves ~60% of compute.

2. **Early stopping:** Kill any candidate with loss > 2× best known after 200 steps. Saves ~30% of bad evaluations.

3. **Warm-starting:** Initialize CMA-ES from Stage 2's best random genome, not from scratch. Reduces generations needed by ~40%.

4. **Subspace CMA-ES:** For large genomes (neural_ca, ~50K params), project into a lower-dimensional subspace using random projections or PCA. Search in 500D instead of 50KD.

5. **Transfer across sizes:** Optimize genome on tiny model, validate on small model, run final evaluation on default model. Each stage filters candidates before scaling up.

6. **Parallelism:** CMA-ES population evaluation is embarrassingly parallel. If running on a Mac Studio M4 Max (your target for Eunosia), 10 parallel evaluations cut Stage 3 time by 5-8×.

---

## Search Space Visualization & Tracking

Every exploration stage should produce:

### Stage 1 Output
- Heatmap: variant × metric showing average performance
- Scatter plot: genome_size vs val_loss for all configs
- Failure mode catalog: which configs produced NaN, divergence, or collapse
- Decision: which 2-3 variants advance to Stage 2

### Stage 2 Output
- Optuna visualization: hyperparameter importance plot per variant
- Parallel coordinates plot: all trials colored by val_loss
- Slice plots: val_loss vs each hyperparameter (marginal effects)
- Interaction analysis: do any hyperparameters interact strongly?
- Decision: best 1-3 (variant, architecture) configs for Stage 3

### Stage 3 Output  
- CMA-ES convergence curve (meta-loss per generation)
- Genome parameter evolution: PCA of population over generations
- Developed weight heatmaps: how do weights change as genome improves?
- Final comparison: best CA init vs all baselines (full benchmark suite)

---

## Implementation Files

```
neurogen/
├── exploration/
│   ├── __init__.py
│   ├── stage1_survey.py        # Broad random+heuristic survey
│   ├── stage2_focused.py       # Bayesian optimization (Optuna)
│   ├── stage3_meta.py          # CMA-ES genome optimization
│   ├── coevolution.py          # Co-evolutionary search
│   ├── budget.py               # Compute budget estimation and tracking
│   └── visualization.py        # Exploration-specific plots

research/
├── experiments/
│   ├── exploration_stage1.yaml
│   ├── exploration_stage2.yaml
│   ├── exploration_stage3.yaml
│   └── exploration_coevolution.yaml

scripts/
├── run_exploration.py           # Run specific exploration stage
├── exploration_report.py        # Generate exploration summary
```

### Additional Dependency

```
# Add to pyproject.toml [project.optional-dependencies]
search = ["optuna>=3.4.0"]
```

---

## Key Insight: Why Not Just Random Search?

For Level 2 (continuous genome params), random search is catastrophically inefficient — a 10K-dimensional genome explored randomly would need astronomical samples. CMA-ES works because it learns the covariance structure of the search space, effectively discovering which directions in genome-space matter.

For Level 1 (architecture), random search is actually decent as a starting point (Stage 1), but Bayesian optimization (Stage 2) quickly outperforms it by building a surrogate model of "which architectural choices tend to produce lower loss."

The three-stage funnel ensures we spend expensive compute only on promising regions, not wasting M1 Pro hours on dead-end configurations.
