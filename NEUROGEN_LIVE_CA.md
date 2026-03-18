# NeuroGen: Live CA — Cellular Automata Operating Within Training

## The Idea

Instead of using the CA only at initialization, the CA runs **alongside** gradient descent at every training step (or every K steps). Two forces shape the weights simultaneously:

```
Standard Training:
    w(t+1) = w(t) - lr * ∇L(w(t))
    (one force: gradient descent)

Live CA Training:
    w(t+1) = w(t) - lr * ∇L(w(t)) + α(t) * CA(w(t))
    (two forces: gradient descent + cellular automaton)
```

This is analogous to the brain, where:
- **Gradient descent** ≈ Hebbian/anti-Hebbian learning (error-driven synaptic updates)
- **CA rules** ≈ developmental programs, homeostatic plasticity, synaptic pruning, neurotropic factors — local rules that operate on the weight structure itself, not on the loss

The CA "sees" the current weights as its grid state and applies local update rules, producing a delta that gets blended with the gradient update. The two processes can cooperate (CA regularizes toward useful structure) or compete (CA pushes toward structure that gradient descent must work around).

---

## Biological Motivation

The brain runs multiple concurrent optimization processes at different timescales:

| Timescale | Brain Process | NeuroGen Analog |
|-----------|--------------|-----------------|
| Milliseconds | Synaptic transmission | Forward pass |
| Seconds | Short-term plasticity (facilitation/depression) | Gradient step |
| Minutes-hours | LTP/LTD (Hebbian learning) | Multiple gradient steps |
| Hours-days | Synaptic scaling (homeostasis) | CA applied periodically |
| Days-weeks | Structural plasticity (spine growth/pruning) | CA modifying weight structure |
| Weeks-months | Myelination changes | CA modifying connectivity patterns |
| Development | Genetic program (axon guidance, arealization) | CA initialization |

**Key insight:** The brain's "training loop" is not just gradient descent. It's gradient descent + multiple local rules operating at different frequencies. NeuroGen's live CA captures this multi-process dynamic.

---

## Architecture: The Dual-Process Training Loop

```
┌─────────────────────────────────────────────────────────┐
│                   TRAINING STEP t                        │
│                                                          │
│  ┌─────────────┐                                         │
│  │  Forward     │──▶ loss                                │
│  │  Pass        │                                        │
│  └─────────────┘                                         │
│         │                                                │
│         ▼                                                │
│  ┌─────────────┐        ┌──────────────────┐            │
│  │  Backward    │        │  CA Step          │            │
│  │  Pass        │        │                   │            │
│  │              │        │  Read: w(t)       │            │
│  │  ∇L(w)      │        │  Apply: local     │            │
│  │              │        │    update rules   │            │
│  │              │        │  Output: Δw_ca    │            │
│  └──────┬──────┘        └────────┬─────────┘            │
│         │                        │                       │
│         ▼                        ▼                       │
│  ┌──────────────────────────────────────┐               │
│  │          WEIGHT UPDATE               │               │
│  │                                      │               │
│  │  w(t+1) = w(t)                       │               │
│  │           - lr * ∇L(w)     ← gradient│               │
│  │           + α(t) * Δw_ca   ← CA      │               │
│  │                                      │               │
│  └──────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

---

## Live CA Modes

### Mode 1: Additive Perturbation (simplest)

```python
# Every step (or every K steps):
grad_update = -lr * grad
ca_update = alpha * ca.step(current_weights)
new_weights = weights + grad_update + ca_update
```

The CA produces a delta based on the current weight landscape. Alpha decays over training (strong developmental signal early, fading as learning takes over).

**Biological analog:** Neurotropic factors that guide axon growth early in development, then fade.

### Mode 2: Homeostatic Regulation

```python
# CA enforces target statistics on weights
for each weight matrix W:
    local_stats = ca.compute_local_statistics(W)  # mean, std per neighborhood
    target_stats = ca.target_statistics            # learned or fixed
    correction = ca.homeostatic_rule(local_stats, target_stats)
    W += alpha * correction
```

The CA doesn't push weights in a specific direction — it pushes them toward target *statistics*. If a region's weights are too large, CA damps them. If too uniform, CA increases variance. If too correlated, CA decorrelates.

**Biological analog:** Synaptic scaling — neurons adjust all their synapses to maintain a target firing rate, independent of what Hebbian learning is doing.

### Mode 3: Structural Pruning/Growth

```python
# CA decides which weights should be active
for each weight matrix W:
    mask = ca.step(W)  # binary or soft mask from CA
    W *= mask           # prune "dead" connections
    # Optionally: regrow pruned connections if CA says so
```

The CA acts as a dynamic sparsity controller. Gradient descent sets the weight values, but the CA decides which weights *exist*. Over training, the CA discovers which connectivity patterns support learning.

**Biological analog:** Synaptic pruning during development — the brain starts overconnected and the developmental program selectively removes connections based on local activity patterns.

### Mode 4: Multi-Timescale CA (most biologically faithful)

```python
# Multiple CA processes at different frequencies
class MultiTimescaleCA:
    def __init__(self):
        self.fast_ca = HomeostaticCA()       # every step
        self.medium_ca = StructuralCA()      # every 100 steps
        self.slow_ca = DevelopmentalCA()      # every 1000 steps

    def step(self, weights, training_step):
        delta = torch.zeros_like(weights)

        # Fast: homeostatic correction (always)
        delta += self.fast_ca.step(weights) * alpha_fast

        # Medium: structural adjustment (periodic)
        if training_step % 100 == 0:
            delta += self.medium_ca.step(weights) * alpha_medium

        # Slow: large-scale reorganization (rare)
        if training_step % 1000 == 0:
            delta += self.slow_ca.step(weights) * alpha_slow

        return delta
```

Each CA operates at a different timescale with different rules:
- **Fast CA:** small corrections, prevents gradient pathologies
- **Medium CA:** structural adjustments, promotes modularity
- **Slow CA:** large-scale reorganization, consolidation

**Biological analog:** The full stack — synaptic scaling (fast), structural plasticity (medium), myelination and large-scale reorganization (slow).

### Mode 5: CA as Learned Optimizer

```python
# The CA replaces or augments the optimizer
class CAOptimizer:
    """CA-based update rule that learns to optimize."""

    def step(self, weights, gradients):
        # CA sees both the weights AND the gradients as state
        combined_state = torch.stack([weights, gradients], dim=-1)
        ca_output = self.ca.step(combined_state)

        # CA output IS the weight update (not blended, it IS the optimizer)
        return weights + ca_output
```

The CA takes both the current weights and the current gradients as input and produces the weight update directly. This makes the CA a *learned optimizer* — it can implement momentum, adaptive learning rates, weight decay, etc., all as emergent behavior from local rules.

**Biological analog:** Neuromodulation — dopamine, serotonin, etc. modulate synaptic plasticity rules based on both the synapse state and the error signal.

---

## The CA Weight-Space View

The key conceptual shift: **treat the weight matrix as a 2D cellular automaton grid.**

Each cell (i, j) in the weight matrix has:
- **State:** the weight value w[i,j]
- **Neighborhood:** surrounding weights w[i±1, j±1] (or learned neighborhood)
- **External input:** the gradient ∇L[i,j] at this position
- **Update rule:** a function of state + neighborhood + gradient

```python
class WeightSpaceCA:
    """Treats each weight matrix as a CA grid."""

    def __init__(self, rule_net: nn.Module, neighborhood_size: int = 3):
        self.rule_net = rule_net  # small MLP: the genome
        self.k = neighborhood_size

    def step(self, W: torch.Tensor, grad_W: torch.Tensor = None) -> torch.Tensor:
        """Single CA step over the weight matrix."""
        H, W_dim = W.shape

        # Perception: gather neighborhood features
        # Each cell sees its local neighborhood statistics
        padded = F.pad(W, [self.k//2]*4, mode='circular')
        neighborhoods = padded.unfold(0, self.k, 1).unfold(1, self.k, 1)
        # neighborhoods: (H, W_dim, k, k)

        # Compute local features per cell
        local_mean = neighborhoods.mean(dim=(-2, -1))
        local_std = neighborhoods.std(dim=(-2, -1))
        local_max = neighborhoods.amax(dim=(-2, -1))
        local_min = neighborhoods.amin(dim=(-2, -1))
        center_val = W

        # Build feature vector per cell
        features = [center_val, local_mean, local_std, local_max, local_min]

        if grad_W is not None:
            # CA can also see the gradient landscape
            grad_padded = F.pad(grad_W, [self.k//2]*4, mode='circular')
            grad_neigh = grad_padded.unfold(0, self.k, 1).unfold(1, self.k, 1)
            grad_mean = grad_neigh.mean(dim=(-2, -1))
            grad_magnitude = grad_W.abs()
            features.extend([grad_mean, grad_magnitude])

        feature_stack = torch.stack(features, dim=-1)  # (H, W_dim, n_features)

        # Apply update rule (same MLP at every cell — weight sharing)
        delta = self.rule_net(feature_stack).squeeze(-1)  # (H, W_dim)

        return delta
```

### What the CA "Sees" at Each Cell

| Feature | Meaning | Why it matters |
|---------|---------|---------------|
| `center_val` | This weight's value | Self-reference |
| `local_mean` | Average of neighbors | Is this weight an outlier? |
| `local_std` | Variance of neighbors | Is this a high-variance region? |
| `local_max/min` | Range of neighbors | Are there extreme values nearby? |
| `gradient` | Error signal at this weight | Should this weight change for loss? |
| `grad_mean` | Avg gradient of neighbors | Is the gradient uniform or structured? |
| `step_number` | Current training step | Temporal awareness (decay behavior) |

---

## Integration with the Training Loop

```python
# neurogen/training/live_ca_trainer.py

class LiveCATrainer:
    """Training loop with CA operating at every step."""

    def __init__(self, model, ca_engine, config: LiveCAConfig):
        self.model = model
        self.ca = ca_engine
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    def train_step(self, batch, step: int):
        # 1. Standard forward + backward
        x, y = batch
        logits, loss = self.model(x, y)
        loss.backward()

        # 2. Collect gradients before optimizer step
        gradients = {}
        if self.config.ca_sees_gradients:
            for name, param in self.model.named_parameters():
                if name in self.ca.target_params:
                    gradients[name] = param.grad.clone()

        # 3. Standard optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # 4. CA step (after gradient update)
        if step % self.config.ca_interval == 0:
            alpha = self.config.get_alpha(step)  # decaying CA influence

            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in self.ca.target_params:
                        grad = gradients.get(name, None)
                        ca_delta = self.ca.step(param.data, grad_W=grad)

                        # Apply CA update with scaling
                        param.data += alpha * ca_delta

                        # Optional: clamp to prevent CA from destabilizing
                        if self.config.clamp_weights:
                            param.data.clamp_(-self.config.max_weight, self.config.max_weight)

        return loss.item()
```

### Alpha Schedule: The Developmental Curve

The CA influence should follow a biologically-inspired schedule:

```python
@dataclass
class AlphaSchedule:
    """Controls CA influence over training."""

    mode: str = "developmental"  # how alpha changes over time

    # Option 1: Exponential decay (development fades)
    # alpha(t) = alpha_0 * exp(-decay * t)
    alpha_0: float = 0.01
    decay: float = 0.001

    # Option 2: Cosine anneal (smooth fade)
    # alpha(t) = alpha_0 * 0.5 * (1 + cos(pi * t / T))

    # Option 3: Step function (developmental phases)
    # alpha(t) = alpha_phase[current_phase(t)]
    phase_boundaries: list = field(default_factory=lambda: [0, 1000, 3000])
    phase_alphas: list = field(default_factory=lambda: [0.01, 0.005, 0.001])

    # Option 4: Adaptive (CA influence depends on loss dynamics)
    # alpha(t) = alpha_0 * f(loss_improvement_rate)
    # If loss is improving fast → reduce CA (gradient is doing fine)
    # If loss is stagnating → increase CA (shake things up)
    adaptive_sensitivity: float = 1.0

    # Option 5: Cyclic (periodic developmental bursts)
    # alpha(t) = alpha_base + amplitude * sin(2π * t / period)
    # Mimics sleep/wake consolidation cycles
    cycle_period: int = 500
    cycle_amplitude: float = 0.005
    alpha_base: float = 0.002

    def get_alpha(self, step: int, loss_history: list = None) -> float:
        if self.mode == "exponential_decay":
            return self.alpha_0 * math.exp(-self.decay * step)
        elif self.mode == "cosine":
            return self.alpha_0 * 0.5 * (1 + math.cos(math.pi * step / self.total_steps))
        elif self.mode == "phased":
            for i, boundary in enumerate(self.phase_boundaries):
                if step < boundary:
                    return self.phase_alphas[max(0, i-1)]
            return self.phase_alphas[-1]
        elif self.mode == "adaptive":
            if loss_history and len(loss_history) > 10:
                recent_improvement = loss_history[-10] - loss_history[-1]
                if recent_improvement < 0.01:  # stagnating
                    return self.alpha_0 * 2.0
                else:
                    return self.alpha_0 * 0.5
            return self.alpha_0
        elif self.mode == "cyclic":
            return self.alpha_base + self.cycle_amplitude * math.sin(
                2 * math.pi * step / self.cycle_period)
```

---

## Specific Live CA Rules to Implement

### Rule 1: Local Weight Normalization (homeostatic)

```python
class LocalNormCA:
    """Each cell normalizes toward local statistics.
    Prevents gradient explosion/vanishing at the local level."""

    def step(self, W, grad_W=None):
        local_mean = neighborhood_mean(W, k=3)
        local_std = neighborhood_std(W, k=3)
        target_std = 0.02  # healthy weight scale for transformers

        # Push toward target statistics
        mean_correction = -0.1 * (W - local_mean)         # reduce outliers
        std_correction = (target_std / (local_std + 1e-8) - 1) * W * 0.01  # normalize scale

        return mean_correction + std_correction
```

### Rule 2: Modularity Enforcer

```python
class ModularityCA:
    """Encourages block-diagonal structure.
    Weights within blocks are strengthened, between blocks are weakened."""

    def __init__(self, n_blocks: int = 6):
        self.n_blocks = n_blocks

    def step(self, W, grad_W=None):
        H, W_dim = W.shape
        bh, bw = H // self.n_blocks, W_dim // self.n_blocks
        delta = torch.zeros_like(W)

        for i in range(self.n_blocks):
            for j in range(self.n_blocks):
                block = W[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                if i == j:
                    # On-diagonal blocks: reinforce (strengthen)
                    delta[i*bh:(i+1)*bh, j*bw:(j+1)*bw] = block * 0.001
                else:
                    # Off-diagonal blocks: decay (weaken)
                    delta[i*bh:(i+1)*bh, j*bw:(j+1)*bw] = -block * 0.002

        return delta
```

### Rule 3: Gradient-Aware Pruning CA

```python
class PruningCA:
    """Dynamically prunes weights based on local utility.
    Small weights with small gradients are pushed to zero.
    Active weights (large gradient) are preserved."""

    def step(self, W, grad_W):
        importance = W.abs() * grad_W.abs()  # magnitude × gradient = utility
        local_importance = neighborhood_mean(importance, k=3)

        threshold = local_importance.median()

        # Low-importance weights decay toward zero
        decay_mask = (importance < threshold).float()
        growth_mask = (importance >= threshold).float()

        delta = -0.01 * W * decay_mask  # prune unimportant
        # Important weights get a small boost
        delta += 0.001 * W.sign() * growth_mask

        return delta
```

### Rule 4: Competition CA (lateral inhibition)

```python
class CompetitionCA:
    """Nearby weights compete. The strongest suppress their neighbors.
    Produces sparse, winner-take-all connectivity patterns.
    Biological analog: lateral inhibition in cortical columns."""

    def step(self, W, grad_W=None):
        local_max = neighborhood_max(W.abs(), k=5)
        is_local_winner = (W.abs() >= local_max * 0.95).float()

        # Winners get strengthened, losers get suppressed
        delta = 0.001 * W * is_local_winner + (-0.005) * W * (1 - is_local_winner)

        return delta
```

### Rule 5: Learned CA Rule (the genome)

```python
class LearnedCA:
    """The CA rule itself is a small neural network (the genome).
    This is the most general — it can learn any of the above rules
    and potentially discover novel ones."""

    def __init__(self, n_features=7, hidden=64):
        self.rule_net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),  # bounded output
        )

    def step(self, W, grad_W=None):
        features = compute_cell_features(W, grad_W)  # (H, W_dim, n_features)
        delta = self.rule_net(features).squeeze(-1)    # (H, W_dim)
        return delta * 0.01  # small scale
```

---

## CA Operating on Different Weight Types

Not all weights should get the same CA treatment:

```python
CA_SCOPE_CONFIG = {
    # Attention weights: encourage head specialization
    "attn.q_proj": {"ca": CompetitionCA, "alpha": 0.01},
    "attn.k_proj": {"ca": CompetitionCA, "alpha": 0.01},
    "attn.v_proj": {"ca": ModularityCA, "alpha": 0.005},
    "attn.out_proj": {"ca": LocalNormCA, "alpha": 0.005},

    # FFN weights: encourage sparsity and structure
    "ffn.w1": {"ca": PruningCA, "alpha": 0.01},
    "ffn.w2": {"ca": PruningCA, "alpha": 0.01},

    # Embeddings: light touch only
    "tok_emb": {"ca": LocalNormCA, "alpha": 0.001},
}
```

This reflects neuroscience: different brain regions have different developmental programs. Visual cortex develops differently from prefrontal cortex, even though both are sheets of neurons.

---

## Interaction Effects: CA + Gradient Descent

The two forces can interact in interesting ways:

### Cooperation
- CA promotes weight structure that makes the loss landscape smoother → gradient descent converges faster
- Gradient descent identifies useful features → CA preserves and reinforces them
- CA prevents gradient pathologies (explosion, vanishing) → more stable training

### Competition
- CA pushes toward structure, gradient descent pushes toward performance → creative tension
- If CA is too strong, it fights the gradient and hurts performance
- If gradient is too strong, it overwrites CA structure within steps

### Emergent Behaviors (hypothesized)
- **Phase transitions:** CA + gradient might produce sudden jumps in capability (like "grokking") when CA structure and learned features align
- **Self-organizing criticality:** the system might naturally find edge-of-chaos dynamics where both forces balance
- **Curriculum effects:** early CA dominance → late gradient dominance mimics how biological development creates a scaffold that learning then populates

---

## What to Measure (Live CA-Specific Metrics)

Beyond standard training metrics, Live CA requires:

| Metric | What it tells us |
|--------|-----------------|
| `ca_delta_magnitude` | How much is the CA actually changing weights? |
| `gradient_delta_magnitude` | How much is the gradient changing weights? |
| `ca_gradient_alignment` | cos(Δw_ca, Δw_grad) — cooperation or competition? |
| `ca_delta_vs_alpha` | Is the CA effect dominated by alpha schedule or rule dynamics? |
| `weight_structure_score` | Does structure (modularity, sparsity) increase over training? |
| `ca_contribution_ratio` | ||Δw_ca|| / (||Δw_ca|| + ||Δw_grad||) — who dominates? |
| `loss_with_vs_without_ca_step` | Ablation: skip CA for one step, measure loss difference |
| `attention_specialization_rate` | How fast do heads diverge? Faster with CA? |

### Critical Diagnostic: The CA-Gradient Alignment Curve

```python
def measure_alignment(ca_delta, grad_delta):
    """Cosine similarity between CA update and gradient update.
    +1 = CA helps gradient (cooperation)
     0 = orthogonal (independent)
    -1 = CA fights gradient (competition)
    """
    flat_ca = ca_delta.flatten()
    flat_grad = grad_delta.flatten()
    return F.cosine_similarity(flat_ca.unsqueeze(0), flat_grad.unsqueeze(0)).item()
```

Plotting this over training should reveal:
- Early training: alignment may be low (CA has its own agenda)
- Mid training: alignment should increase (CA structure becomes useful for loss)
- Late training: alignment stabilizes or CA influence fades

If alignment is persistently negative → the CA rule is fighting learning → bad rule.

---

## Implementation Plan

### New Files

```
neurogen/
├── ca/
│   ├── live/
│   │   ├── __init__.py
│   │   ├── base.py              # LiveCA base class
│   │   ├── local_norm.py        # Rule 1: homeostatic
│   │   ├── modularity.py        # Rule 2: block-diagonal enforcer
│   │   ├── pruning.py           # Rule 3: gradient-aware pruning
│   │   ├── competition.py       # Rule 4: lateral inhibition
│   │   ├── learned.py           # Rule 5: learned genome rule
│   │   ├── multi_timescale.py   # Mode 4: multiple CAs at different frequencies
│   │   ├── ca_optimizer.py      # Mode 5: CA as optimizer
│   │   └── alpha_schedule.py    # Alpha decay schedules
│   └── ...
├── training/
│   ├── live_ca_trainer.py       # Modified training loop with CA integration
│   └── ...
```

### New Experiment Phases

**Phase 9: Live CA — Fixed Rules**
- Test each hand-designed rule (Rules 1-4) independently
- Test each alpha schedule
- Compare against init-only CA and baseline
- Question: does *any* fixed rule help during training?

**Phase 10: Live CA — Learned Rules**
- Meta-learn the CA rule (Rule 5 genome) using CMA-ES
- The genome is now optimized for live operation, not just initialization
- Compare: learned live rule vs best fixed rule vs init-only

**Phase 11: Live CA — Multi-Timescale**
- Combine multiple CA rules at different frequencies
- Search over: which rules at which timescales
- Question: does the multi-timescale approach outperform single-rule?

**Phase 12: Live CA — Scope Differentiation**
- Different CA rules for different layer types (attention vs FFN vs embeddings)
- Question: does scope-specific CA outperform uniform CA?

### New Benchmarks

**BM9: Live CA Training Dynamics**
- CA-gradient alignment curve over training
- CA delta magnitude vs gradient delta magnitude
- Weight structure evolution (modularity score, sparsity, spectral properties)
- Per-layer CA contribution ratio

**BM10: Live CA vs Init-Only CA**
- Head-to-head: same CA rule used at init-only vs live
- Loss curves, convergence speed, final quality
- The critical comparison for this entire investigation

---

## Compute Budget (M1 Pro 16GB)

Live CA adds overhead per training step:

| CA Rule | Overhead per step | Notes |
|---------|------------------|-------|
| LocalNormCA | ~5% | Just neighborhood stats |
| ModularityCA | ~3% | Block iteration, simple math |
| PruningCA | ~8% | Requires gradient storage |
| CompetitionCA | ~10% | Max-pooling neighborhoods |
| LearnedCA | ~15-25% | MLP forward per cell |
| MultiTimescale | ~10-30% | Depends on active CAs this step |

For the default model (10M params) on M1 Pro:
- Standard training: ~20 min for 5K steps
- Live CA (simple rule): ~22 min for 5K steps
- Live CA (learned rule): ~25 min for 5K steps
- Live CA (multi-timescale): ~26 min for 5K steps

The overhead is very manageable. The real cost is in the exploration — searching over rules, schedules, and scopes multiplies the number of runs needed.

---

## Key Insight: The CA as Inductive Bias During Training

Standard regularization techniques (dropout, weight decay, spectral norm) are simple, global rules applied uniformly. Live CA is **learned, local, and structure-aware** regularization. It's a much richer class of inductive bias.

If this works, the implication is profound: instead of designing regularizers by hand (L2, dropout, etc.), you can meta-learn a *developmental program* that discovers the right structural biases for a given task family. The CA genome becomes a transferable artifact — "this small rule set, when run alongside gradient descent, makes language models train faster."

That's the biological claim made computational: **evolution doesn't optimize weights, it optimizes the developmental and homeostatic programs that shape how weights are organized during learning.**
