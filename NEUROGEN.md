# NeuroGen: Research Reference

## Core Idea

The brain does not learn from noise. Before experience begins, genetic developmental programs — small local rules iterating on a cellular substrate — grow *functional structure* into neural tissue. Not specific memories, but organizational principles that make learning efficient: modularity, specialization, long-range connectivity, competition, homeostasis, hierarchical processing.

The language circuit illustrates this perfectly:

- **Production (Broca's, frontal):** Pre-wired for sequential planning, hierarchical structure building, motor output coordination. Functions *before* a child speaks.
- **Comprehension (Wernicke's, temporal):** Pre-wired for sound-to-meaning mapping, pattern recognition, associative retrieval. Functions *before* a child understands words.
- **Connection (Arcuate fasciculus):** Long-range fiber bundle linking production and comprehension. The highway exists *before* any traffic flows.
- **Integration (Angular gyrus):** Bridges modalities — connecting visual input (reading) with auditory/semantic representations. Cross-modal integration pre-wired.
- **Modulation (Right hemisphere):** Prosody, emotion, metaphor, pragmatics — the "color" of language. Separate processing stream, different functional signature.

When these areas are damaged (aphasia), the deficits are *specific*: Broca's damage → can't produce fluent speech but comprehension is intact. Wernicke's damage → fluent speech but meaningless. This proves the functional specialization exists independently of learned content.

**NeuroGen's hypothesis:** A CA can grow analogous *functional* specialization into transformer weights — creating regions biased toward sequential processing, associative retrieval, long-range routing, cross-modal integration, and competitive selection — before gradient descent begins. This scaffold should make learning faster and/or better.

**Metric:** `val_bpb` (validation bits per byte). Lower is better.

---

## Functional Principles → Weight Structures → CA Rules

The brain's language circuit reveals seven functional principles. Each maps to a concrete weight structure in a transformer, and each can be produced by a specific CA developmental rule.

### Principle 1: Functional Specialization

**In the brain:** Different regions do different things (Broca's ≠ Wernicke's). This specialization exists before learning. A neuron in Broca's area is *structurally different* from one in Wernicke's — different dendritic trees, different local connectivity, different neurotransmitter profiles.

**In a transformer:** Different attention heads *can* specialize (some track local syntax, others do long-range retrieval, others copy tokens). But with random init, all heads start identical and must discover specialization through gradient descent alone.

**CA approach:** Initialize different heads with different structural signatures. Not identical random noise for every head — diverse seeds and development rules that create distinct functional biases:

```python
def specialized_heads_init(n_heads, head_dim, seq_len):
    """Each head gets a different CA seed → different functional bias."""
    head_weights = []
    for h in range(n_heads):
        if h % 4 == 0:
            # "Local processing" head — strong near-diagonal (like Broca's: sequential/local)
            w = ca_develop(shape=(head_dim, head_dim), seed="diagonal_band", n_steps=32)
        elif h % 4 == 1:
            # "Associative" head — distributed connectivity (like Wernicke's: pattern matching)
            w = ca_develop(shape=(head_dim, head_dim), seed="distributed", n_steps=64)
        elif h % 4 == 2:
            # "Long-range" head — off-diagonal bands (like arcuate fasciculus: bridging)
            w = ca_develop(shape=(head_dim, head_dim), seed="off_diagonal", n_steps=48)
        elif h % 4 == 3:
            # "Integration" head — smooth gradients (like angular gyrus: cross-modal)
            w = ca_develop(shape=(head_dim, head_dim), seed="gradient", n_steps=32)
        head_weights.append(w)
    return head_weights
```

### Principle 2: Hierarchical Processing

**In the brain:** Language processing flows through a hierarchy. Raw auditory input → phoneme recognition → word recognition → syntactic parsing → semantic integration → pragmatic interpretation. Each stage transforms representations, and earlier stages are more local/concrete while later stages are more global/abstract.

**In a transformer:** Earlier layers tend to learn local patterns (character/token adjacency, syntax), later layers learn global patterns (semantics, long-range dependencies). But this hierarchy emerges slowly from uniform init.

**CA approach:** Initialize weight matrices with *depth-dependent* CA rules. The CA rule itself changes based on layer position — earlier layers get CAs that produce local structure, later layers get CAs that produce distributed structure:

```python
def hierarchical_init(model, n_layers):
    for layer_idx in range(n_layers):
        # Locality ratio: 1.0 at layer 0 (fully local), 0.0 at last layer (fully distributed)
        locality = 1.0 - layer_idx / (n_layers - 1)
        ca_config = {
            "neighborhood_size": 3 if locality > 0.5 else 5,
            "n_steps": int(32 + 64 * (1 - locality)),  # more steps → more complex
            "seed": "local_clusters" if locality > 0.5 else "distributed_sparse",
        }
        weights = ca_develop(target_shapes_for_layer(model, layer_idx), **ca_config)
        set_layer_weights(model, layer_idx, weights)
```

### Principle 3: Long-Range Connectivity

**In the brain:** The arcuate fasciculus is a physical highway — a bundle of axons spanning centimeters, connecting production and comprehension regions that are far apart. Without it, you can understand speech (Wernicke's intact) and produce speech (Broca's intact) but can't repeat what you hear (conduction aphasia). The brain *pre-builds* these highways.

**In a transformer:** The residual stream is the highway. Attention heads at early layers need to write information that late layers can read. The skip connections make this possible, but the weight patterns that facilitate long-range information flow must be discovered by training.

**CA approach:** CA rules that produce *band structure* and *smooth gradients* in weight matrices create natural long-range pathways. Reaction-diffusion CAs naturally produce stripe and band patterns (Turing patterns) that function as connectivity highways:

```python
def arcuate_init(shape, band_width=0.1):
    """Reaction-diffusion CA tuned for stripe/band formation.
    Creates natural long-range connectivity channels in weight matrices."""
    # Gray-Scott parameters in the "stripes" regime
    return reaction_diffusion(shape, feed=0.04, kill=0.06, n_steps=300)
```

### Principle 4: Modular Organization (Cortical Columns)

**In the brain:** The cortex is organized into repeating modular units — cortical columns roughly 0.5mm wide, each a mini-processing circuit. Within a column, neurons are densely connected. Between columns, connections are sparser and more selective. This modular structure exists before learning.

**In a transformer:** Attention heads within a layer can be thought of as "columns." FFN layers can be thought of as having sub-networks. But with random init, there's no pre-existing modular organization.

**CA approach:** CAs that produce block-diagonal or periodic structure create natural modularity. Multiple seeds (one per intended "column") grow independently and create modular blocks:

```python
def modular_init(shape, n_modules=4):
    """Multiple CA seeds → independent modules that grow separately.
    Produces block-diagonal-like structure."""
    grid = torch.zeros(shape)
    block_h, block_w = shape[0] // n_modules, shape[1] // n_modules
    for m in range(n_modules):
        # Each module gets its own CA seed
        seed = torch.randn(block_h // 4, block_w // 4) * 0.1
        module_weights = ca_develop_from_seed(seed, (block_h, block_w), n_steps=48)
        grid[m*block_h:(m+1)*block_h, m*block_w:(m+1)*block_w] = module_weights
    # Light cross-module connections (sparse, low magnitude)
    grid += torch.randn(shape) * 0.001
    return grid
```

### Principle 5: Competition and Selection (Lateral Inhibition)

**In the brain:** Neurons in a region compete. When one fires strongly, it suppresses its neighbors through inhibitory interneurons. This lateral inhibition sharpens responses: instead of a blurry activation pattern, you get crisp selectivity. It's a structural circuit, not a learned behavior.

**In a transformer:** Softmax in attention is a form of competition (tokens compete for attention budget). But within the weight matrices themselves, there's no competitive dynamics.

**CA approach:** A live CA rule where strong weights suppress nearby weights during training. This creates emergent sparsity and selectivity, analogous to how lateral inhibition sharpens cortical responses:

```python
def competition_step(W, k=5):
    """Live CA: lateral inhibition. Strongest weights suppress neighbors."""
    local_max = F.max_pool2d(W.abs().unsqueeze(0).unsqueeze(0), k, stride=1,
                              padding=k//2).squeeze()
    is_winner = (W.abs() >= local_max * 0.95).float()
    delta = 0.001 * W * is_winner + (-0.003) * W * (1 - is_winner)
    return delta
```

### Principle 6: Homeostatic Regulation

**In the brain:** Synaptic scaling maintains neural activity within a functional range. If a neuron gets too active, all its synapses are scaled down. If too quiet, they're scaled up. This homeostasis operates independently of Hebbian learning (gradient descent analog). It prevents runaway excitation and silent death.

**In a transformer:** Gradient descent can cause weight explosion/vanishing, dead neurons in FFN, attention heads that collapse to uniform distributions. These are functional pathologies that standard training must fight through.

**CA approach:** A live CA rule that monitors local weight statistics and applies corrections — like an artificial synaptic scaling mechanism running alongside gradient descent:

```python
def homeostatic_step(W, target_std=0.02):
    """Live CA: synaptic scaling. Maintains healthy weight statistics locally."""
    local_mean = neighborhood_mean(W, k=3)
    local_std = neighborhood_std(W, k=3)
    # Pull outliers toward local mean
    mean_correction = -0.1 * (W - local_mean)
    # Push variance toward target
    std_correction = (target_std / (local_std + 1e-8) - 1) * W * 0.01
    return mean_correction + std_correction
```

### Principle 7: Critical Periods and Developmental Timing

**In the brain:** The developmental program doesn't run at constant intensity. There are *critical periods* — windows where specific circuits are highly plastic and being shaped. Phonological processing has a critical period before age 1. Syntactic processing has one before age 5. After the critical period closes, the structure is largely fixed and learning becomes refinement rather than construction.

**In a transformer:** Training has implicit phases — early steps make large changes (high loss, big gradients), later steps refine (low loss, small gradients). But the init is uniform across time.

**CA approach:** The alpha schedule (CA influence over training time) should mimic critical periods. Strong CA influence early (the developmental program is active, building structure), fading as training progresses (learning takes over on the scaffold):

```python
def critical_period_alpha(step, total_steps, alpha_0=0.01):
    """Mimics biological critical periods.
    Strong early influence, then rapid closing."""
    # Critical period closes at 20% of training
    critical_end = total_steps * 0.2
    if step < critical_end:
        # Active development
        return alpha_0 * (1.0 - step / critical_end)
    else:
        # Post-critical: minimal CA, learning dominates
        return alpha_0 * 0.01
```

For different components at different times (mimicking how phonology closes before syntax closes before semantics):

```python
def layerwise_critical_period(step, layer_idx, n_layers, total_steps, alpha_0=0.01):
    """Earlier layers close their critical period first."""
    # Layer 0 closes at 10% of training, last layer at 30%
    close_frac = 0.1 + 0.2 * (layer_idx / (n_layers - 1))
    critical_end = total_steps * close_frac
    if step < critical_end:
        return alpha_0 * (1.0 - step / critical_end)
    return alpha_0 * 0.01
```

---

## CA Variant Toolkit

Each variant is a different "genetic program" that produces different types of functional structure.

### Grid CA

2D grid, neighborhood-based updates, shared MLP rule.

**Produces:** Local clustering, gradients, blob regions, edge patterns.
**Good for:** Principles 1 (specialization via different seeds), 2 (hierarchy via depth-dependent params), 4 (modularity via multi-seed).

### Neural CA (Mordvintsev-style)

Multi-channel hidden state per cell, Sobel perception, stochastic updates. Architecture from Growing NCA (Mordvintsev et al., 2020), applied to weight generation as in HyperNCA (Najarro & Risi, 2022). See Prior Art section for how NeuroGen extends HyperNCA's approach.

**Produces:** Complex self-organizing patterns, hierarchical spatial structure.
**Good for:** Principle 1 (rich specialization), Principle 2 (hierarchical development).

### Reaction-Diffusion

Activator-inhibitor dynamics. Gray-Scott, FitzHugh-Nagumo, Brusselator.

**Produces:** Turing patterns — stripes, spots, mazes, periodic structures.
**Good for:** Principle 3 (band/stripe = long-range connectivity), Principle 4 (regular spots = modular columns).

**Key parameter regimes (Gray-Scott):**
- `feed=0.04, kill=0.06` → stripes (connectivity highways)
- `feed=0.03, kill=0.06` → spots (modular columns)
- `feed=0.025, kill=0.06` → branching patterns (dendritic-like)
- `feed=0.055, kill=0.062` → worms/labyrinthine (complex mixed structure)

### Spectral CA

Operates in frequency domain, generates Fourier coefficients.

**Produces:** Smooth periodic patterns, multi-scale structure.
**Good for:** Principle 2 (multi-scale hierarchy), Principle 3 (smooth long-range patterns).

### Handcrafted Structural Priors (baselines)

Not CA, but direct structured init for comparison:
- Block-diagonal (modular)
- Low-rank + sparse (highway + local)
- Orthogonal (maximal information preservation)
- Identity-like (residual pass-through)

---

## Live CA Rules (operate during training)

These correspond to *ongoing* developmental processes that continue after initial structure is built.

| Rule | Brain Analog | What It Does to Weights |
|------|-------------|------------------------|
| `homeostatic_step` | Synaptic scaling | Maintains healthy local weight statistics |
| `modularity_step` | Column boundary maintenance | Reinforces block structure, decays cross-block |
| `pruning_step` | Synaptic pruning | Eliminates low-utility connections |
| `competition_step` | Lateral inhibition | Winners suppress neighbors → sparsity |
| `learned_step` | General developmental program | Small MLP genome, meta-learnable |

**Alpha schedules (CA influence over time):**
- `exponential_decay` — simple fade
- `critical_period` — strong early, rapid close at 20% of training
- `layerwise_critical_period` — earlier layers close first
- `adaptive` — increase when loss stagnates (developmental rescue), decrease when improving
- `cyclic` — periodic bursts (sleep/wake consolidation analog)

**Per-layer scope:**
- Attention Q/K → competition (head specialization)
- Attention V/O → modularity (information routing)
- FFN → pruning (sparse computation)
- Embeddings → homeostatic only (light touch)

---

## Key Diagnostics

Print these from `train.py` so they appear in the log:

| Metric | What it reveals |
|--------|----------------|
| `val_bpb` | Primary metric |
| `init_loss` | Loss at step 0 — directly measures init quality |
| `ca_delta_norm` | How much is the CA changing weights? |
| `grad_delta_norm` | How much is the gradient changing weights? |
| `ca_grad_alignment` | cos(Δw_ca, Δw_grad) — cooperation vs competition |
| `weight_sparsity` | Fraction of near-zero weights (pruning effect) |
| `head_diversity` | Cosine distance between attention head weight vectors (specialization) |

---

## Benchmarking

The autoresearch loop (program.md) optimizes val_bpb greedily. Benchmarking adds statistical rigor to know *why* something works and whether improvements are real.

### Usage

```bash
uv run benchmark.py --compare "default,xavier,grid_ca,modular_ca" --seeds 5 --minutes 2
```

Produces: `outputs/benchmark_<timestamp>.md` (report) and `outputs/benchmark_<timestamp>.csv` (raw data).

### Diagnostic Metrics

| Metric | Validates | Principle |
|--------|-----------|-----------|
| `init_weight_std` | CA output scale (~0.02 expected) | General health |
| `init_head_diversity` | Cosine distance between Q-projections | P1: Functional Specialization |
| `init_block_diag_ratio` | Energy in block-diagonal vs off-diagonal | P4: Modular Organization |
| `init_layer_similarity` | Cosine sim between adjacent layers (low = good) | P2: Hierarchical Processing |
| `ca_delta_norm` | CA update magnitude | P5-7: Live CA activity |
| `ca_grad_alignment` | cos(Δw_ca, Δw_grad): +1=cooperate, -1=compete | P5-7: CA-gradient interaction |
| `ca_contribution_ratio` | ||Δw_ca|| / (||Δw_ca|| + ||Δw_grad||) | P5-7: CA influence level |
| `weight_sparsity` | Fraction of near-zero weights | P5: Competition/pruning effect |

### Statistical Protocol

- **Minimum 5 seeds** per method (seeds 42-46)
- **Paired t-test** between each method and best baseline
- **p < 0.05** threshold for statistical significance
- Run benchmarks at phase transitions (after Phase 1, after Phase 2) before proceeding
- All runs use the same time budget as autoresearch experiments (default 2 min)

---

## External Comparison

NeuroGen's claim is NOT "CA model beats GPT-2." The claim is "CA init reaches the same val_bpb in fewer steps/FLOPs than standard init at the same model size."

### Level 1: val_bpb vs xavier baseline (always do this)

Every CA result must be reported as "val_bpb X (baseline Y, improvement Z%)." Run `uv run benchmark.py --baseline --seeds 5` to establish the xavier reference. All subsequent `--compare` runs automatically show `vs_baseline_pct`. Positive = CA is better.

### Level 2: DCLM CORE score (if Level 1 is positive)

CORE evaluates across 22 tasks (GPT-2 reference: CORE=0.2565). NeuroGen's small models will score low in absolute terms — the comparison is CA-init vs xavier-init at matched size. See `evaluate_core.py` for manual evaluation path via nanochat.

### Level 3: FLOPs-matched comparison (most rigorous)

CA init costs compute. `benchmark.py` accounts for this: it times CA development separately, gives the baseline those extra seconds of training, and reports `total_flops` including CA overhead. Wall time on M1 Pro is the practical proxy since there's only one device.

### Fair comparison protocol

- Same model size, same data, same time budget
- CA development time is subtracted from training budget (baseline gets those seconds back)
- FLOPs include CA development overhead
- Minimum 5 seeds, paired t-test, p < 0.05

---

## Experiment Priority

1. **Baseline.** Unmodified train.py, establish val_bpb reference.
2. **Handcrafted structured init.** Block-diagonal, orthogonal. Does structure matter at all?
3. **Principle 4 — Modular init.** Multi-seed grid CA → block structure. Does modularity help?
4. **Principle 1 — Specialized heads.** Different CA seeds per head. Does pre-specialization help?
5. **Principle 2 — Hierarchical init.** Depth-dependent CA params. Does hierarchy help?
6. **Principle 3 — Long-range init.** Reaction-diffusion stripes/bands. Does connectivity help?
7. **Principle 6 — Live homeostatic.** CA maintaining weight health during training.
8. **Principle 5 — Live competition.** Lateral inhibition → emergent sparsity.
9. **Principle 7 — Critical periods.** Phased alpha, layerwise timing.
10. **Combined.** Best CA init + best live rule + critical period timing.

---

## The Genome

The CA rule's parameters (MLP weights, reaction-diffusion rates, seed patterns) constitute the "genome" — the developmental program. Key properties:

- **Compression:** The genome should be ~100-1000× smaller than the weights it produces. A 500-parameter genome that generates 10M-parameter weight matrices = biologically-inspired compression.
- **Universality:** Ideally, one genome produces good structure for any layer size (analogous to how one genome builds brains of different sizes across development).
- **Evolvability:** The autoresearch loop *is* evolution. Each experiment that improves val_bpb = one generation of selection. The git history accumulates the "evolutionary trajectory" of the genome.

---

## Prior Art & How NeuroGen Differs

### Directly Related

**HyperNCA: Growing Developmental Networks with Neural Cellular Automata** (Najarro, Risi et al., 2022) — [paper](https://arxiv.org/abs/2204.11674)
The closest existing work. Uses 3D Neural Cellular Automata to grow weight matrices for RL policy networks, optimized with CMA-ES. Demonstrated that CA-grown networks can solve CartPole, LunarLander, and other control tasks, and that "metamorphosis networks" can transform weights to solve task variations.
**NeuroGen differs:** HyperNCA targets small RL policies (~100s of params). NeuroGen targets transformers/LLMs (~millions of params). HyperNCA uses CA at init only. NeuroGen also tests live CA during training. HyperNCA doesn't use neurolinguistic functional principles to design seeds.

**Growing Neural Cellular Automata** (Mordvintsev, Randazzo, Niklasson, Levin, 2020) — [paper](https://distill.pub/2020/growing-ca/)
The foundational Neural CA paper. Trains differentiable CA (~8K params) to grow, maintain, and self-repair visual patterns. Introduced the Sobel perception → MLP update → stochastic mask → residual update architecture that NeuroGen's Variant B (Neural CA) is based on.
**NeuroGen differs:** Growing NCA grows images. NeuroGen grows weight matrices. The CA architecture transfers but the objective and evaluation are completely different.

**Weight Agnostic Neural Networks** (Gaier & Ha, 2019) — [paper](https://arxiv.org/abs/1906.04358)
Searches for network *topologies* (via NEAT) that perform well even with random weights. Demonstrates that architecture alone, without weight training, can encode useful inductive biases.
**NeuroGen differs:** WANNs search over topology (which connections exist) in small custom architectures. NeuroGen searches over weight *values* (what structure the connections have) in a fixed transformer topology. Complementary ideas — WANNs say structure matters, NeuroGen says structured *initialization* matters.

**HyperNetworks** (Ha, Dai & Le, 2017) — [paper](https://arxiv.org/abs/1609.09106)
A small network generates weights for a larger main network. Inspired by HyperNEAT where CPPNs are evolved to define weight structure. Applied to RNNs/LSTMs for sequence modeling and machine translation.
**NeuroGen differs:** HyperNetworks use a feedforward generator. NeuroGen uses iterative CA dynamics (local rules, many steps), which naturally produce spatially-structured patterns (modularity, bands, gradients) that feedforward generators don't.

### Related Initialization & Growth

**LiGO: Learning to Grow Pretrained Models** (Wang et al., 2023, ICLR) — [paper](https://arxiv.org/abs/2303.00980)
Learns linear operators to map smaller pretrained transformer weights into larger transformer initialization. Accelerates training of scaled-up models.
**NeuroGen differs:** LiGO transfers from a trained small model. NeuroGen generates structure from scratch (no pretrained model needed). Different use case but same insight: smart initialization beats random.

**Principled Weight Initialization for Hypernetworks** (Chang et al., 2023) — [paper](https://arxiv.org/abs/2312.08399)
Shows that standard init (Xavier, Kaiming) applied to hypernetworks fails to produce mainnet weights at correct scale. Develops principled scaling techniques.
**Relevant to NeuroGen:** CA-generated weights will face identical scale problems. The agent should normalize CA output to match healthy weight statistics (std ≈ 0.02 for transformers). This paper's insights should be applied.

**Developmental Graph Cellular Automata** (Waldegrave, Stepney & Trefzer, 2023, ALIFE) — [paper](https://doi.org/10.1162/isal_a_00658)
Extends CA to grow graph structures for recurrent neural networks and reservoir computing. Characterizes five classes of CA growth behavior on graphs.
**NeuroGen differs:** Grows weight values on a fixed graph (transformer), not the graph topology itself.

### Conceptual Foundations

**Fast Weight Programmers & Linear Transformers** (Irie & Gershman, 2025) — [paper](https://arxiv.org/abs/2508.08435)
Comprehensive review connecting transformers, RNNs, and biological synaptic plasticity. Shows that many modern sequence models can be viewed as networks whose weights change dynamically — "fast weights" programmed by a "slow" network. Discusses biological parallels including homeostatic decay and neuromodulation.
**Relevant to NeuroGen:** NeuroGen's live CA is a form of fast weight programming — a local rule that modifies weights during training, alongside gradient descent. This paper provides theoretical grounding.

**The Neuroscience of Transformers** (2026) — [paper](https://arxiv.org/abs/2603.15339)
Recent survey comparing biological neural networks and transformers. Notes that the brain uses dense recurrent connections, lateral inhibition, multiple neuromodulators, and learning across multiple timescales — all features absent from standard transformer training but present in NeuroGen's design.

### What NeuroGen Uniquely Combines

No existing project combines all of these:
1. CA developmental programs → transformer weight matrices (not RL policies)
2. Live CA operating during training alongside gradient descent (not init-only)
3. Neurolinguistic functional principles (specialization, hierarchy, modularity, competition, homeostasis, critical periods) driving the CA rule design
4. Autoresearch loop (Karpathy pattern) as the evolutionary outer loop for CA genomes
5. Apple Silicon / single-GPU accessibility (not requiring H100 clusters)

---

## References

```
Mordvintsev, A., Randazzo, E., Niklasson, E., & Levin, M. (2020).
    Growing Neural Cellular Automata. Distill.
    https://distill.pub/2020/growing-ca/

Najarro, E., Sudhakaran, S., Glanois, C., & Risi, S. (2022).
    HyperNCA: Growing Developmental Networks with Neural Cellular Automata.
    ICLR 2023. https://arxiv.org/abs/2204.11674

Gaier, A. & Ha, D. (2019).
    Weight Agnostic Neural Networks. NeurIPS 2019.
    https://arxiv.org/abs/1906.04358

Ha, D., Dai, A., & Le, Q. (2017).
    HyperNetworks. ICLR 2017.
    https://arxiv.org/abs/1609.09106

Wang, P., Panda, R., et al. (2023).
    Learning to Grow Pretrained Models for Efficient Transformer Training.
    ICLR 2023. https://arxiv.org/abs/2303.00980

Chang, O., Flokas, L., & Lipson, H. (2023).
    Principled Weight Initialization for Hypernetworks.
    https://arxiv.org/abs/2312.08399

Waldegrave, R., Stepney, S., & Trefzer, M. (2023).
    Developmental Graph Cellular Automata. ALIFE 2023.
    https://doi.org/10.1162/isal_a_00658

Irie, K. & Gershman, S. (2025).
    Fast Weight Programming and Linear Transformers: From ML to Neurobiology.
    https://arxiv.org/abs/2508.08435

Stanley, K. & Miikkulainen, R. (2002).
    Evolving Neural Networks through Augmenting Topologies. Evolutionary Computation.

Turing, A. (1952).
    The Chemical Basis of Morphogenesis. Phil. Trans. R. Soc. B.

Karpathy, A. (2025-2026).
    nanochat / autoresearch. https://github.com/karpathy/nanochat
```
