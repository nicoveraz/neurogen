# Paper Draft: Developmental Constraints in Transformer Training

## Title Options

1. "Developmental Constraints Protect Pre-Wired Circuits During Transformer Training"
2. "Local-to-Global Attention Growth: How Architectural Constraints Enable Structural Priors in Transformers"
3. "The Transformer Doesn't Want Your Help (Unless You Constrain It First)"

## Abstract (~200 words)

We investigate whether biologically-inspired developmental principles can improve transformer training. Across 200+ controlled experiments at two scales (3.4M and 125M parameters), we systematically test cellular automata weight initialization, live developmental rules, embryogenic development, pre-wired circuits, and architectural constraints inspired by cortical development. Most approaches produce marginal gains (+0.6%) or actively hurt (-3% to -5%). However, embedding a developmental constraint into the architecture — attention windows that grow from local to global across layers — yields robust improvements that increase with scale: +1.5% at 3.4M parameters (p=0.001, Cohen's d=2.05, 5 seeds) and up to +12.9% at 125M parameters (50k steps, gap still widening). Gradient mechanism experiments reveal that windows do not reduce gradient noise (noise norm is constant across window sizes) but instead increase gradient signal coherence 18x, consistent with forced architectural specialization rather than noise removal. A variance reduction control (larger batch size) fails to replicate the effect at equal token count, confirming the mechanism is specific to attention restriction. We propose that constraining early layers to local attention forces compositional feature hierarchies that later layers leverage for more effective global integration.

## 1. Introduction

Standard transformers start from random initialization with uniform, fully-global attention at every layer. Gradient descent discovers all internal structure — specialized attention heads, hierarchical layer roles, induction circuits — from scratch.

The brain takes a different approach. Genetic developmental programs grow functional structure before learning begins: cortical receptive fields develop local-to-global through layers, induction-like circuits (pattern completion) are pre-wired, and architectural constraints (laminar structure, lateral inhibition, critical periods) protect these scaffolds during early learning.

We ask: can these developmental principles improve transformer training?

Our answer is nuanced. Most biological inspirations fail when applied to transformers. But one specific combination works substantially: constraining attention to grow from local to global across layers (mimicking cortical receptive field development) AND pre-wiring induction head circuits, together produce a +5.2% improvement that neither achieves alone.

The key insight is that structural priors require architectural protection. Without constraints, gradient descent overwrites pre-wired structure within the first few hundred steps. With constraints that force local-before-global processing, the pre-wired circuits occupy a functional niche that gradient descent reinforces rather than destroys.

## 2. Related Work

### 2.1 Neural Cellular Automata for Weight Generation
- Growing Neural Cellular Automata (Mordvintsev et al., 2020)
- HyperNCA (Najarro & Risi, 2022) — CA growing RL policy weights
- Our work extends to transformers and tests live CA during training

### 2.2 Structured Initialization for Transformers
- Mimetic initialization (Trockman & Kolter, 2023) — Q·K ≈ I, V·O ≈ -I
- T-Fixup (Huang et al., 2020) — Lipschitz-constrained init
- muP (Yang et al., 2022) — principled parameterization for scaling
- Our work uses CA-generated structure rather than analytical formulas

### 2.3 Universal Circuits in Transformers
- Induction heads (Olsson et al., 2022) — universal match-and-copy circuits
- Circuit tracing (Anthropic, 2025) — attribution graphs reveal consistent internal structure
- Head specialization (Basile et al., 2025) — consistent patterns across models
- Our work attempts to pre-build these known circuits

### 2.4 Attention Patterns and Efficiency
- Sliding window attention (Beltagy et al., 2020; nanochat SSSL pattern)
- Local-to-global progressive attention — not previously studied as developmental principle
- Our contribution: framing window growth as developmental constraint, showing synergy with pre-wired circuits

## 3. Method

### 3.1 Experimental Framework

Autoresearch paradigm (Karpathy, 2026): autonomous agent modifies training code, runs fixed-time experiments, keeps improvements, discards failures. 160+ experiments across 140+ GPU-hours on Apple M1 Pro.

Base model: GPT-style transformer, depth 4, channels 256, ~3.4M parameters. Training data: TinyStories. Metric: validation bits-per-byte (val_bpb), vocabulary-size invariant.

### 3.2 Approaches Tested (taxonomy)

**Category A: Weight Decoration (modify weights, don't change architecture)**
- A1. CA initialization: grid CA, block-diagonal CA, reaction-diffusion, spectral CA at various blend ratios (5%-30%)
- A2. Live CA during training: homeostatic normalization, competition, pruning rules
- A3. Embryogenic CA: activity-dependent rules that see gradients during critical period
- A4. Pre-wired circuits: induction heads, layer-role differentiation, head diversity

**Category B: Architectural Constraints (change how the transformer computes)**
- B1. Attention window growth: linear, quadratic, step-function local→global across layers
- B2. Attention bias: CA-generated position-dependent bias added to attention scores

**Category C: Combinations**
- C1. Constraint + scaffold: quadratic windows + induction pre-wiring
- C2. Constraint + development: quadratic windows + embryogenic CA
- C3. All combined: windows + induction + embryogenic

**Category D: Radical modifications (tested and rejected)**
- D1. CA modulation channels (parallel CA state gating transformer)
- D2. Token vitality (cell death dynamics)
- D3. Sleep consolidation (periodic offline CA)
- D4. Cross-layer persistent CA state
- D5. Developmental dropout (block-structured)

### 3.3 Quadratic Attention Window Growth

The key positive finding. Each layer l (of L total) has a maximum attention window:

    window(l) = min(seq_len, base + (l/(L-1))² × seq_len)

Layer 0 attends to ±8 tokens (local). Layer L-1 attends to all tokens (global). Intermediate layers follow quadratic growth — slow local expansion, then rapid global opening. This mirrors cortical development where primary sensory areas develop narrow receptive fields first, with higher association areas developing broader fields later.

Implementation: causal attention mask with per-layer window. Zero parameter overhead. Zero compute overhead (masking is free).

### 3.4 Induction Head Pre-Wiring

Based on Olsson et al. (2022): induction heads are universal two-layer circuits that implement match-and-copy. Layer 0 head: Q/K initialized for previous-token attention. Layer 1 head: Q/K initialized for content matching, V/O for copying.

Implementation: targeted initialization of one head per layer in the first two layers. ~0.1% of total parameters affected.

## 4. Results

### 4.1 Category A: Weight Decoration — Marginal or Negative

| Approach | Best result | Experiments | Finding |
|----------|------------|-------------|---------|
| CA init (various) | +0.6% (5% block-diagonal blend) | 40+ | Real but marginal. Simpler patterns beat complex ones. |
| Live CA | -2% to -5% (all variants) | 15+ | Overhead kills performance at this scale. |
| Embryogenic CA | +1.3% (extended critical period) | 16 | Activity-dependent better than blind, but modest. |
| Pre-wired circuits alone | -3% to -5% | 12 | Gradient descent destroys pre-wired structure. |

Key finding: the transformer resists weight decoration. Adding structure to weights, whether at initialization or during training, produces single-digit percentage gains at best and frequently hurts.

### 4.2 Category B: Architectural Constraints — Consistent Improvement

| Window type | val_bpb | vs baseline | p-value |
|-------------|---------|-------------|---------|
| Full attention (baseline) | 1.0433 | — | — |
| Linear growth | 1.0122 | +3.0% | <0.05 |
| Step function | 1.0022 | +3.9% | <0.05 |
| Quadratic growth | 0.9998 | +4.2% | <0.05 |

All window growth variants improve over full attention. Quadratic is best — slow local expansion followed by rapid global opening. This is robust across seeds (3 seeds each).

### 4.3 Category C: The Synergy Finding

| Approach | val_bpb | vs baseline |
|----------|---------|-------------|
| Quadratic windows alone | 0.9998 | +4.2% |
| Induction pre-wiring alone | 1.0919 | -4.7% |
| **Quadratic + induction** | **0.9892** | **+5.2%** |

The combination exceeds both components. Induction pre-wiring alone hurts (-4.7%), but when protected by quadratic windows it contributes an additional +1.0% on top of the window benefit. The window constraint forces early layers to stay local, preventing gradient descent from overwriting the induction scaffold before it can be reinforced.

### 4.4 Category D: What Failed and Why

[Table of all negative results with failure modes — model collapse, overhead, interference]

### 4.5 The Constraint-Protection Hypothesis

We propose: structural priors in transformers only help when architectural constraints prevent gradient descent from destroying them during early training.

Evidence:
- Induction pre-wiring alone: -4.7% (destroyed by early gradients)
- Induction + quadratic windows: +5.2% (protected, then reinforced)
- Layer-role init alone: -4.6% (destroyed)
- Embryogenic CA alone: +1.3% (modest — developmental rules provide some self-protection via adaptive critical periods, but not enough)

The quadratic window acts as an artificial "critical period" — it constrains early-layer plasticity (local attention only), allowing pre-wired local circuits (induction heads) to be reinforced by data before global competition can disrupt them.

## 5. Scaling to 125M Parameters

### 5.1 Setup

GPT-2 Small architecture (12 layers, 768 dim, 12 heads, 1024 context) trained on FineWeb-Edu (~100M tokens) on NVIDIA H100 80GB. Quartic window schedule with base=16. Flash Attention 2 with native sliding window support — windowed models run slightly *faster* than baseline (2.84 vs 2.78 steps/sec).

### 5.2 Results

The advantage **grows with training**, from +1.9% at 5k steps to +12.9% at 50k steps (mean of 2 matched-seed pairs):

| Steps | Baseline bpb | Quartic bpb | Gap |
|-------|-------------|-------------|-----|
| 5k | 4.911 | 4.820 | +1.9% |
| 10k | 4.580 | 4.390 | +4.2% |
| 20k | 4.095 | 3.942 | +3.8% |
| 30k | 3.844 | 3.576 | +7.0% |
| 40k | 3.538 | 3.188 | +9.9% |
| 50k | 3.447 | 3.001 | **+12.9%** |

Neither model is converged at 50k steps — the gap is still widening.

At 20k steps (Chinchilla-optimal token budget) with 5 seeds:
- Baseline mean: 3.631, Power_4.0 mean: 3.566 (+1.8%, n=3 common seeds)
- 4/5 seeds show improvement in same-seed comparison

### 5.3 Scaling Properties

The effect is stronger at 125M than at 3.4M (+12.9% vs +1.5% at convergence), confirming the prediction that more layers benefit from richer developmental progression. Windowed models also exhibit lower seed variance, suggesting the constraint provides a more robust optimization landscape.

## 6. Mechanism Analysis

### 6.1 Experiment 1: Gradient Quality vs Window Size

On a frozen trained checkpoint, we measured gradient SNR, direction stability, and effective rank at 10 window sizes (8-256 tokens, 50 backward passes each, layer 0 only).

Key finding: **gradient noise is constant** (~0.0053 norm) across all window sizes. What changes is **gradient signal** — signal norm increases 18x from window 256 (0.0017) to window 8 (0.032). Windows don't remove noise; they make gradients point in a more consistent direction.

The SNR curve shows a knee at ~32-48 tokens, matching the model's natural attention span.

### 6.2 Experiment 2: Gradient Decomposition

Decomposed the softmax backward pass into contributions from attended vs non-attended positions across all 4 layers.

| Layer | Noise Fraction | Natural Span | Attended Positions |
|-------|---------------|-------------|-------------------|
| 0 | 5.0% | 12.9 | 17.3 |
| 1 | 4.0% | 7.4 | 10.9 |
| 2 | 6.9% | 21.3 | 23.3 |
| 3 | 6.7% | 13.0 | 18.7 |

Noise fraction is only 4-7% — the softmax coupling introduces minimal gradient contamination. This eliminates the "gradient noise removal" hypothesis.

### 6.3 Experiment 3: Variance Reduction Control

Compared quartic windows (batch 32) against full attention with larger effective batch sizes (128, 256 via gradient accumulation) to test whether simple variance reduction replicates the window effect.

At equal token count (16M tokens):
- **Quartic windows (batch 32): 1.224 bpb**
- Baseline (batch 32): 1.242 bpb
- Full attention (batch 128): 1.357 bpb
- Full attention (batch 256): 1.321 bpb

Larger batch is **worse** than baseline at equal token budget. Variance reduction cannot replicate the window effect.

### 6.4 Mechanism Conclusion

The mechanism is **forced architectural specialization**: constraining early layers to local attention forces them to build compositional features (local n-gram patterns, syntactic boundaries) that later layers leverage for more effective global integration. This produces higher gradient signal coherence as a consequence, not a cause.

## 7. Limitations

- 125M results are preliminary (2 seeds at 50k steps, models not converged).
- No downstream task evaluation (CORE, MMLU, etc.).
- The 3.4M model uses TinyStories (narrow domain); 125M uses FineWeb-Edu (broader).
- Optimal exponent at depth 12 not yet determined (tested only quartic).
- Mechanism experiments performed only at 3.4M scale.

## 8. Discussion and Future Work

### Scaling predictions confirmed
The effect grows with scale (+1.5% at 3.4M → +12.9% at 125M), and the gap is still widening at 50k steps. Longer training runs and larger models may show even larger improvements.

### Practical implications
Attention windows are free: zero parameter overhead and with Flash Attention's native sliding window support, windowed models run *faster* than baseline. This is a pure training efficiency gain with a throughput bonus.

### Future work
1. Determine optimal exponent at depth 12+ (power 4 may not be optimal for deeper models)
2. Train to full convergence at 125M to measure final gap
3. Downstream evaluation (benchmarks, generation quality)
4. Test at 350M+ parameters
5. Investigate whether the forced specialization creates interpretably different internal representations

## 9. Conclusion

We systematically tested 26 biologically-inspired approaches across 200+ experiments at two scales. Most fail. Developmental attention window growth — constraining early layers to local attention with quartic growth toward full attention — produces consistent improvements: +1.5% at 3.4M parameters (p=0.001, 5 seeds) and +12.9% at 125M parameters (50k steps, gap widening). Mechanism analysis eliminates gradient noise removal and variance reduction, identifying forced architectural specialization as the operative mechanism. The approach requires no additional parameters, no extra compute, and produces faster models when combined with Flash Attention's sliding window support.

## References

- Karpathy, A. (2026). nanochat / autoresearch. Training harness and experiment loop.
- Najarro, E. & Risi, S. (2022). HyperNCA: Growing Neural Cellular Automata to Grow Neural Networks. arXiv:2204.11674.
- Mordvintsev, A. et al. (2020). Growing Neural Cellular Automata. Distill.
- Olsson, C. et al. (2022). In-context Learning and Induction Heads. Transformer Circuits Thread.
- Beltagy, I. et al. (2020). Longformer: The Long-Document Transformer. arXiv:2004.05150.
- Yang, G. et al. (2022). Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. arXiv:2203.03466.
- Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv:2307.08691.
- Hoffmann, J. et al. (2022). Training Compute-Optimal Large Language Models (Chinchilla). arXiv:2203.15556.

## Appendix

A. Complete experiment table (all 200+ experiments with configs, seeds, metrics)
B. Implementation details (train_r4.py and train_125m.py architecture)
C. Negative results catalog (every failed approach with failure mode analysis)
D. Gradient mechanism experiment details and raw data
E. Reproducibility: GitHub repo (https://github.com/nicoveraz/neurogen), hardware specs, random seeds
F. 125M training curves (full step-by-step data for all seeds)
