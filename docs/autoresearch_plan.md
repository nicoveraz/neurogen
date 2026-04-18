# NeuroGen Autoresearch Plan: Trajectory Geometry and Topographic Representations

**Branch:** `autoresearch/trajectory-topography`
**Base model:** NeuroGen 3.4M (existing config)
**Hardware:** M1 Pro (MLX preferred over PyTorch/MPS)
**Storage budget:** ~50GB active, external HDD for archive
**Estimated wall-clock:** 2–3 weeks with iteration

---

## Motivation

Current transformer LMs handle long-tail compositional cases by approximating composition through data coverage, which scales exponentially with the combinatorial depth of the domain. This plan tests whether two structural interventions — (1) topographic organization of token embeddings and (2) trajectory-conditioned prediction — buy measurable compositional generalization at small scale, and whether trajectory geometry itself encodes information about how representations form.

The plan is structured so each experiment either confirms a specific hypothesis or kills it, and later experiments only run if earlier ones show signal. Partial completion still yields reusable artifacts: a characterized baseline, a topographic regularizer implementation, a compositional evaluation set, and trajectory analysis tooling.

---

## Core hypotheses

1. **Substrate hypothesis:** Token embedding trajectories in a 3.4M model on this corpus contain readable structure (convergence, divergence, shared paths, path shapes), not noise.
2. **Topography hypothesis:** A 2D topographic regularizer on token embeddings, driven by co-occurrence, improves compositional generalization relative to baseline and to a structure-free regularization control.
3. **Trajectory-geometry hypothesis:** Trajectory-geometric features (convergence rate, divergence point, path similarity, path shape) correlate with per-token or per-concept compositional generalization performance.
4. **Trajectory-conditioning hypothesis:** Conditioning next-token prediction on a trajectory summary (FiLM-style, gradient-detached, slow-schedule) improves compositional generalization beyond topography alone.

---

## Experiment sequence

### Experiment 0 — Substrate check

**Goal:** Confirm pipeline works end-to-end with the checkpoint cadence downstream experiments require.

**What runs:**
- NeuroGen baseline config on MLX, unchanged architecture.
- Checkpoint cadence: every 500 steps for first 10K steps, then every 1000 steps.
- Save both full checkpoints (weights + optimizer state, for resumability) and lightweight snapshots (token embedding matrix only, for trajectory analysis).

**Outputs:**
- `runs/exp0_baseline/` with checkpoints, loss curves, final perplexity.
- Confirmed disk-usage numbers per checkpoint (validates storage plan).

**Stop condition:** If MLX pipeline has issues, fix before proceeding. Don't build on broken infrastructure.

**Time:** 1 day.

---

### Experiment 1 — Trajectory visibility

**Goal:** Characterize what trajectories look like at this scale, before any architectural changes. Everything downstream assumes trajectories contain readable structure; verify.

**What runs:** Analysis only, on Experiment 0 checkpoints. No new training.

**Analyses:**
- Per-token embedding trajectory through training (PCA to 2D for visualization).
- Pairwise CKA between consecutive checkpoints (trajectory smoothness).
- Per-token movement magnitude over time (when does each token stop moving).
- Early clustering vs late clustering (which tokens group together at different training stages).
- Qualitative inspection: do frequent tokens stabilize first? Do semantically-related tokens converge?

**Outputs:**
- `analysis/exp1_trajectories/` with notebook, plots, summary statistics.
- A written paragraph characterizing the trajectory regime at 3.4M on this corpus.

**Decision gate:** If trajectories look like noise (no visible structure, no token clusters, no stabilization patterns), the plan's premises need reconsideration before continuing. Log this as a finding and stop.

**Time:** 1 day.

---

### Experiment 2 — Topographic regularizer (minimum viable)

**Goal:** Test whether co-occurrence-driven topographic organization of token embeddings improves compositional generalization.

**Architecture change:**
- Add 2D grid parameter (16×16 initially, adjust based on vocab size).
- Topographic loss term: for each token pair (i, j) with co-occurrence above threshold, penalize `||grid_pos(i) - grid_pos(j)||` weighted by co-occurrence strength.
- Co-occurrence window: 5 tokens, measured on training corpus before training starts (static).

**Curriculum:**
- Neighborhood width: anneal from 8 to 1 over first 20% of training (SOM-style).
- Regularizer weight: ramp from 0 to target over first 10% of training (avoid fighting LM loss at initialization).

**Three conditions (ablation):**
1. **Baseline:** NeuroGen unchanged (same as Experiment 0, can reuse).
2. **Topographic:** Co-occurrence-driven topographic loss as specified.
3. **Random-permutation control:** Same topographic loss but with randomly permuted co-occurrence targets. Isolates structural signal from generic regularization.

**Diagnostics during training:**
- Both loss terms tracked separately (LM loss, topographic loss).
- Gradient norms per loss term.
- Cosine similarity between the two gradients (detect fighting).
- Grid layout snapshots every 5K steps.

**Outputs:**
- `runs/exp2_baseline/`, `runs/exp2_topographic/`, `runs/exp2_control/` with full checkpoints and snapshots.
- Grid visualizations over time for topographic and control conditions.

**Decision gate:** If topographic does not beat baseline on compositional eval (Experiment 3), and/or if random-permutation control is indistinguishable from topographic, the structural signal isn't doing what's hypothesized at this scale. Trajectory-conditioning (Experiment 5) probably won't help either; stop or rethink.

**Time:** 3–4 days including setup and debugging.

---

### Experiment 3 — Compositional evaluation construction

**Goal:** Build a compositional generalization evaluation matched to this corpus. Without it, the other experiments only produce perplexity numbers, which don't distinguish composition from coverage.

**Runs in parallel with Experiment 2** (or immediately after).

**Design:**
- Identify feature axes in the training corpus that compose (e.g., if the corpus has gender-marked and tense-marked verbs: gender × tense × subject).
- Enumerate combinations that appear in training (supervised set).
- Construct held-out combinations: specific (gender, tense, subject) triples where each component appears separately in training but the combination does not.
- Evaluation metric: likelihood ratio on held-out combinations vs matched controls, or rank of the correct completion.

**Size:** A few hundred items, hand-verified. Better small and clean than large and noisy.

**Outputs:**
- `eval/compositional_v1.jsonl` with items and expected answers.
- `eval/compositional_eval.py` runnable against any checkpoint.

**Time:** 2 days.

---

### Experiment 4 — Trajectory geometry measurement

**Goal:** Test whether trajectory-geometric features correlate with compositional generalization.

**What runs:** Analysis only, on Experiment 2 runs.

**Features computed per token:**
- **Convergence:** For each token pair, when (if ever) do their trajectories enter a shared region. Rate and timing.
- **Divergence:** Points in training where previously-similar tokens move apart. Direction of divergence.
- **Shared paths:** Trajectory similarity between tokens, independent of endpoints (DTW on PCA projections, or similar).
- **Path shape:** Straightness (ratio of endpoint distance to path length), curvature, presence of reversals, presence of sharp kinks (phase transitions).

**Correlation analysis:**
- Per-token compositional generalization performance from Experiment 3.
- Correlate with each trajectory-geometric feature.
- Report which features correlate and with what sign.

**Outputs:**
- `analysis/exp4_geometry/` with feature computations, correlation tables, qualitative examples.
- Identification of tokens with high-compositional vs low-compositional performance and their trajectory signatures.

**Decision gate:** If no trajectory-geometric feature correlates meaningfully with compositional performance, the trajectory-conditioning hypothesis loses its main motivation. Experiment 5 becomes speculative rather than grounded; decide whether to proceed or stop.

**Time:** 2 days.

---

### Experiment 5 — Trajectory-conditioned prediction

**Goal:** Test whether conditioning next-token prediction on a trajectory summary improves performance beyond topography.

**Only run if Experiments 2 and 4 show positive signals.**

**Architecture change:**
- For each token, compute trajectory summary from last N checkpoints (N = 5 initially).
- Trajectory summary: concatenation of (position at each of last N checkpoints, velocity, path curvature measure). Small vector, ~20-50 dimensions.
- FiLM-style modulation: `logits = W · x · g(trajectory_summary(x))` where g is a small learned MLP.
- Gradient from main loss does NOT flow through trajectory summary (detach).
- Trajectory summaries updated on slower schedule than main weights (every 500 steps, not every step).

**Comparison:**
- Baseline (Experiment 0 / Experiment 2 baseline).
- Topographic (Experiment 2 topographic).
- Topographic + trajectory-conditioned.

**Inference-time handling:**
- Decision needed: freeze final trajectory summaries at end of training and use at inference (persistent architectural commitment), OR use only during training (pure regularizer).
- Recommendation: start with training-only to minimize architectural commitment; escalate if results warrant.

**Outputs:**
- `runs/exp5_topographic_trajcond/` with checkpoints.
- Comparison table across all five conditions (baseline, topographic, control, trajectory-conditioned with and without topography if both can be run).

**Time:** 3 days.

---

## Infrastructure notes

### MLX vs PyTorch+MPS

Start with MLX — better memory behavior and performance on M1 Pro. PyTorch+MPS as fallback if MLX lacks specific primitives needed for the topographic regularizer.

### Checkpoint strategy

Two tiers:
- **Full checkpoints** (weights + optimizer + RNG + metadata): every 2K steps, keep only most recent 3. For resumability.
- **Trajectory snapshots** (token embedding matrix only): every 500 steps, keep all. For analysis. ~8-10MB each.

### Storage layout

```
/runs/
  exp0_baseline/
    full_ckpts/       # rotated, 3 most recent
    snapshots/        # kept all, token embeddings only
    logs/
    config.yaml
  exp2_topographic/
    ...
/analysis/
  exp1_trajectories/
  exp4_geometry/
/eval/
  compositional_v1.jsonl
  compositional_eval.py
/archive/               # move completed runs to external HDD
```

### Move policy

Active run on internal SSD. Completed runs archived to external HDD once analysis is extracted. Internal SSD keeps analysis artifacts (derived, small) long-term.

### Diagnostics to log every run

- LM loss per step.
- Auxiliary loss(es) per step, if present.
- Gradient norm per loss term.
- Cosine similarity between gradient of LM loss and gradient of auxiliary loss.
- Learning rate, regularizer weight, neighborhood width (for curriculum tracking).
- Compositional eval score at each full checkpoint.

---

## Decision gates (summary)

| After | If... | Then... |
|---|---|---|
| Exp 0 | pipeline broken | fix before continuing |
| Exp 1 | trajectories look like noise | reconsider premises, log finding, stop |
| Exp 2 | topographic doesn't beat baseline, or control matches topographic | structural signal not present at this scale; stop or rethink |
| Exp 4 | no geometric features correlate with compositional performance | trajectory-conditioning loses motivation; decide whether to run Exp 5 anyway as exploration |
| Exp 5 | no improvement over topographic alone | observation-modulation hypothesis not supported at this scale |

---

## What counts as success

Success is **learning something clean**, not confirming every hypothesis. Any of the following is a useful outcome:

- Topographic constraint helps → direction worth scaling.
- Topographic constraint doesn't help but trajectory geometry correlates with composition → interesting analytical finding, different paper.
- Neither helps → negative result at this scale, informs whether to try at larger scale or abandon.
- Infrastructure and tooling built → reusable for future experiments regardless of this plan's outcome.

The plan is designed so partial completion is always informative and the artifacts (baseline characterization, topographic implementation, compositional eval, trajectory analysis) are reusable for unrelated future work.

---

## Out of scope

- Scaling to larger models. Premature until small-scale signals exist.
- Embodied/affect signal integration. Belongs to a different (robotics) line of work, not NeuroGen.
- ArXiv/publication planning. Decide after Experiment 4 results.
- Cross-architecture comparison (vs slot attention, vs disentanglement methods). Too expensive at this scale; prioritize getting one architecture working cleanly first.

---

## Next action

Create branch `autoresearch/trajectory-topography`, commit this file, start Experiment 0.
