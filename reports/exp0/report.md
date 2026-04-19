# Exp 0 — Substrate check

**Date:** 2026-04-18
**Branch:** `autoresearch/trajectory-topography`
**Training commit:** `a3cc8ae`
**Run dir:** `runs/exp0_baseline/`
**Seed:** 42
**Wall-clock:** 21,122 s (5h 52m) on M1 Pro / MPS
**Final val_bpb:** 0.7943 (best 0.7913 @ step 92K)
**Params:** 3,408,002 (3.41M)
**Framework:** PyTorch + MPS (deviation from plan's MLX preference; logged below)

## Status

Pipeline works end-to-end with the checkpoint cadence Exp 1–5 require. **Proceed to Exp 1.**

---

## 1. Pipeline integrity

| check | result |
|---|---|
| 100,000 optimizer steps completed | ✅ |
| 111 trajectory snapshots (20 dense @ step%500 ≤ 10K, 90 sparse @ step%1000 > 10K, plus step-0) | ✅ cadence exact |
| 3 rotated full checkpoints kept (96K, 98K, 100K) | ✅ |
| All snapshots and ckpts finite, no NaNs | ✅ |
| Full ckpt round-trips (model + optimizer + RNG state + step + config) | ✅ |
| Disk usage | 146 MB total |
| `wte` std: init 0.800 → final 0.0441 | healthy shrinkage, no collapse |

**Takeaway:** infrastructure is safe to build on. Downstream experiments can reuse this cadence and layout.

---

## 2. Baseline reproduces prior NeuroGen long-horizon result

Compared against the prior `train_r4.py` baseline ("default nanochat init") reported in `README.md`:

| step | prior baseline | Exp 0 | delta |
|---|---|---|---|
| 20K | 0.9002 (5-seed mean, σ≈0.0075) | 0.9624 | +0.062 (worse) |
| ~96K | 0.7980 (baseline best @ 96K) | 0.7943 | −0.004 (within noise) |
| 100K final | — | 0.7943 | — |

**At long horizon (≥90K steps), Exp 0 reproduces the prior NeuroGen baseline within noise.** This match is the implementation-correctness sanity check for the whole plan: every downstream comparison in Exp 2 and Exp 5 depends on the baseline being the same baseline we've been using. It is. Do not re-question this.

### The 20K-step gap is a schedule artifact, not a regression

Prior baseline runs used `train_r4.py --minutes 40` (for 20K-style comparisons), which cosined the LR over ≈20K steps — i.e., LR was decayed to `min_lr` by step 20K. This run uses `max_steps=100_000` in the cosine schedule, so at step 20K the LR is still ≈1.67e-3 (far from `min_lr`). Different schedule, different intermediate-step loss. Same architecture, same final behavior.

**Consequence for the notebook:** direct comparisons to prior NeuroGen baseline val_bpb numbers from the README are only valid at the *end* of training. Intermediate-step numbers from this branch are **not** apples-to-apples with the README numbers. For Exp 2 and Exp 5 ablations this does not matter because those experiments will compare conditions trained under the same schedule. But if anyone later tries to cross-compare a 20K-step run on this branch with a 20K-step run from a prior branch, they will be confused. Flagging here so that confusion does not happen.

---

## 3. Substrate signal: displacement statistics

Per-token displacement over training: `d_i = w_final[i] − w_init[i]`, taken across all V = 256 byte-tokens.

### Magnitude

| stat | value |
|---|---|
| min ‖d_i‖ | 10.04 |
| median | 12.65 |
| mean | 12.61 |
| max | 14.25 |
| std | **0.61** |
| range | 4.21 |

**All tokens moved substantially. No dead embeddings. Magnitude is remarkably uniform** — std/mean ≈ 4.8%. This rules out both failure modes:

- *Dead embeddings* (a subset of tokens that didn't train) — not present.
- *Scale domination* (a few high-movement tokens that would swamp downstream analyses) — not present.

### Direction: displacement PCA (see `displacement_pca.py`, `displacement_pca.png`)

Centered the displacement matrix (removes any global translation) before SVD.

| quantity | value | interpretation |
|---|---|---|
| translation fraction ‖mean(d)‖ / ‖d‖ | 0.004 | trivially small — displacement is token-specific, not a bulk drift |
| top PC explained variance | 1.53% | **flat spectrum** |
| cumulative variance @ k=5 | 7.26% | |
| cumulative variance @ k=10 | 13.82% | |
| cumulative variance @ k=20 | 25.63% | |
| cumulative variance @ k=50 | 52.74% | need ~50 of 256 PCs for half the variance |
| participation ratio | **127.6 / 256** | effective rank is ~half the embedding dim |

### What this means for Exp 1

The substrate hypothesis (trajectories contain readable structure) is **likely true** — tokens moved by nearly identical magnitudes in different directions, which is exactly the regime where trajectory analysis is informative. But the structure is **high-dimensional, not low-rank**. Practical consequences:

- **2D PCA projections will be fuzzy.** The plan's default "PCA to 2D for visualization" is still worth running, but do not expect clean geometric structure in the projection. Expect blobs that correlate loosely with token frequency / class, not crisp clusters.
- **Pairwise metrics (CKA, cosine similarity, DTW on full-D paths) are the right primary tools.** These don't need low rank to work.
- **Path-shape features must be computed in the full 256-D space**, not after projection. The information is spread across components.
- **Trajectory clustering may work** but probably needs many centroids, not 3–4. Don't assume a small number of macro-clusters.

The flat spectrum is not a negative result; it's a calibration that says "analyze in high-D, don't compress first."

---

## 4. Eval noise floor → minimum detectable difference for Exp 2

Over steps 80K–100K (eval-plateau region), eval-to-eval val_bpb fluctuation is ≈ ±0.015 within this single run. Between-run variance under different seeds is typically larger than within-run late-training noise.

**Calibration for Exp 2 and Exp 5 interpretation:**

| val_bpb delta vs baseline | call it |
|---|---|
| < ±0.02 | noise — not informative |
| ±0.02 – ±0.03 | suggestive; re-run with a second seed before believing it |
| ±0.03 – ±0.05 | meaningful |
| > ±0.05 | substantial |

Use ±0.02–0.025 as the "is this real?" threshold for cross-run comparisons. Do not write the thesis on a 0.01 improvement. This number is calibrated here so we don't relitigate it later.

---

## 5. Train/eval divergence implies Exp 2 protocol change

Over 100K steps:
- Train loss: 0.63 @ 30K → 0.58 @ 80K → 0.56 @ 100K (still declining)
- Val bpb:    0.932 @ 30K → 0.815 @ 80K → 0.794 @ 100K (plateaued by ~85K)

Declining train loss with flat eval = model is beginning to memorize past step ~85–90K. This is not a problem for Exp 0 as a substrate check, but tells us something about the 3.4M-on-TinyStories regime: we are near the useful training horizon.

**Protocol consequences for Exp 2 and Exp 5:**

- **Train length:** default to ~90–95K steps, not 100K. Extra 5–10K steps are mostly fitting noise and add wall-clock without discriminative power.
- **Comparison metric:** compare **best val_bpb over the run**, not final val_bpb. All conditions will have the same plateau-then-overfit pattern, and their best points carry more signal than their terminal points.
- **Eval cadence:** keep eval every 1K steps — needed to locate each condition's best point reliably.

---

## 6. Framework deviation

Plan called for MLX preferred, PyTorch+MPS fallback. This experiment ran on PyTorch+MPS because a clean MLX port of the `train_r4` model would cost multi-day effort (CA modulation channels, value-embedding gates, backout lambda, block-diagonal CA init — all custom), and Exp 0's purpose is to validate the checkpoint-cadence infrastructure, which is framework-agnostic. MPS worked without hitting the known compatibility issues noted in memory.

**Carried forward:** Exp 2 and Exp 5 will also run on PyTorch+MPS unless a specific primitive (e.g., an efficient grid-position gather for the topographic regularizer) forces a port. This deviation is intentional and logged.

---

## 7. Positive findings (for later cross-reference)

- Baseline reproduces prior NeuroGen long-horizon eval loss within noise (0.7943 vs 0.7980) → **implementation is correct**. Any Exp 2 / Exp 5 deltas attribute to the intervention, not infrastructure.
- Displacement magnitudes are uniform (std/mean ≈ 5%) → **no scale artifacts** will contaminate trajectory analysis.
- Translation fraction ≈ 0 → **no trivial drift** contaminates the signal.
- Cadence hooks (snapshot rotation, full-ckpt rotation, jsonl logging) worked end-to-end on a 6-hour run → **reusable verbatim for Exp 2 conditions**.

---

## 8. Outstanding follow-ups

- [ ] Decide Exp 2 regularizer weight and anneal schedule — depends on Exp 1 results on trajectory density.
- [ ] Decide Exp 3 compositional axes — TinyStories byte-level means gender/tense composition isn't directly available; likely axes are character-class boundaries, digit arithmetic, or quoted-string patterns. Defer until Exp 2 is running.
- [ ] Decide Exp 2 step budget — currently proposed 90K–95K based on §5 analysis, to be confirmed when first Exp 2 condition is set up.

## 9. Next action

Proceed to **Exp 1 — trajectory visibility**:
- Per-token displacement already covered by this report (magnitude + PCA preliminary).
- New work: pairwise CKA between consecutive checkpoints (trajectory smoothness), per-token movement-magnitude curves over time (when each token stabilizes), clustering at early vs late training steps, qualitative inspection of high-frequency vs low-frequency byte trajectories.
- Expect high-D analyses to dominate; 2D projections are supplementary, not primary.
