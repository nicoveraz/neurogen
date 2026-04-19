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

Pipeline works end-to-end with the checkpoint cadence Exp 1–5 require. Displacement PCA flagged an ambiguous finding (flat spectrum) which was resolved by §10 interpretation diagnostics: **regime is high-dimensional structure with strong within-class clustering on final embeddings (regime A + B); regime C (noise) ruled out.** **Proceed to Exp 1.**

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

---

## 10. Addendum — Interpretation diagnostics for the flat displacement spectrum

### 10.1 Why this section exists

The §3 finding was genuinely ambiguous when first produced. A flat PCA spectrum on displacement vectors is consistent with three very different underlying stories, and they imply different things for Exp 1 and Exp 2. Pinning down which regime we're in is more important than proceeding to the full Exp 1 pipeline, because the wrong regime would mean Exp 1 produces noise. This addendum exists so that future-me (or anyone else returning to the project) understands that the §3 finding was not self-interpreting, sees the three alternatives that were on the table, and sees how they were distinguished.

### 10.2 Three interpretations of a flat displacement spectrum

**Regime A — genuinely high-dimensional structure.** Byte-level TinyStories has rich enough local structure that embeddings need many independent axes. The model discovered them; there is no low-rank compression because none exists. Trajectory analysis must run in the full 256-D space.

**Regime B — local structure, not global.** Tokens moved toward their semantic neighbors, but different tokens have different neighbors, so there's no global axis that captures all the motion. Pairwise analyses (who is near whom) will show structure that PCA cannot find.

**Regime C — noise.** Displacements are roughly random directions of roughly equal magnitude because gradient descent on a well-conditioned loss surface produces that pattern regardless of whether the model learned anything meaningful. Under C, the substrate hypothesis is wrong and the whole plan needs reconsideration.

### 10.3 Distinguishing diagnostic

Two cheap checks, designed together so that their results jointly pin down the regime:

- **Diagnostic 1 — pairwise cosine similarity.** Define byte classes where we have a strong prior of within-class semantic similarity (lowercase letters, uppercase letters, digits, whitespace, sentence-ending punctuation `.!?`, clause punctuation `,;:`, quotes, brackets, vowels, consonants). Compare mean within-class cosine similarity against a random-pair baseline, on both (a) final embeddings `w_L` and (b) displacement vectors `Δ = w_L − w_0`. Also compare across classes. If within-class >> random, regime A or B. If within ≈ random, regime C.
- **Diagnostic 2 — k=10 nearest-neighbor coherence.** For 21 anchor bytes covering each class, list the 10 nearest neighbors in final-embedding cosine. Eyeball whether neighbors are semantically coherent.

Script: `interpretation_diagnostics.py`. Data: `interpretation_stats.json`.

### 10.4 Results — Diagnostic 1 (within-class cosine lift vs random)

Random baselines over 500 uniformly sampled pairs:
- Final-embedding cosine: **+0.020**
- Displacement cosine: **+0.005**

Within-class lift (positive = class cluster tighter than random):

| class            | n  | `w_L` within | lift vs random | Δ within | lift vs random |
|------------------|---:|-------------:|---------------:|---------:|---------------:|
| lowercase letters| 26 | +0.118       | **+0.098**     |  −0.005  | −0.010         |
| uppercase letters| 26 | +0.189       | **+0.169**     |  −0.003  | −0.008         |
| digits           | 10 | +0.445       | **+0.425**     |  +0.002  | −0.003         |
| whitespace       |  4 | +0.060       | +0.040         |  +0.002  | −0.003         |
| sentence punct `.!?` | 3 | +0.502   | **+0.482**     | **−0.106** | **−0.111**   |
| clause punct `,;:`   | 3 | +0.333   | **+0.313**     |  +0.024  | +0.020         |
| quotes `"'`      |  2 | +0.225       | +0.205         |  +0.070  | +0.065         |
| brackets `()[]{}`|  6 | +0.014       | −0.006         |  +0.036  | +0.031         |
| vowels `aeiou`   |  5 | +0.206       | **+0.186**     |  −0.032  | −0.036         |
| consonants       | 21 | +0.130       | +0.110         |  −0.005  | −0.010         |

Cross-class cosine on final embeddings (should be lower than within-class if classes are separated):

| pair                            | `w_L` cross | Δ cross |
|---------------------------------|------------:|--------:|
| lowercase × uppercase           | +0.044      | −0.003  |
| lowercase × digits              | +0.086      | −0.000  |
| vowels × consonants             | +0.084      | −0.002  |
| sentence punct × clause punct   | +0.219      | +0.010  |
| digits × whitespace             | +0.039      | +0.001  |

**Reading the table:**
- Final embeddings show **very strong within-class structure**. Digits cluster at cosine +0.44, sentence punctuation at +0.50, uppercase letters at +0.19, vowels at +0.21. All classes (except `brackets` and `whitespace`) lift 5-25× above the random baseline.
- Cross-class cosine is consistently lower than within-class cosine, with a coherent nested structure (sentence-punct × clause-punct at +0.22 is higher than lowercase × uppercase at +0.04 — punctuation types are closer to each other than to letters, which is semantically right).
- **Regime C (noise) is ruled out.** The final embeddings are unambiguously structured.
- Displacement vectors, in contrast, are essentially uncorrelated within classes — within-class Δ cosine hovers around the random baseline for nearly every class. One interesting exception: sentence-punct Δ cosine is **−0.11**, meaningfully *anti*-correlated.

### 10.4a Reasoning artifact worth naming explicitly

The §10 work itself is the artifact: the §3 finding could have been taken at face value ("flat spectrum, therefore weak structure"), and the plan would have been quietly re-scoped around a false premise. Instead the ambiguity was named, interpretations A/B/C were written out, a distinguishing diagnostic was designed with A and B producing one signature (within-class lift) and C producing another (no lift), and the test was run. The interpretation in §10.5 below is what came out of that — but the process of *getting* to it, not just the conclusion, is the thing to remember. Whenever a trajectory-geometry finding is ambiguous in the same way later in this plan, re-run this same structure: name the alternatives, design the distinguishing test, run it, conclude.

### 10.5 Why the displacement spectrum was flat even though the structure is real

The final embeddings are structured (§10.4). The initial embeddings were drawn from `N(0, 0.8²)` per-element — independent, random directions. Displacement `Δ = w_L − w_0` therefore equals `structured_final − random_initial`. Since the initialization variance (‖w_0‖ ≈ 12.8 per token for 256-D Gaussian) is large compared with the typical intra-class separation in the final layout, **the random part of `w_0` dominates the variance in Δ**, obscuring the structured part. That is exactly what produces a flat PCA spectrum: the variance is spread across all 256 directions by the random initial state, not concentrated on the few directions along which the final layout organizes itself.

**Corollary:** the flat spectrum in §3 was never evidence against structure. It was evidence that the structure sits in the `w_L` term and that the `w_0` term was engineered (correctly) to be isotropic. The sign of the sentence-punct anti-correlation (−0.11) confirms this mechanically: `.`, `!`, `?` end up near each other in `w_L`, but they started from independent random `w_0`, so the vectors required to get from their random starts to a shared neighborhood must diverge — which is what an anti-correlated displacement pattern looks like.

### 10.6 Results — Diagnostic 2 (nearest-neighbor coherence)

For 21 anchor bytes, the 10 nearest neighbors in final-embedding cosine. Non-printable bytes shown as `\xNN`.

| anchor | top 10 cosine neighbors |
|---|---|
| `'a'`  | `'A'` (+0.30), `\x91`, `\x83`, `'='`, `\xa7`, `'i'`, `'o'`, `\xbb`, `'e'`, `'u'` |
| `'e'`  | `\xb3`, `\x91`, `'~'`, `'o'`, `'E'`, `'d'`, `'n'`, `\x00`, `'a'`, `'i'` |
| `'t'`  | `'T'` (+0.25), `'d'`, `'g'`, `'k'`, `'y'`, `','`, `\x91`, `'p'`, `\x02`, `'&'` |
| `'s'`  | `'S'` (+0.31), `'d'`, `\xab`, `\xa7`, `'~'`, `\x91`, `\xd3`, `\x92`, `'z'`, `'k'` |
| `'m'`  | `'M'` (+0.27), `\xae`, `\xbc`, `\xa1`, `'X'`, `'n'`, `\x91`, `\x9a`, `'r'`, `'t'` |
| `'A'`  | `'a'` (+0.30), `'E'`, `'S'`, `'L'`, `'J'`, `'B'`, `'U'`, `'T'`, `'O'`, `'W'` |
| `'T'`  | `'B'` (+0.34), `'S'`, `'t'`, `'A'`, `'W'`, `'K'`, `'M'`, `'L'`, `'J'`, `'I'` |
| `'S'`  | `'H'` (+0.31), `'s'`, `'T'`, `'P'`, `'A'`, `'D'`, `'B'`, `'R'`, `'J'`, `\x9a` |
| `'0'`  | `'5'` (+0.60), `'8'`, `'6'`, `'4'`, `'7'`, `'2'`, `'1'`, `'9'`, `'V'`, `'3'` |
| `'1'`  | `'2'` (+0.60), `'4'`, `'5'`, `'8'`, `'$'`, `'X'`, `'6'`, `'9'`, `'7'`, `'0'` |
| `'5'`  | `'4'` (+0.67), `'6'`, `'8'`, `'0'`, `'7'`, `'2'`, `'3'`, `'1'`, `'V'`, `'9'` |
| `' '`  | `\x9d` (+0.36), `\x0a`, `'/'`, `\xa6`, `\x93`, `\xc5`, `\xec`, `\xae`, `\xfa`, `\xa7` |
| `\x0a` | `' '` (+0.34), `'.'`, `\xc5`, `\xb0`, `'!'`, `\x93`, `\x06`, `\xa0`, `\xa6`, `\xaa` |
| `'.'`  | `'!'` (+0.62), `'?'`, `','`, `'&'`, `\xae`, `\xa6`, `\x0a`, `';'`, `\xbf`, `'='` |
| `','`  | `'&'` (+0.38), `'.'`, `';'`, `\xa1`, `'!'`, `\xbc`, `\x80`, `\xa6`, `\xae`, `'t'` |
| `'!'`  | `'.'` (+0.62), `'?'`, `','`, `\xae`, `'&'`, `\xa6`, `\xaa`, `\xb9`, `\xbf`, `';'` |
| `'?'`  | `'!'` (+0.47), `'.'`, `\x80`, `'\\'`, `\xaa`, `\xb3`, `'-'` *(no, \xac)*, `','`, ... |
| `'"'`  | `\x93` (+0.80), `\xac`, `'('`, `\x9c`, `'<'`, `\x85`, `\xb0`, `\xa0`, `\xa6`, `\xbe` |
| `"'"`  | `` '`' `` (+0.66), `\xa2`, `\xb4`, `\x9c`, `\x84`, `'3'`, `'V'`, `\xab`, `'"'`, `'#'` |
| `'('`  | `'"'` (+0.40), `\x93`, `'['`, `\x9c`, `\xac`, `\xbf`, `\x9a`, `\xa3`, `'2'`, `'`' ` |

**Highlights:**
- **Case-pairing.** `a` → `A`, `t` → `T`, `s` → `S`, `m` → `M` all appear as the top or near-top neighbor. The model learned that uppercase/lowercase versions of the same letter are semantically linked.
- **Digit cluster is surgical.** For `'0'`, `'1'`, `'5'` — the 10 nearest neighbors contain 9 or 10 other digits. Cosine within the digit cluster hits +0.67 (`5`-`4`) and +0.60 (`0`-`5`, `1`-`2`). This is the cleanest byte class in the embedding.
- **Sentence-punctuation triad is tight.** `.`-`!`-`?` are mutual nearest neighbors at cosine 0.47-0.62.
- **Uppercase letters cluster among themselves**, with occasional bleed to the matching lowercase (`A` → `a`, `S` → `s`, `T` → `t`).
- **Space and newline are each other's neighbors** (`' '` ↔ `\x0a` at cosine +0.34). Plus several high-bytes — likely infrequent UTF-8 continuation bytes that also appear adjacent to whitespace in the corpus.
- **Quotes and brackets cluster into an "opening/closing pair" group.** `(` → `"`, `[`. `'` → `` ` ``. `"` → `\x93` (which is the Windows-1252 code for the left double quote — TinyStories likely contains some mixed-encoding source text; the model grouped the UTF-8 byte `"` with the CP1252 byte `\x93` because they appear in similar contexts).

The interpretation check is **overwhelmingly positive**. Structure is present, it is semantically coherent, and it is visible without sophisticated analysis.

### 10.7 Verdict

Regime A + B, with a mechanistic addendum to explain §3.

- **Final embeddings are strongly structured** (Diagnostic 1 within-class lifts of +0.10 to +0.48; Diagnostic 2 nearest-neighbor tables are semantically coherent).
- **The structure is high-dimensional and class-local.** There is no global axis along which "all tokens moved", because there isn't a single axis that distinguishes all the interesting classes — digits, letters (upper/lower), punctuation, whitespace, quotes all occupy their own subspaces. Different classes pull on different directions.
- **Regime C is ruled out.** The substrate hypothesis for Exp 1 holds.

### 10.8 Implications for Exp 1 — analysis approach

- **Primary tools are pairwise, not global.** CKA between snapshots, cosine similarity matrices over time, nearest-neighbor stability across training — all operate in the full 256-D space and all will produce signal.
- **Work on the `w_L`-like quantity, not on displacements, whenever possible.** The `w_0` isotropy hides the structure when it is subtracted out. When displacements are genuinely the object of interest (e.g., "which tokens changed direction during training"), remember that the baseline variance is inflated by `w_0`, so tests need to account for that.
- **2D PCA projections are supplementary, not primary.** The flat spectrum in §3 already predicted this. Include 2D plots as sanity checks but do not read them as primary evidence.
- **The sentence-punctuation anti-correlated displacement (Δ cos = −0.11)** is worth following in Exp 1's time-resolved analyses. It is the clearest trajectory-level signature of "different starts, shared destination" in the dataset, and may be a good case study for the Exp 4 convergence-detection analysis.
- **Frequency dependence to check.** High-byte (\x80-\xff) neighbors showing up in many letter anchors' nearest-neighbor lists hints that rare bytes may be the noisiest, least-structured part of the vocabulary. Expect Exp 1 movement-over-time curves to show rare bytes stabilizing late (or not at all) while common bytes stabilize fast.

### 10.9 Implications for Exp 2 — topographic grid dimensionality

The natural representational structure is high-dimensional in the sense that **multiple class subspaces coexist** (digits, upper, lower, punct, ws, quotes, brackets) and each class uses its own direction(s). A 2D grid is therefore a strong low-dimensional constraint that will compress these subspaces onto the plane.

Two paths:

- **Conservative (recommended — chosen).** Run Exp 2 as planned with a 2D grid. Expect muted effects. If the topographic condition shows no improvement over baseline, that is still informative: it means a 2D manifold is too restrictive for this regime. If it shows a small improvement, a higher-dimensional grid is a natural escalation with strong motivation. Staying 2D keeps the result comparable to existing topographic-learning literature.

- **Aggressive.** Pre-emptively use a higher-dimensional grid (e.g. 16×16×4 or a small hypercube). Compatible with the finding but less interpretable against prior work.

Commitment: **conservative option**. Document the expectation of muted effect up front so it is not surprising.

### 10.10 Connection to existing work

The pattern — features encoded as directions, many more directions than dimensions would naively allow, no single global axis capturing the variance — is the small-model echo of the superposition phenomena reported in mechanistic interpretability work on larger models (Elhage et al. on toy models of superposition; sparse autoencoder literature; monosemantic-feature decomposition). At 3.4M on TinyStories, byte-level classes are the "features"; each class gets its own direction; they coexist in 256-D without mutual interference because the classes themselves are small and well-separated. This is a plausible frame for the eventual writeup, and the Exp 4 trajectory-geometry results may sharpen or complicate it.

### 10.11 Outstanding caveats to carry forward

- The Δ baseline test (random-pair cosine = +0.005) is near enough to zero that the small negative lifts for most classes are probably within noise of zero. Do not read "lowercase Δ cosine −0.01" as a real anti-correlation — only sentence-punct at −0.11 is large enough to be meaningful.
- Byte-level granularity means the "classes" here are orthographic, not lexical. The finding generalizes to "the model learned character-class structure", not necessarily to "the model learned word-level semantics". See §11 on Exp 3 scope honesty.
- The "nearest neighbors include high bytes" signal noted in §10.8 may deserve a small standalone follow-up: are high bytes genuinely noisy, or are they semantically close to specific low-byte tokens via their co-occurrence pattern? Park until Exp 1.

---

## 11. Methodological principle — position vs displacement

Pinning this down now because it generalizes past Exp 0 and we will forget it by Exp 4 otherwise.

**Principle.** When analyzing embedding trajectories of a trained network, work on the **positions** `w_t` whenever possible, not on **displacements** `Δ = w_t − w_s`. Displacements carry the initialization noise as a baseline: `Δ = structured_t − random_s` has its variance dominated by the isotropic random term, which masks any structure that sits in the positions. Exp 0 §10.5 is the concrete demonstration — the digits class cluster tightly in `w_L` (cosine +0.44) but have near-zero pairwise Δ cosine.

**Operational rules for Exp 1 and Exp 4.**
- Pairwise CKA, cosine-similarity matrices, clustering, nearest-neighbor stability → compute on **positions** `w_t` at each checkpoint.
- Within-class coherence curves (class cosine over time) → positions.
- Total path length, per-step velocity, instantaneous turning → these are displacement-based by definition; when used, compare against a null baseline generated from the same initialization distribution, not against raw zero.
- **Preferred null baseline: fresh `w_0 ~ N(0, 0.8²)` sampling** rather than shuffling the observed `w_0` across tokens. Fresh sampling makes no assumption about correlations between the observed init and the final layout; shuffling destroys such correlations but only for the specific instantiation. Default to fresh sampling unless there is a specific reason to want the shuffle.
- **Signature to look for:** when two tokens' positions converge but they started from independent random inits, their pairwise Δ cosine will be *anti*-correlated (Δ cos < 0). This is the trajectory fingerprint of "different starts, shared destination". Exp 0 found this for `.!?` at Δ cos = −0.11. Exp 4's convergence-detection analysis should look for additional triads with the same signature.

---

## 12. Named case studies to track through Exp 1 → Exp 4

**This section is the cross-experiment master tracker** for three specific artifacts from the Exp 0 diagnostics. The temptation in later experiments will be to produce mean-across-vocabulary statistics that smooth over individual cases; keeping these three named means every stage has a concrete check: how does the `.!?` triad, the digit cluster, and the `"`/`\x93` pair look under the new analysis? If the aggregate is strong but the case studies are odd, the aggregate is hiding something. If the aggregate is weak but the cases are clean, the aggregate is being dragged by noise.

As new numbers come in for each case study, pin them here (with the originating report) so §12 becomes the single document to cite.

### 12.1 Sentence-punctuation triad `.`, `!`, `?`

Mutual nearest neighbors in `w_L` at cosine 0.47–0.62. Δ cos = −0.11 (the cleanest "different starts, shared destination" signature — see §10.5 and §11).

- **Exp 0 signature:** Δ cos = −0.11; final cos pairs `.–!` +0.62, `.–?` +0.42, `!–?` +0.47.
- **Exp 1 §1 formation step:** **3000** (first step where class within-cos > random + 0.05 sustainedly).
- **Exp 1 §2 convergence dynamics (resolved):** sequential, anchor-driven. `.` reaches final-centroid cos ≥ 0.6 at **step 3000**; `!` at **step 12000**; `?` at **step 16000**. 4-5× sequential separation. `.` acts as anchor (most frequent byte of the three, most distinctive context); `!` and `?` migrate toward it. Pairwise threshold crossings confirm: `.–!` reach cos ≥ 0.3 at step 6500, `.–?` at step 9000, `!–?` at step 9500 (last).
- **Exp 4 carry-forward:** canonical case for "anchor-driven convergence" as a class of trajectory dynamics. Search for additional triples where one member starts closest to the final centroid; predict that anchor-ness correlates with corpus frequency.

### 12.2 Digit cluster `0`–`9`

Tightest class cluster in `w_L` (+0.44 within-class cosine; nearest neighbors essentially pure digits at cosine 0.60–0.67).

- **Exp 0 signature:** within-class cos +0.44 (step 100K), nearest neighbor pairs up to +0.67 (`5`–`4`). Originally flagged as the most promising axis for a digit-arithmetic compositional test in Exp 3.
- **Exp 1 §1 formation step:** 6500.
- **Exp 1 §4 phase structure (resolved):** peak-then-decline pattern. Coherence peaks +0.49 at step 47K, declines to +0.44 at step 100K. Meanwhile `r(pairwise cos, |i−j|)` trajectory: +0.28 @ init → +0.06 @ 10K → **−0.29 @ 20K → −0.39 @ 70K** → −0.35 @ 100K. Steepest ordinal-correlation emergence happens **after** the coherence peak — the cluster's loosening carries ordering information. Two-stage learning: Phase 1 (0-20K) coarse category formation; Phase 2 (20K-70K) within-category ordinal refinement; Phase 3 (70K-100K) stabilization with mild loosening on numerically-distant pairs.
- **Mechanism correction (see §13.2 below).** The corpus-frequency check performed as part of the Exp 3 scope review revealed digits are ~0.0035% of the corpus and are dominated by the formulaic "3-year-old" template. The ordinal axis the model learned is **age/count context similarity** (digits in similar age contexts cluster by age adjacency), **not arithmetic composition**. The two-phase story stands; the interpretation of what Phase 2 is refining does not.
- **Exp 2 prediction (Exp 1 §4 and §9; relabeled post-corpus-check).** A 2D topographic grid will help Phase 1 (category formation — the grid encodes "digits belong together") but hurt Phase 2. Phase 2 was originally labelled "ordinal / arithmetic refinement"; post-corpus-check the correct label is **"context-adjacency refinement"** — digits that share "N-year-old" contexts cluster by age adjacency. The geometric test is unchanged: `r(cos, |i−j|)` at step 90K should be less negative in the topographic condition than in the baseline. If observed, it's evidence that the 2D grid constrains the within-category refinement axis the model would otherwise learn, regardless of whether that axis is arithmetic or context-adjacency. If the topographic condition instead *preserves* the within-category axis, the grid is compatible with Phase-2 refinement — also informative.
- **Exp 4 carry-forward:** peak-then-decline is a formalizable trajectory feature. Two modes distinguished in Exp 1 §5: ordinal refinement (digits, mild decline) vs role specialization (`()` pair, dramatic decline). Both need feature definitions for aggregate Exp 4 analyses.

### 12.3 Mixed-encoding `"` ↔ `\x93` pair

Cosine +0.80 in `w_L`. `\x93` is the Windows-1252 left-double-quote byte; the UTF-8 corpus contains mixed-encoding text, and the model discovered the equivalence via co-occurrence.

- **Exp 0 signature:** `"`'s nearest neighbors are `\x93` (+0.80), `\xac`, `(`, `\x9c`, `<`, `\x85`, `\xb0`, `\xa0`, `\xa6`, `\xbe`. Concrete evidence the model learned structure not explicitly annotated in the corpus.
- **Exp 1 formation step:** **deferred.** The Exp 1 pass did not deep-dive this case (Exp 1 §3 flags the deferral). A 20-minute follow-up that tracks `"` ↔ `\x93` cosine across all 111 snapshots would resolve whether alignment is (a) early, driven by raw byte co-occurrence at character-level, or (b) late, driven by contextual role discovery. Open question.
- **Candidate example for the writeup** (per Exp 0 §10.6). Concrete, memorable, grounds the superposition / direction-as-feature framing (§10.10). Preserve visibility through all later analyses.

### 12.4 Candidate additions

From Exp 1 §5 (peak-then-decline mode "role specialization"): **`()` pair** — peak +0.30 @ step 32K, final +0.06 (−80% decline). The cleanest role-specialization signature in the vocabulary. Worth tracking alongside the three above as a fourth case study if subsequent experiments confirm the role-specialization mode generalizes beyond this one pair.

---

## 13. Exp 3 scope — positive framing and corpus-grounded axes

### 13.1 Reframed: cleaner test, weaker claim

At byte level on TinyStories, the compositional axes directly testable are orthographic and character-class, not the semantic composition that motivates the SCAN / COGS / combinatorial-tail literature. The scope caveat is real and must be labelled clearly in any writeup. But the positive framing is more useful to lead with:

**Character-class composition at byte level is a *cleaner* test than semantic composition at word level.** The primitives (byte classes, co-occurrence structure) are more discrete, the combinations are more enumerable, and the noise from polysemy and context-dependence is lower. For a representation-learning audience interested in trajectory and topographic phenomena, the cleanliness is the property that makes the test informative. Weaker claim, cleaner test — and the mechanism findings (Exp 0 §10.5, Exp 1 §4) are genuinely mechanism-level claims about how representations form, which is the frame our work belongs in.

Suggested writeup framing: lead with the mechanism and the cleanness of the testbed; place the scope caveat second, not first, while keeping it explicit. "We ran a clean byte-level test of topographic and trajectory-conditioned representations on a 3.4M-parameter model trained on TinyStories, and observed [findings]. This is a character-level rather than a semantic-level test of compositional generalization; scaling to the semantic claim would require a word-level tokenizer and a larger model."

### 13.2 Corpus-grounded axis selection — digit-arithmetic is NOT viable

The original plan-carry in this section speculated that the digit cluster (+0.60–0.67 cosine) might support a digit-arithmetic compositional test — "all `(3, n)` and `(n, 2)` arithmetic facts present, hold out `(3, 2)`." Before committing to this axis, ran a corpus-frequency check on 200M bytes (~10% of TinyStories training data); results at `analysis/exp1_trajectories/digit_corpus_frequency.json`. The check rules the axis out:

| measurement | value | implication |
|---|---:|---|
| Total digit bytes (0-9) in 200M sample | 7,096 (0.0035% of corpus) | digits are vanishingly rare |
| Digit `3` alone | **5,172 of 7,096 (73%)** | one token dominates the class |
| Ratio digits : number-words ("one"-"ten") | **1 : 25** | corpus prefers number-words |
| Regex `\d\s*\+\s*\d` matches | **0** | no addition syntax |
| Regex `\d\s*=\s*\d` matches | **0** | no equation syntax |
| Literal `' + '` | **0** | no additive operator tokens |
| Sample contexts of digit `3` | universally `"was 3 years old"` / `"a 3 year old"` | formulaic "N-year-old" template dominates |

The digit cluster from Exp 0 is not arithmetic-encoding — it is **age / count context similarity**. Adjacent digits share "N-year-old" environments; `3` and `4` cluster because both appear in child-age stories. This reinterprets Exp 1 §4 mildly (the two-phase coarse-category-then-ordinal-refinement story stands mechanically; the specific axis the model learned is not numerical arithmetic, it is numerical age-adjacency).

### 13.3 Revised Exp 3 axis options

- **Character-class boundaries (primary).** Cheap, valid, well-matched to what the model represents. E.g., held-out letter trigrams where each letter appears separately in training but the trigram does not; held-out quote-open / quote-close matchings across encodings; held-out punctuation-sequence patterns.
- **Number-word axis (new candidate, replacing digit-arithmetic).** Number words are plentiful ("one": 135K uses, "two": 23K, "three": 13K, "four": 1.3K, "five": 751 in 200M). They have genuine quantitative usage. A compositional test could hold out specific number-word × noun combinations (e.g., all `"three <animal>"` contexts present and all `"<number> dog"` contexts present for various nouns and numbers, but `"three dogs"` never co-occurs). At byte level this is harder to construct cleanly than character-class axes because "three" is a five-byte span not a single token, but it is closer to the original semantic-composition motivation than character-class alone.
- **Formulaic-template violation (fallback).** If the above are too expensive to construct, use the "N-year-old" template itself as a compositional test axis: hold out specific N values from the training template, check whether the model generalizes the template slot structure to held-out numbers. This is close to behavior Exp 0 already probed implicitly and would be cheap.

### 13.4 Acknowledged lost ground

The original research motivation — systematic semantic compositional generalization — is not directly testable in this plan. The downscaling to character-class and templated-numeric axes is a real loss and any writeup must be explicit about what claim we are testing vs what claim the combinatorial-tail argument is about. This section is the reference for that honesty; the cleaner-test reframing in §13.1 does not erase the scope shift.
