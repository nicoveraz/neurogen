# Writeup outline — Exp 0 + Exp 1 + Exp 2 gradient-vanishing

**Status:** pre-draft outline for review. No sections drafted yet. **Scope decisions requested** at the end.

## Provisional title

Two candidates, pick one:

- **A (mechanism-forward):** *Phase-Structured Representation Formation in Small Transformers: A Trajectory-Analytic Account*
- **B (contrast-forward):** *Trajectories Tell What Endpoints Don't: Small-Scale Representation Dynamics and the Limits of Topographic Regularization*

Title B advertises the Exp 2 negative result up front; title A keeps Exp 2 as an appendix-like section. My vote is **B** — the combined framing ("what trajectories show + what trying to shape them can't do") is the distinctive contribution. But A is more conservative and closer to standard framing.

## Audience and venue options

| venue | length | fit | pacing |
|---|---|---|---|
| arXiv + Substack companion | 10–15 pages | high — independent research, mech-interp-adjacent community | draft, iterate, post in ~2 weeks |
| Workshop paper (e.g. ICLR Re-Align, NeurIPS Mech-Interp, ICML TAG-ML) | 4–8 pages | high for workshops on representation analysis / developmental dynamics | would need to check specific CfPs; deadlines drive timing |
| Short conference paper (track TBD) | 8–10 pages | medium — negative-result halves of papers can be hard to place | longer path |
| Blog-only (Substack/LessWrong) | 2500–4000 words | clean for the mech-interp audience; faster | ~1 week |

**Default recommendation:** arXiv preprint + Substack post. Write at the ~12-page technical level; distill to blog post after arXiv is up. If a workshop CfP with a good match appears while drafting, adapt the arXiv draft to fit.

## Key claims — three, in priority order

1. **Trajectory analysis surfaces phase-structured phenomena invisible in endpoint analysis.** Concretely: the digit class forms a coarse category (phase 1, steps 0–20K), then refines along a within-category ordinal axis (phase 2, steps 20K–70K), loosening the crude category in favor of a more-informative gradient. Visible only in the trajectory; the endpoint shows the refined state without the phase history. The Saxe/McClelland hierarchical-feature-emergence story observed at 3.4M byte-level on TinyStories, in a class small enough to see the whole structure in a 10×10 matrix.

2. **Anchor-driven sequential convergence is a reproducible trajectory motif.** The `.!?` triad converges from random starts via a specific dynamic: the most frequent member (`.`) reaches its final region first at step ~3000, and the other two migrate toward it, crossing pairwise cosine thresholds 4–5× later. The anchor is identified by initial-cosine-with-final-centroid, which correlates with corpus frequency. A formalizable trajectory feature for identifying anchor tokens.

3. **Gaussian-kernel topographic regularizers are structurally pathological for gradient-based learning of grid positions.** Any loss factoring through `K(d) = exp(−d²/2σ²)` has vanishing gradient at both `d → 0` and `d → ∞`, so stable equilibria of such losses are stationary points but not attractors. Three Exp 2 pilot formulations (Gaussian pure attractive, MSE-simple, equilibrium-MSE) each produced distinct-looking pathologies that unify under this property. Distance-based losses avoid the problem. SOM-style non-gradient competitive learning avoids it via different mechanism (fixed grid positions, learning weights).

Claims 1 and 2 are positive contributions. Claim 3 is a "what we tried and why it doesn't work" finding that is useful to the field — future researchers attempting topographic regularization on transformers should know about the pathology.

## Section outline

### Section 1 — Introduction

- The representation-learning puzzle: endpoints show the organized state, but how did the organization form?
- Why small-scale matters: superposition literature and mech-interp insights are often derived from models where you can see every feature; NeuroGen 3.4M on TinyStories is the small-model echo of that, with the advantage that the whole vocabulary is 256 bytes.
- Three contributions (summary of claims 1–3 above).
- Scope caveat (lead positive, caveat second, per `feedback_exp3_scope_honesty.md`): "cleaner test, weaker claim" — byte-level TinyStories is a clean testbed for representation dynamics, not a direct test of the semantic-compositional claims that motivate some of this research thread.
- Roadmap.

**Target length:** 1.5 pages.
**Figures:** none; a single pull-quote result ("digit class coherence peaks at step 47K at cosine +0.49, then declines to +0.44 with ordinal correlation strengthening from −0.29 to −0.39") can go here.

### Section 2 — Setup

- Model: 3.4M-parameter transformer (NeuroGen baseline architecture — depth 4, channels 256, byte-level vocab 256, 256-token context). One-liner reference to the architecture details in `train_r4.py`.
- Corpus: TinyStories, byte-level encoding, 1.9B bytes training.
- Training: 100K steps (Exp 0), 10K steps per pilot (Exp 2), AdamW with cosine LR schedule, M1 Pro / MPS.
- Methodological principles that matter for the findings:
  - Dense-early snapshot cadence (500-step for first 10K, 1000-step after) — most formation happens early (validated retroactively).
  - Positions-not-displacements for trajectory analyses (the core principle; introduced via Exp 0 §10.5's flat-spectrum resolution).
  - Matched-null comparison for cross-codepath variance (required given MPS ~0.055 val_bpb non-determinism at 10K steps).
  - Substrate-check before committing to interpretations (detailed in appendix / methodology section as a small methodological contribution).

**Target length:** 2 pages.
**Figures:** none required; a method-summary diagram optional.

### Section 3 — What we found: phase-structured representation formation

**Subsection 3.1 — Within-class trajectory coherence (headline figure).**

- Definition: mean pairwise cosine of byte positions over training, per byte class.
- Finding: ten byte classes, two qualitative regimes: monotone growth (letters, punct) and peak-then-decline (digits, clause-punct, `()` pair).
- Figure: `reports/exp1/headline_coherence.png` — ten traces on log-step x-axis, with annotations for the two peak-then-decline cases. **This is the paper's Figure 1.**
- Formation-step table across classes.

**Subsection 3.2 — Digit deep dive (the centerpiece).**

- The peak-then-decline pattern on digits is the most important single finding.
- Figures: `reports/exp1/digit_pairwise_peak_vs_final.png` (matrices at peak and final) and `reports/exp1/digit_ordering_correlation.png` (trajectory of ordinal `r(cos, |i−j|)`).
- Result: coherence peaks +0.49 at step 47K, declines to +0.44 at 100K; ordinal correlation strengthens from +0.28 (init) to −0.39 (step 70K). Two-phase learning: coarse category (phase 1), then within-category ordinal refinement (phase 2).
- Mechanism interpretation: ordinal axis is age/count context adjacency (formulaic "N-year-old" in TinyStories), not arithmetic — validated by the 200M-byte corpus frequency check showing zero arithmetic patterns. Honest scope caveat: *the phase-structure finding stands, the axis-label finding was mechanism-corrected post-hoc, and the latter distinction is worth the honesty because it determines what we can and can't claim*.
- Testable prediction: classes currently monotone may show their own Phase 2 at longer training (deferred follow-up).

**Subsection 3.3 — Anchor-driven convergence: the `.!?` triad.**

- Case study: `.!?` sentence punctuation triad.
- Finding: sequential, not simultaneous. `.` reaches final-centroid cosine ≥0.6 at step 3000; `!` at 12000; `?` at 16000. 4–5× sequential separation. `.` is the anchor.
- Figures: `reports/exp1/triad_centroid_pull.png` and `reports/exp1/triad_pairwise_cos.png`.
- Correlate with corpus frequency: `.` is most frequent by far — anchor-ness tracks frequency. Formalizable feature: *for any byte cluster, the anchor is the member with highest initial-cosine-with-final-centroid*.
- Deferred: verify on 3–4 additional triads (listed in §5 as open question).

**Subsection 3.4 — Two modes of peak-then-decline.**

- Peak-then-decline isn't a single dynamic; it's two distinct modes:
  - **Ordinal refinement** (digits): cluster partially loosens as an interpretable within-category axis emerges.
  - **Role specialization** (`()` pair): cluster loosens dramatically (−80% peak-to-final) as members specialize into opposing roles.
- Table distinguishing the modes, with the specific measurements for each.
- Figure: `reports/exp1/bracket_subclass_coherence.png` — shows `()` peak-then-decline alongside the combined-brackets failure and the openers/closers subclass split (data-sparsity confounder noted honestly).

**Subsection 3.5 — Trajectory smoothness (CKA).**

- Consecutive-snapshot linear CKA ≥ 0.98 throughout: no discontinuous reorganization.
- Minimum at step 20K — fastest *relative* reorganization, coincides with the Phase 1 → Phase 2 transition for digits.
- CKA(t, init) and CKA(t, final) cross near step 22K — representational halfway at 22% of training steps, consistent with the dense-early cadence validation.
- Figure: `reports/exp1/consecutive_cka.png`.

**Subsection 3.6 — Common vs rare byte velocity (supplementary).**

- Per-token velocity analysis. Common bytes stay active longer; rare bytes have brief bursts followed by dormancy.
- Methodological note: the "stabilization" metric used here is biased for rare bytes; redefined in the appendix.
- Figure: `reports/exp1/token_movement.png` (may demote to appendix depending on length budget).

**Target length:** 5–6 pages for §3.
**Figures:** 5 main-text figures (3.1, 3.2 [2 figures], 3.3 [2 figures], 3.4, 3.5), 1 supplementary.

### Section 4 — What we tried and why it didn't work: topographic regularization

**Subsection 4.1 — Design rationale.**

- Hypothesis: topographic organization of token embeddings on a 2D grid (co-occurrence-driven) might improve compositional generalization by imposing geometric structure that aids learning.
- Design: 256 bytes on a 16×16 grid; pairwise loss drives co-occurring bytes to be grid-adjacent.
- Inspired by SOM (Kohonen 1982) and topographic organization in cortex.

**Subsection 4.2 — Three formulations, three pathologies.**

Table (reproduce from `reports/exp2/gradient_vanishing_analysis.md`):

| formulation | loss | observed pathology |
|---|---|---|
| Gaussian pure attractive | `L = − Σ w_ij · K(d_ij)` | Collapse to coincidence; grid-spread ratio saturates at 0.56 regardless of LR (pilots 1 / 2A / 2B) |
| MSE-simple, target=0 for non-cooccur | `L = mean((K − cap·w_ij)²)` | Escape; positions expand past 1.5× init spread, kernel goes near-zero, gradients vanish (pilot 2C-MSE) |
| Equilibrium-MSE, target=max(er(σ), cap·w_ij) | Same loss, corrected target matrix | Bimodal; high-w pairs collapse past equilibrium d=1.38 to d≈0.2, low-w pairs escape past d=5.73 to d≈16 (pilot 2D) |

Figure: a single overlay plot of the three pilot trajectories' diagnostics (grid-spread ratio, topo-loss, frac-positions-moved) is useful here. **Doesn't exist yet; needs to be produced.**

**Subsection 4.3 — Unified diagnosis: gradient-vanishing at both tails of a Gaussian kernel.**

- Analytical result: for any loss `L = f(K(d_ij), T_ij)`, the gradient with respect to position is proportional to `[∂f/∂K] · K(d) · (pos_i − pos_j)`. At `d → 0` the `(pos_i − pos_j)` factor vanishes; at `d → ∞` the `K(d)` factor vanishes. Any loss factoring through `K(d)` inherits this envelope; no target matrix can force gradient into the vanishing regions.
- Consequence: stable equilibria of such losses are stationary points but not attractors. They lack pullback basins on either side.
- Empirical confirmation: the three pilot pathologies are three different manifestations of the same property.
- Reference to `reports/exp2/gradient_vanishing_analysis.md` for the full derivation.

**Subsection 4.4 — What avoids the pathology.**

- Distance-based losses (`L = mean((d_ij − target_d_ij)²)`) have gradient magnitude proportional to distance error. Globally well-behaved; equilibria are true attractors.
- SOM-style non-gradient competitive learning avoids the issue entirely (fixed grid positions, learning weights).
- Recommendation for future topographic work: default to distance-based formulations or SOM-style dynamics, not kernel-based gradient formulations.

**Target length:** 3–4 pages for §4.
**Figures:** 1 main-text overlay figure (to produce) + maybe a derivation-visualization.

### Section 5 — Discussion

- What the trajectory findings say about small-scale representation learning (phase structure, anchor-driven dynamics, two-mode peak-then-decline as a formalizable trajectory phenomenon).
- What the gradient-vanishing finding says about the difficulty of imposing geometric structure via gradient-based losses in transformer embedding spaces.
- Connection to mech-interp / superposition literature (Elhage et al.; sparse autoencoder work): the direction-encoded-features story is consistent with the flat PCA spectrum on displacements (Exp 0 §10.5) — features live in many directions, positions in high-D space.
- Honest scope limits:
  - Byte-level tests orthographic composition, not semantic composition (SCAN/COGS). Digit-arithmetic axis is not available in TinyStories (corpus check).
  - Results at 3.4M may not extend to larger models; the whole plan was "small-scale signal worth scaling" and we haven't yet scaled.
- Open questions:
  - Do monotone-growth classes show their own Phase 2 at longer training?
  - Does anchor-driven convergence generalize to additional triads?
  - Does distance-based topographic regularization actually produce measurable effects on downstream dynamics (open to future work — this paper is not claiming it does or doesn't, only that Gaussian-kernel formulations don't).
  - Would initialization-only topography (Option B from the project decision) produce effects? Cleanest follow-up.

**Target length:** 1.5–2 pages.

### Section 6 — Conclusion

Short, 0.5 page. Summarize three claims.

### Appendices

- A. **Methodological principles** (positions over displacements, matched null, substrate-check habit, dense-early cadence). Short, but makes the repro story clean.
- B. **Corpus characterization** (digit frequency, number-word check) — for the mechanism-correction in §3.2.
- C. **Additional figures** — displacement PCA, token movement, interpretation-diagnostic NN tables.
- D. **Full Exp 2 pilot trajectories** — data for the three pathologies.
- E. **Reproducibility** — code pointers (`runs/exp0_baseline`, `analysis/exp1_trajectories/*.py`, `autoresearch/*.py`), git commit SHA, compute budget.

## Figure inventory

Ready to use as-is:
- [x] `reports/exp1/headline_coherence.png` — §3.1 (Figure 1)
- [x] `reports/exp1/digit_pairwise_peak_vs_final.png` — §3.2
- [x] `reports/exp1/digit_ordering_correlation.png` — §3.2
- [x] `reports/exp1/triad_centroid_pull.png` — §3.3
- [x] `reports/exp1/triad_pairwise_cos.png` — §3.3
- [x] `reports/exp1/bracket_subclass_coherence.png` — §3.4
- [x] `reports/exp1/consecutive_cka.png` — §3.5
- [x] `reports/exp1/token_movement.png` — §3.6 / appendix
- [x] `reports/exp0/displacement_pca.png` — appendix

Need to produce:
- [ ] Exp 2 three-pilot overlay (grid-spread, topo-loss, frac-moved across formulations) — §4.2. Estimated: 1 hour.
- [ ] Gradient-envelope illustration — §4.3. Simple line plot of `|∇L|` vs d for a kernel-based loss vs a distance-based loss. Estimated: 30 min.

## What's missing / still needed

**Analytical:**
- Anchor verification on 3–4 additional triads (listed in discussion as open; could do this in ~2 hours as a §3.3 robustness check, would strengthen claim 2).
- Sanity check that distance-based formulation works as claimed (single 10K pilot; user flagged that this is optional — an appendix-level substantiation rather than a core requirement).

**Writing:**
- Everything. Nothing drafted yet.

**Decisions:**
- Title (A or B)
- Venue (arXiv + Substack vs workshop vs blog only)
- Length target (12 pages arXiv vs 6-page workshop vs 3500-word blog)
- Scope of claim 2: do we run the 3-triad verification, or leave it as a claim with one case study + a deferred-verification note?

## Proposed drafting sequence

1. **Outline approval** (this document).
2. **Figure production** for §4.2 and §4.3 (~1.5 hours).
3. **Section 3 draft** (the positive findings; strongest part; 2 days).
4. **Section 4 draft** (the negative result; 1 day — most of the work is already in `reports/exp2/gradient_vanishing_analysis.md`).
5. **Section 2 draft** (setup; 0.5 day).
6. **Section 1 + Section 5 drafts** (intro + discussion; 1 day; these come last because they depend on the findings being settled).
7. **Section 6 + appendices** (0.5 day).
8. **Full-pass edit** (1 day).

Total: ~6–7 working days to a complete arXiv-quality draft. Blog-post adaptation after.

## Decision items for your review

1. **Title: A or B.** My vote: B.
2. **Venue and length: arXiv (~12 pages)? Workshop? Blog only?** My vote: arXiv + Substack companion.
3. **Should claim 2 (anchor-driven convergence) be supported by just the `.!?` case + a frequency-correlation argument, or should I run the 3–4 additional triads first?** My vote: run 3 additional triads (~2 hours) to make the claim robust; the finding is worth solidifying.
4. **Scope of claim 3's "recommendation" wording.** Options: (a) "future topographic work should default to distance-based formulations" (strong, prescriptive), (b) "distance-based formulations avoid the gradient-vanishing pathology; which approach is best is an empirical question" (descriptive, neutral). My vote: (b) — the prescriptive form overclaims without having actually run a distance-based main experiment to show it works at scale. The descriptive form is honest and more durable.
5. **Do you want the optional appendix pilot for distance-based formulation as empirical corroboration of claim 3?** My vote: no, skip. The analytical claim is clean; the pilot would be a week of work for a reviewer who insists on empirical confirmation, and that cost is better spent after the paper is on arXiv if anyone actually raises the question.
