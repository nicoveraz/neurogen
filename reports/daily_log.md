# Autoresearch daily log

Short (2-3 sentence) tactical entries for work between numbered-experiment
reports. For per-experiment results and reasoning, see `reports/exp{N}/report.md`.

Format: `## YYYY-MM-DD` heading per session; under it, one or more short
bullets. If something deserves more than three sentences, it belongs in the
next numbered-experiment report, not here.

---

## 2026-04-18

- Exp 0 complete (5h52m, val_bpb 0.7943, reproduces prior long-horizon baseline). Full report in `reports/exp0/report.md`; interpretation diagnostics ruled out regime C (noise), confirmed high-dimensional structure with strong within-class clustering in final embeddings.
- Exp 0 report extended with methodological principle (§11: use positions, not displacements), named case studies (§12: `.!?` triad, digit cluster, `"`/`\x93` mixed-encoding gem), and Exp 3 scope-honesty caveat (§13: orthographic not semantic composition at byte level; digit-arithmetic axis is the one lever toward the original motivation).
- Next: Exp 1 starts with within-class trajectory coherence — per-class mean pairwise cosine of positions over all 111 snapshots, checking when each class forms as a cluster.

### 2026-04-18 — Exp 1 within-class coherence results

Ran first Exp 1 analysis (`analysis/exp1_trajectories/within_class_coherence.py`; plot at `reports/exp1/within_class_coherence.png`). Four surprising findings vs the "digits form fast, vowels slow" prior:

- **Sentence-punct `.!?` forms fastest** among real classes (formation step 3000), not digits — likely because sentence-boundary bytes have uniquely distinctive distributional context. Peak within-class cosine +0.51, still growing at step 100K. This anchors the §12 named-case-study finding: the `.!?` convergence from random starts completes by step ~3000.
- **Peak-then-decline** for digits (+0.49 @ 47K → +0.44 @ 100K) and clause_punct (+0.40 @ 38K → +0.33 @ 100K). Classes form as crude groupings, then partially de-cluster as later training specializes within-class roles. Phase-transition signature; worth a dedicated follow-up in Exp 1.
- **Brackets never form a cluster** (peak cos +0.05, barely above random). Exposes a bad class definition: `()[]{}` are not within-class similar — openers vs closers have disjoint semantic roles, and `(`/`[`/`{` differ from each other too. Useful negative: our class grouping encoded a false assumption.
- **Most class formation in first 10K steps.** Retroactive validation of the dense-cadence (500-step) snapshot decision.

Vowels (step 4500) do form before consonants (step 6000) as predicted. Digits and uppercase tie at step 6500; lowercase at 8000; whitespace slow (28K, likely because `\t`/`\r` are rare in TinyStories).

Next: pairwise CKA between consecutive snapshots (trajectory smoothness), per-token movement-magnitude curves, and the `.!?` triad case-study deep dive.

### 2026-04-18 (later) — Exp 1 complete

All Exp 1 passes done; full report at `reports/exp1/report.md`. Headline: digit peak-then-decline is phase-structured learning — coarse category (phase 1) then ordinal refinement (phase 2). `r(cos, |i−j|)` for digits goes from +0.28 @ init → +0.06 @ 10K → −0.29 @ 20K → −0.39 @ 70K → −0.35 @ 100K; steepest ordinal emergence happens *after* coherence peak. `.!?` convergence is anchor-driven and sequential — `.` @ 3000, `!` @ 12000, `?` @ 16000 (user prediction confirmed crisply). Bracket subclass finding is data sparsity, not just class definition — `[]` and `{}` are too rare in TinyStories. Consecutive-snapshot CKA minimum at step 20K (fastest reorganization); CKA(t, init) and CKA(t, final) cross near step 22K. Proceed to Exp 2 with diagnostic addition: report digit ordering r in all Exp 2 conditions, predict topographic 2D grid trades Phase-1 coherence for Phase-2 ordinal structure.

## 2026-04-19

- Retroactive refinements to Exp 0 report applied: §11 null-baseline preference (fresh `N(0, 0.8²)` sampling over shuffle); §12 converted to cross-experiment master tracker with Exp 1 numbers pinned per case study; §13 reframed with cleaner-test-weaker-claim positive lead + digit-arithmetic corpus check.
- **Corpus check killed the digit-arithmetic axis for Exp 3.** 200M-byte scan: digits are 0.0035% of corpus, `3` alone is 73% of all digit bytes, virtually all digit-`3` occurrences are the "N-year-old" template, zero `\d + \d` or `\d = \d` patterns. The Exp 0 digit cluster reflects age/count context similarity, not arithmetic. Exp 1 §4 mechanism interpretation corrected (the two-phase story stands; the ordinal axis is context-adjacency, not arithmetic).
- Exp 3 revised axis options: character-class boundaries (primary), number-word composition (new candidate replacing digit-arithmetic — 25× more frequent than digits), formulaic-template violation (fallback).
- Memories updated: trajectory methodology (prefer fresh sampling over shuffle), Exp 3 scope (digit-arithmetic ruled out; corpus-frequency check is required before committing any compositional axis), project tracker (§12 master-tracker pointer; Exp 2 design constraints consolidated).
- Next: Exp 2 scaffold. Three training runs (baseline, topographic, random-control) at 90K steps each on the same schedule as Exp 0. Reuse `autoresearch/exp0_train.py` infrastructure.

## 2026-04-19 (later)

- Added `PROJECT_TRACKER.md` at branch root — one-page table of contents for the research line. Lists experiment status, methodological principles, named case studies, corpus-substrate facts, Exp 2 design state, deferred follow-ups. Read-first document for new sessions.
- Named the "substrate-check before writeup" habit as a feedback memory (`feedback_substrate_check.md`). Pattern has caught two near-overclaims in this project (§10.5 flat-spectrum resolution; digit-arithmetic corpus kill).
- **Number-word corpus feasibility check: mostly negative.** `one` is 75% "one day" (time-marker), `three` is 77% "three year(s)" (same age template as digit `3`), `four`–`ten` are rare (<2K each). Top shared-context slots are pronouns (`she`, `he`, `they`), not compositional nouns. The only viable thin compositional sub-axis is `{four, six, eight} × {legs, blocks, wheels, pieces}` — animal/object counting with 5-6 non-zero cells. Reinforces: Exp 3 primary axis stays character-class; number-word composition available only as thin supplement.
- Exp 2 digit-ordering prediction reworded in both Exp 0 §12.2 and Exp 1 §9 — Phase-2 mechanism relabelled from "arithmetic / ordinal refinement" to "context-adjacency refinement". Geometric prediction unchanged.
- Next: Exp 2 scaffold.

## 2026-04-19 (late night)

- Exp 2 scaffold complete (`autoresearch/exp2_train.py` + `precompute_cooccur.py`). Co-occurrence window=5 computed on full 1.9B-byte corpus. **Matrix domination check caught raw-frequency concentration** — max/median 3.7M, top-1% of pairs = 57% of mass; switched default to log-normalized (max/median 4.3, top-1% = 3%).
- Three pilots run to calibrate grid learning rate:
  - **Pilot 1** (grid_lr_scale=10, w=0.1): topo σ=1 loss saturated at −1.0 by step ~500. Too fast — Adam normalization eats the weight coefficient, grid LR is the dominant knob.
  - **Pilot 2A** (scale=0.1, w=0.1): σ=1 loss only −0.05 at step 10K, grid spread 0.94. Too slow.
  - **RNG-matched baseline** (w=0, same code path): val_bpb @ 10K = 0.9511, **identical to pilot 2A's 0.9533**. Both ~0.055 below Exp 0's 1.007.
- **Big methodological finding: MPS non-determinism is ~0.055 val_bpb at 10K steps** — as large as the "substantial" band of the Exp 0 §4 calibration table. The 0.055 "improvement" in all three Exp 2 pilots over Exp 0 is NOT topographic effect; it's run-to-run variance from MPS floating-point non-determinism. Matched-null (same code path, weight=0) is the correct baseline for Exp 2.
- New feedback memory `feedback_matched_null.md` names the principle. PROJECT_TRACKER updated with the principle. Exp 0 §4.1 adds the cross-codepath asterisk to the calibration table.
- **Pilot 2B** (scale=1, midpoint of log-space triangulation) running. Expected healthy target: σ=1 loss −0.3 to −0.7 at step 10K, spread 0.6-0.8. Will report trajectory (not just endpoints) + LM confirmation against matched null.

## 2026-04-19 (late late)

- Pilots 2B (Gaussian scale=1) and 2C-MSE and 2D (equilibrium-MSE with fixed σ=3) all revealed distinct pathologies. Pilot 2D still running for the record.
- **Theoretical finding unified across all three formulations:** any topographic regularization loss that factors through a Gaussian kernel has vanishing gradient at both d→0 and d→∞; stable equilibria are stationary points but not attractors. Written up in full at `reports/exp2/gradient_vanishing_analysis.md`. Memory at `feedback_topo_loss_equilibrium.md` updated with the generalized claim.
- New feedback memory `feedback_pilot_loop_momentum.md`: after three distinct pilot failure modes, step back and analyze rather than launching the fourth pilot. The pattern "each pilot fixes the previous pilot's pathology but introduces a new one" is the signal.
- PROJECT_TRACKER updated: Exp 2 status set to "paused pending design review" with three path options (A: distance-based formulation; B: initialization-only topography; C: skip Exp 2, write up Exp 0 + Exp 1 + gradient-vanishing finding as a first paper). **Not making the decision today.** Sitting with it.
- Letting pilot 2D finish for the completeness of the record but not launching further pilots until the step-back is resolved.

## 2026-04-20

- User chose Option C path (consolidate Exp 0 + Exp 1 + gradient-vanishing analysis). Decisions: title B ("Trajectories Tell What Endpoints Don't…"), arXiv + Substack venue, run 3-triad verification before drafting (my rec), descriptive wording on claim 3, skip distance-based pilot.
- Anchor-pattern verification on additional triads: scope-narrowed claim. Anchor-driven convergence is *asymmetry-gated*. In 2 triads with high frequency asymmetry (`.!?` at 30×, `,;:` at 978×), anchor matches most-frequent member; time spread 5.3× and 1.3× respectively. In 0/5 digit triads (<4× asymmetry), the pattern holds. Letter triples don't form tight enough clusters for dynamic to apply. More interesting claim than original universal one — specifies *when* the dynamic emerges.
- Three writeup figures produced: triad_asymmetry_comparison (§3.3), pilot_comparison (§4.2), gradient_envelope (§4.3). All in reports/.
- Full Option C draft written: `reports/writeup/draft.md`, ~8500 words, 6 sections + 5 appendices. §1 intro, §2 setup, §3 phase-structured formation (six subsections), §4 gradient-vanishing diagnosis (five subsections), §5 discussion, §6 conclusion, appendices A–E. Draft ready for review.
- **User instruction overnight: test whether quartic-window architecture produces emergent topographic-like embedding organization without explicit loss.** Launched `exp_quartic` training (arch=window_power_4.0, 100K steps, same snapshot cadence as Exp 0). ~6h ETA. Watcher script will auto-run `quartic_vs_baseline.py` comparison when training exits. Results will be in `runs/exp_quartic/analysis.out` by morning.
- Comparison script built (`analysis/exp1_trajectories/quartic_vs_baseline.py`): checks cooccur×cosine correlation, per-class final coherence, digit ordinal trajectory, tracked-pair distances. If quartic shows higher cooccur-cosine correlation than baseline, emergent topographic organization is present.
