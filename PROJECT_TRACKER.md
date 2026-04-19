# Autoresearch / trajectory-topography — Project Tracker

One-page table of contents for the research line on this branch. Read this first; dive into the individual reports for detail. Updated at the end of each numbered experiment.

**Branch:** `autoresearch/trajectory-topography`
**Last update:** 2026-04-19
**Latest commit touching research:** see `git log` — this file lags commit history between updates.
**Plan:** [`docs/autoresearch_plan.md`](docs/autoresearch_plan.md)

---

## Experiment status

| exp | status | report | headline finding |
|---|---|---|---|
| 0 | ✅ complete | [`reports/exp0/report.md`](reports/exp0/report.md) | Baseline reproduces prior NeuroGen long-horizon eval (0.7943 vs 0.7980), pipeline validated. Displacement PCA spectrum flat — resolved as init-isotropy artifact (§10.5), not absence of structure. |
| 1 | ✅ complete | [`reports/exp1/report.md`](reports/exp1/report.md) | Phase-structured representation formation. Digit cluster coherence peaks at step 47K (+0.49) then loosens; simultaneously an ordinal axis emerges (r(cos, \|i−j\|): +0.28 → −0.39). Two-phase learning: coarse category then within-category refinement. |
| 2 | 🚧 designing | — | Three runs (baseline / topographic 2D / random-permutation control) at 90K steps. Prediction: topographic grid weakens Phase-2 within-category axis. |
| 3 | ⏸ deferred | — | Compositional eval. Digit-arithmetic axis ruled out by corpus check. Character-class primary, number-word composition thin but viable sub-axis. |
| 4 | ⏸ after Exp 2 | — | Trajectory-geometry features correlated with compositional performance. Peak-then-decline and anchor-driven convergence formalized here. |
| 5 | ⏸ gated on Exp 2+4 | — | Trajectory-conditioned prediction. Only if Exp 2 and Exp 4 show positive signals. |

---

## Methodological principles established

| principle | origin | memory |
|---|---|---|
| Positions, not displacements, for trajectory analyses. Δ = structured − random_init has variance dominated by init noise. | Exp 0 §10.5 + §11 | `feedback_trajectory_methodology.md` |
| Null baseline for displacement analyses: fresh `N(0, σ²)` sampling, not shuffling observed `w_0`. | Exp 0 §11 (refined) | `feedback_trajectory_methodology.md` |
| Default dense-early snapshot cadence (every 500 for first 10K, every 1K after); most formation happens early. | Exp 0 §5 + Exp 1 §1 | `feedback_snapshot_cadence.md` |
| ±0.02–0.025 val_bpb is minimum detectable difference for cross-run comparisons. Don't write conclusions on smaller deltas. | Exp 0 §4 | `project_autoresearch_tracking.md` |
| Notebook cadence: full reports end-of-experiment only; 2-3 sentence daily entries between. | user guidance | `feedback_notebook_cadence.md` |
| Scope honesty: byte-level tests orthographic composition, not semantic (SCAN/COGS sense). Lead writeup with "cleaner test, weaker claim." | Exp 0 §13 | `feedback_exp3_scope_honesty.md` |
| **Sanity-check the substrate before writing up a clean finding.** Caught two near-overclaims in this project (flat-spectrum artifact; digit-arithmetic corpus). | Exp 0 §10.5 + Exp 1 §13.2 | `feedback_substrate_check.md` |

---

## Named case studies — master tracker

Master record: [`reports/exp0/report.md` §12](reports/exp0/report.md). This table summarizes last-known state.

| case | signature | Exp 1 resolution | open question |
|---|---|---|---|
| `.!?` sentence-punct triad | cos 0.47–0.62 in `w_L`; Δ cos = −0.11 (different starts, shared destination) | Anchor-driven sequential convergence: `.` @ step 3000, `!` @ 12000, `?` @ 16000. `.` is anchor (most frequent, most distinctive context). | Does anchor-ness correlate with corpus frequency across 3-4 additional triads? (Exp 4) |
| digit cluster `0-9` | +0.44 within-class cosine; NN pairs up to +0.67 | Peak-then-decline: coherence +0.49 @ 47K → +0.44 @ 100K. Ordinal r strengthens after peak (+0.28 → −0.39). **Mechanism corrected post-corpus-check: context-adjacency ("N-year-old") not arithmetic.** | Do monotone-growth classes eventually show their own Phase 2 at longer training? (deferred) |
| `"` ↔ `\x93` mixed-encoding | cos +0.80 in `w_L`; CP1252/UTF-8 quote equivalence learned via co-occurrence | **Deep-dive deferred.** | Does alignment form early (byte co-occurrence) or late (contextual role)? |
| `()` pair (candidate) | peak +0.30 @ step 32K, final +0.06 (−80% decline) — cleanest role-specialization | Recognized as second peak-then-decline mode (distinct from ordinal refinement). | Does role specialization generalize to other opener-closer pairs? |

---

## Corpus substrate facts (carried forward)

These are load-bearing for downstream experiment design. Reference: `analysis/exp1_trajectories/digit_corpus_frequency.json` and `number_word_feasibility.json`.

- **TinyStories training corpus** ≈ 1.9B bytes, byte-level vocab 256.
- **Digit bytes**: 0.0035% of corpus. `3` = 73% of all digits. Zero `\d+\d` or `\d=\d` patterns. Digit cluster reflects "N-year-old" age template, not arithmetic.
- **Number words** (one–ten): 264K total occurrences. `one` = 83% "one day" (time marker), `three` = 77% "three year(s)" (age). `two` is most diverse. `four`–`ten` rare.
- **Viable compositional sub-axis for Exp 3:** `{four, six, eight} × {legs, blocks, wheels, pieces}` — animal/object counting. Thin but real. 5-6 non-zero cells.
- **Primary Exp 3 axis:** character-class composition (unchanged). Cheap, valid, well-matched to byte-level model.

---

## Exp 2 — design state (pre-kickoff)

Design constraints consolidated from Exp 0 + Exp 1 + corpus checks. Full rationale in the respective report sections.

- **Step budget:** 90K. Memorization past ~85K (Exp 0 §5).
- **Schedule:** reuse Exp 0 — cosine LR warmup 200 / max 2e-3 / min 2e-4. Snapshot cadence identical.
- **Infrastructure:** reuse `autoresearch/exp0_train.py`, extend with topographic-loss and random-control variants.
- **Three conditions:**
  - baseline (`arch_cfg={}`)
  - topographic (2D grid, co-occurrence-driven, SOM-style anneal)
  - random-permutation control (same topographic loss with shuffled targets)
- **Primary metric:** best val_bpb over run (not final; Exp 0 §5).
- **Secondary metric:** digit ordering correlation `r(cos, |i−j|)` at step 90K in all three conditions. Prediction: topographic grid weakens the within-category axis — context-adjacency refinement constrained by 2D grid (Exp 0 §12.2, Exp 1 §9; mechanism relabeled post-corpus-check from "arithmetic" to "context-adjacency").
- **Topographic-layout diagnostics:** common-byte subset only. Rare bytes unreliable (Exp 1 §6 + §8).
- **MDD for cross-condition comparison:** ±0.02–0.025 val_bpb. Smaller deltas = noise.

Open design questions to settle before kickoff:
- Topographic grid size: plan says 16×16; verify this against common-byte subset size.
- Topographic regularizer weight ramp schedule.
- Co-occurrence window size for topographic targets (plan: 5 tokens).

---

## Deferred follow-ups (non-blocking)

- Mixed-encoding `"` ↔ `\x93` trajectory deep dive (Exp 1 §3).
- Monotone-classes-eventually-decline test at 200-300K steps (Exp 1 §4).
- Anchor-frequency correlation on additional triads (Exp 1 §10).
- Redefine stabilization metric with sustained-low-motion criterion (Exp 1 §8).
- Number-word composition axis: confirm `{four, six, eight} × {legs, blocks, ...}` cells are sufficient for a held-out test, or fall back to character-class only (this session).

---

## How to use this document

- **Starting a new session:** read this file first, then the specific report you need.
- **End of each numbered experiment:** update the experiment-status row, pin any new numbers to the case studies, append new methodological principles.
- **This file is the index, not the record.** Details live in `reports/exp{N}/report.md`. Keep entries here to one line where possible.
