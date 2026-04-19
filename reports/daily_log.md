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
