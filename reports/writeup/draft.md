# Trajectories Tell What Endpoints Don't: Small-Scale Representation Dynamics and the Limits of Topographic Regularization

**Draft · 2026-04-19**
**Authors:** [TBD]

---

## Abstract

[To be drafted after §3–§5 are complete.]

---

## 1. Introduction

[Placeholder — drafted after §3 and §4.]

---

## 2. Setup

[Placeholder — drafted after §3 and §4.]

---

## 3. Phase-structured representation formation

The endpoint of training shows token embeddings arranged into recognizable classes: digits cluster with digits, sentence punctuation clusters with sentence punctuation, letters loosely cluster by case. This is the state the model has learned. What the endpoint does not show is *how* the state was reached — whether the organization arose monotonically, whether different classes organized at different times or at the same time, whether the final organization passed through intermediate configurations that were later refined away, and whether specific members of a class played distinguishable roles in the class's formation.

Trajectory analysis surfaces these features. By sampling the token embedding matrix at regular checkpoints throughout training — every 500 steps for the first 10K steps and every 1K steps thereafter — and analyzing the resulting sequence of embedding states, phenomena invisible to endpoint analysis become legible. This section presents six findings from that analysis. The headline finding (§3.2) is that at least two byte classes go through a two-stage formation dynamic: a coarse category forms, then a within-category axis emerges, partially loosening the crude category in favor of a more-informative gradient. This is the Saxe-McClelland hierarchical-feature-emergence story [@saxe2019], observed on byte classes small enough to see the whole dynamic unfold in a 10×10 matrix.

All analyses in this section operate on the 111 token-embedding snapshots saved during the 100K-step Exp 0 training run (3.4M-parameter byte-level transformer on TinyStories). Per the methodological principle established in §2, analyses use *positions* `w_t` at each checkpoint rather than displacements `w_t − w_0`, to avoid the variance contamination from isotropic initialization that would otherwise dominate PCA and related analyses.

### 3.1 Within-class trajectory coherence (headline)

For each of ten byte classes with orthographic motivation (digits, sentence-punct `.!?`, clause-punct `,;:`, uppercase letters, lowercase letters, vowels `aeiou`, consonants, whitespace, quotes, brackets), we compute the mean pairwise cosine similarity among member token positions at each snapshot. The resulting traces characterize when each class forms as a cluster (coherence rises above the random-pair baseline) and what the final coherence magnitude is.

**Figure 1** (`reports/exp1/headline_coherence.png`) shows all ten traces on a symlog x-axis. Two qualitative regimes are visible:

- **Monotone growth**: sentence-punct, uppercase letters, lowercase letters, vowels, consonants. These classes' coherence rises and stays rising through 100K steps. Final within-class cosine: +0.50 (sentence-punct), +0.21 (vowels), +0.19 (uppercase), +0.13 (consonants), +0.12 (lowercase). Sentence-punct is the fastest to form (formation step 3000 — first step where within-class cos exceeds random + 0.05 sustainedly) and the tightest cluster at end of training.
- **Peak-then-decline**: digits, clause-punct `,;:`, and the `()` bracket pair when isolated (see §3.4). These classes reach a coherence peak mid-training, then partially de-cluster over subsequent training steps. Digit coherence peaks at +0.49 at step 47K and declines to +0.44 by step 100K. Clause-punct peaks at +0.40 at step 38K and declines to +0.33.

Formation order is not what the naive "frequent things form first" prior predicts. Sentence-punct forms fastest despite being less frequent than individual letters; the speed reflects distributional distinctiveness (sentence-end contexts are uniquely constrained) rather than raw frequency. Vowels (step 4500) form before consonants (step 6000) because there are fewer of them (five vs. twenty-one) and each vowel is correspondingly more distinctive. Whitespace forms slowly (step 28000) because the class definition grouped bytes ` \t\n\r` together, but `\t` and `\r` are extremely rare in TinyStories, so the class has effectively two members and clusters poorly.

The peak-then-decline pattern is not standard in representation-learning descriptions, which typically model feature formation as monotonic. It is the most important finding of this section and is characterized in detail next.

### 3.2 Digit class — phase-structured learning

The digit class offers the cleanest instance of the peak-then-decline pattern. Coherence peaks at +0.49 at step 47K and declines to +0.44 at step 100K — a ~10% relative reduction after peak. The question is whether the decline is noise, structural refinement, or something else.

To test, we compute the Pearson correlation between each digit pair's cosine similarity and their *numerical* distance |i − j| at each snapshot. If the decline is refinement along a numerical-ordering axis — digits close in numeric value becoming more similar in embedding space than digits distant in numeric value — we expect this correlation to strengthen (become more negative) over training, especially after the coherence peak.

**Figure 2a** (`reports/exp1/digit_pairwise_peak_vs_final.png`) shows the 10×10 pairwise cosine matrices at peak (step 47K) and final (step 100K) alongside their difference. At peak, most off-diagonal entries cluster around +0.45–0.60, with relatively weak ordering structure. At final, the same pairs show a visible gradient: numerically adjacent digits (e.g., 4↔5, 5↔6) retain cosine ~0.65, while numerically distant digits (e.g., 0↔9, 1↔8) drop to ~0.25.

**Figure 2b** (`reports/exp1/digit_ordering_correlation.png`) plots the correlation trajectory. The result:

| step | mean digit cosine | Pearson `r(cos, |i−j|)` |
|---:|---:|---:|
| 0 | +0.01 | +0.28 |
| 10K | +0.10 | +0.06 |
| 20K | +0.31 | **−0.29** |
| 47K (peak) | +0.49 | −0.30 |
| 70K | +0.41 | **−0.39** (most negative) |
| 100K | +0.44 | −0.35 |

The trajectory has three phases:
1. **Steps 0–20K — coarse category formation.** Mean digit cosine rises from +0.01 to +0.31 as digits pull together as a crude "these are digits" group. The ordering correlation is near zero or positive during this phase: the category forms without internal structure.
2. **Steps 20K–70K — ordinal refinement.** Mean digit cosine continues rising to peak (+0.49 at step 47K) while the ordering correlation turns sharply negative (−0.29 at step 20K → −0.39 at step 70K). The model is simultaneously tightening the category and learning a within-category axis.
3. **Steps 70K–100K — stabilization with mild loosening.** Mean cosine drops from peak +0.49 to +0.44; ordering correlation stabilizes around −0.35. The category's decline (the 10% loosening of the mean cosine) reflects specifically that numerically-distant pairs have moved apart. Numerically-adjacent pairs remain close. Information has been gained, not lost.

This is phase-structured representation formation in a canonical form: a coarse feature (category membership) forms first, and a finer feature (within-category ordinal position) emerges later. The endpoint of training shows a category that is less tight than at its peak; the trajectory shows that the looseness *carries* information. Endpoint analysis alone cannot distinguish "the category is degrading" from "the category is refining along an informative axis."

#### Mechanism correction

An initial interpretation of the ordinal axis — that the model is learning *arithmetic* relations among digits — is not supported by the corpus. A frequency check on 200M bytes of the TinyStories training corpus shows:

- Digit bytes are 0.0035% of the corpus (7,096 in the 200M-byte sample).
- Digit `3` alone accounts for 73% of all digit occurrences.
- Virtually every digit-`3` occurrence sits in the formulaic "`N-year-old`" template (`"She was 3 years old"`, `"a 3 year old boy"`).
- The regex `\d\s*\+\s*\d` matches **zero** times in the sample; `\d\s*=\s*\d` matches zero times; the literal token `" + "` matches zero times.

TinyStories contains essentially no arithmetic syntax. The ordinal axis the model learned is therefore best interpreted as *age/count context adjacency*: digits that appear in similar numeric-quantity contexts (`"3 years"`, `"4 years"`, `"5 years"`) share distributional neighborhoods, and the model organizes their embeddings by age adjacency rather than by arithmetic composition. The two-phase story stands; the label on the within-category axis changes from "arithmetic refinement" to "context-adjacency refinement".

We flag this correction to emphasize that phase-structured finding is mechanism-agnostic: the model learned *some* within-category axis, and the Exp 1 trajectory analysis detects it; identifying the nature of that axis required a separate corpus check.

### 3.3 Anchor-driven convergence, conditional on asymmetry

A second trajectory motif is visible in how tightly-clustered triads of tokens reach their final mutual neighborhood. In the sentence-punct triad `.`, `!`, `?`, which in the final embedding sits at mutual cosine 0.47–0.62, convergence is *sequential*:

- `.` reaches cosine ≥0.6 with the final-triad centroid at step 3000.
- `!` crosses the same threshold at step 12000.
- `?` crosses at step 16000.

The spread between the fastest and slowest member is 5.3×. The pairwise cosine crossings confirm the ordering: `.–!` reaches 0.3 at step 6500 (the earliest pair), `.–?` at 9000, and `!–?` at 9500 (last). `.` acts as an anchor — it arrives at the triad's final region early, and the other two migrate toward it.

The predicted anchor mechanism is a correspondence between *initial* cosine with the final centroid and corpus frequency: the most-frequent member starts closest to the final centroid (because a randomly-initialized embedding receives gradient from all co-occurring contexts, and the most frequent member's gradient signal is strongest early on), so it reaches the centroid region first, and others follow. In the `.!?` triad, `.` is the most-frequent member by a factor of roughly 30× over `?`; its initial-centroid cosine is +0.55, compared with +0.17 and +0.18 for `!` and `?`; and it is predicted-and-verified the anchor.

**Figure 3** (`reports/exp1/triad_asymmetry_comparison.png`) compares four triads to test whether this dynamic generalizes.

| triad | frequency asymmetry (most÷least) | final min pair cosine | anchor-matches-most-frequent? | time spread (threshold-crossing) |
|---|---|---|---|---|
| `.!?` | 30× | 0.42 | ✓ | 5.3× |
| `,;:` | 978× | 0.33 | ✓ | 1.3× |
| `{5, 6, 8}` | 1.6× | 0.60 | ✗ | 1.2× |
| `{4, 5, 8}` | 4.0× | 0.57 | ✗ | 1.4× |
| `{5, 7, 8}` | 4.0× | 0.53 | ✗ | 2.2× (anchor is least-frequent) |
| `{e, s, d}` | 2.3× | 0.30 | ✗ | n/a (never reaches 0.6) |

The anchor-driven pattern appears in two of the six triads: `.!?` and `,;:`. Both have large frequency asymmetry among members (≥30×) and form tight final clusters. In the digit triples, where frequency asymmetry among members is modest (<4×), convergence is simultaneous and no member-acts-as-anchor correspondence holds — in 0 of 5 digit triads examined does the highest-init-centroid member coincide with the most-frequent member. The `{e, s, d}` letter triple doesn't form a tight enough final cluster for the dynamic to be defined (pair cosines remain <0.30).

**Scoped claim.** Anchor-driven convergence is a dynamic gated by two conditions: the triad forms a tight final cluster (min pairwise cos ≳0.4), and one member has strong corpus-frequency asymmetry over the others (≳10×). Under both conditions, the most-frequent member acts as anchor and others migrate toward it with a measurable time spread in threshold crossings. In symmetric clusters like digit triples, convergence is simultaneous. The dynamic is thus *asymmetry-driven* rather than universal; where frequencies are comparable, no clear anchor emerges.

The clause-punct case (`,;:`, asymmetry 978×, anchor `,` matched) confirms the dynamic at much higher asymmetry than `.!?` but with a smaller time spread (1.3×). Time spread appears to depend on factors beyond raw asymmetry — plausibly the distinctiveness of the anchor's distributional context vs. its migrating neighbors — though we do not attempt to characterize those factors here.

### 3.4 Two modes of peak-then-decline

The peak-then-decline pattern observed in §3.2 for digits is not unique. Clause-punct `,;:` shows it (peak +0.40 at step 38K, final +0.33). So does the `()` paren pair when tracked in isolation (peak +0.30 at step 32K, final +0.06 — a dramatic −80% decline).

Comparison of these three cases reveals that peak-then-decline is not a single dynamic; it is two distinct modes distinguishable by endpoint geometry and by whether the loosening carries interpretable structure.

- **Ordinal refinement (digits, clause-punct).** The cluster partially loosens as a continuous within-category axis emerges. Final members still cluster, just less tightly than at peak, with the looseness carrying information (numerical adjacency for digits; plausibly clause-type distinction for `,;:`). Relative peak-to-final decline is modest (10–17%).
- **Role specialization (`()` pair).** The cluster loosens *dramatically* toward anti-clustering. Final cosine drops 80% from peak. The asymptotic state is effectively that the two members have specialized into opposing roles (opening vs. closing), which are semantically *anti-correlated*, not merely "refined categories of a shared type". Openers `(` attach to what follows; closers `)` attach to what precedes; their distributional contexts are near-mirror-images. The cluster forms crudely when the model first discovers "these characters are bracket-like" and then decomposes as the opening/closing distinction emerges.

**Figure 4** (`reports/exp1/bracket_subclass_coherence.png`) shows the bracket-class subclass split that led to this diagnosis: combined bracket class `()[]{}` never forms a cluster (peak +0.05); openers `([{` weakly cluster (final +0.13); closers `)]}` drift negative (final −0.05); the `()` pair alone shows its peak-then-decline signature. The combined-class failure had two causes — the opener/closer opposition (the role-specialization mode applied to the 6-byte grouping), plus data sparsity (`[]` and `{}` are vanishingly rare in TinyStories, so `[`, `]`, `{`, `}` never acquire stable enough representations to cluster with anything).

The two-mode distinction has implications for trajectory analysis: peak-then-decline classes are worth separating by whether the post-peak loosening follows an interpretable continuous axis (as in digits) or diverges into opposing subclasses (as in `()`). Both are signals that the model is learning within-category structure, but the structure is qualitatively different in the two cases.

### 3.5 Trajectory smoothness

We measure trajectory smoothness via linear centered kernel alignment (linear CKA) between consecutive snapshots. **Figure 5** (`reports/exp1/consecutive_cka.png`) plots three traces: CKA(w_{t−1}, w_t) (consecutive smoothness); CKA(w_t, w_final); CKA(w_t, w_0).

Two findings:

- **Consecutive CKA is very high throughout (≥0.98).** There are no discontinuous reorganizations. The embedding matrix evolves smoothly across the entire training run. This rules out concerns that trajectory-geometry analyses of this run would fail numerically at sharp transitions.
- **Consecutive CKA has a minimum at step 20K** (0.983, down from 0.99 elsewhere). This is the step of fastest *relative* reorganization. It coincides with (a) the transition from Phase 1 to Phase 2 in the digit dynamics (§3.2) and (b) the step at which CKA(w_t, w_init) crosses CKA(w_t, w_final) — i.e., the representational halfway point of the trajectory. The halfway point sits at 22% of training steps (step 22000 of 100000), substantially earlier than the temporal halfway, consistent with §3.1's observation that most class formation happens in the first 10K–20K steps.

The consecutive-CKA minimum at step 20K is plausibly the population-level signature of Phase 1 → Phase 2 transition across many classes, not just digits. We do not test this rigorously, but the coincidence is suggestive and worth noting.

### 3.6 Per-token velocity: common vs. rare bytes (supplementary)

A secondary displacement-based analysis (**Figure 6**, `reports/exp1/token_movement.png`) compares per-step velocity `‖w_{t+1} − w_t‖` for common byte classes (letters, digits, common punct, space, newline) against rare classes (high UTF-8 bytes, rare control characters). The finding inverts the naive prior:

- **Median "stabilization step"** (first step after which per-step velocity stays below 20% of that token's peak velocity): common bytes 61000, rare bytes **24000**. Rare bytes stabilize *earlier*.
- **Mean total path length**: common bytes 27.5, rare bytes 19.8. Common bytes move farther overall.

Read jointly: rare bytes have brief high-velocity bursts (when they happen to appear in a batch) followed by long dormancy, producing an early "stabilization" under the peak-relative definition despite receiving very little total training signal. Common bytes have sustained moderate velocity throughout training, consistent with continuous refinement.

The stabilization metric as defined is biased by this spike-then-silence dynamic for rare tokens; a redefined metric that counts sustained low-motion (rather than peak-relative) would avoid the artifact. We flag this as a methodological note for downstream analyses rather than attempting to draw strong claims from the velocity data itself.

---

## 4. [Next to draft] — Topographic regularization attempt

[Placeholder. Will draw on `reports/exp2/gradient_vanishing_analysis.md`, pilot summary table, and the two §4 figures.]

---

## 5. [Next to draft] — Discussion

[Placeholder.]

---

## 6. Conclusion

[Placeholder.]

## Appendices

[Placeholders.]

---

## Reference placeholder

- [@saxe2019] — Saxe, McClelland, Ganguli. *A mathematical theory of semantic development in deep neural networks*. PNAS 116(23), 2019. To cite for the hierarchical-feature-emergence theoretical frame in §3.2.
- Additional references TBD — Elhage et al. on superposition, Kohonen on SOM for §4.4.
