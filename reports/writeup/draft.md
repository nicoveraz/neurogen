# Trajectories Tell What Endpoints Don't: Small-Scale Representation Dynamics and the Limits of Topographic Regularization

**Draft · 2026-04-19**
**Authors:** [TBD]

---

## Abstract

[To be drafted after §3–§5 are complete.]

---

## 1. Introduction

The endpoint of neural network training displays the organized state that the network has learned: tokens grouped into clusters, features distributed across axes, categories recognizable in embedding space. A separate question, often treated as a matter of curiosity rather than a primary research target, is *how* that organization came about — whether classes formed in a particular order, whether the endpoint state was arrived at monotonically or through an intermediate configuration that was later refined away, whether individual members of a class played distinguishable roles in its formation.

This paper argues that trajectory analysis — inspection of representations at regular checkpoints throughout training, rather than only at the end — surfaces phenomena invisible to endpoint analysis, and that at the scale of a small transformer on a simple corpus, the phenomena are both legible and interpretable. We present three findings from a 3.4M-parameter byte-level transformer trained for 100K steps on TinyStories.

**First**, at least two byte classes (digits, clause-punctuation) go through a two-stage formation dynamic: a coarse category forms first, then a within-category axis emerges that partially loosens the crude category in favor of a more-informative gradient. For the digit class, the emergent axis is ordinal — digits close in numeric value become more similar in embedding space than digits far in numeric value — and the correlation strengthens in the *second* phase of training, specifically after the coarse category's coherence has peaked. This is the hierarchical-feature-emergence pattern from the deep-network-theory literature [@saxe2019], observed on a class small enough to see the whole structure in a 10×10 matrix. Endpoint analysis cannot distinguish "the category is degrading" from "the category is refining along an informative axis"; the trajectory shows that the latter is what happens.

**Second**, in certain triads of tightly-clustered tokens, convergence proceeds in an anchor-driven sequential dynamic: one member (identified by high initial cosine with the final centroid) reaches its final region markedly earlier than the others, and the remaining members migrate toward it. The canonical example is the sentence-punctuation triad `.!?`, where `.` reaches its final-centroid neighborhood at step 3000 and `!`, `?` cross the same threshold at steps 12000 and 16000 — a 5.3× sequential separation. A verification pass on additional triads reveals that the dynamic is *conditional on frequency asymmetry*: in triads where one member is ≳10× more frequent than the others (like `.!?` or `,;:`), the asymmetry-anchor dynamic appears; in symmetric triads (like digit triples `{5,6,8}` where all members have comparable frequency), convergence is simultaneous. This is a more informative claim than a universal one: it says *when* the dynamic emerges, which is a falsifiable and specific property of small-transformer learning.

**Third**, we report a negative experimental result that is informative as a methodological finding. We attempted to impose a topographic organization — forcing co-occurring bytes to be grid-adjacent on a 16×16 learnable grid — via a series of Gaussian-kernel-based regularization losses. All four experimental variants failed; the three distinct pathologies that emerged (collapse to coincidence, escape to infinity, bimodal failure of equilibria) share a common mechanism: under gradient descent on a loss that factors through a Gaussian kernel `K(d) = exp(−d²/2σ²)`, the gradient with respect to positions vanishes at both `d → 0` and `d → ∞`, so stationary points of the loss are not reliably attractors. The property is structural, not calibration-dependent, and it implies that future topographic-regularization work on transformer embeddings should use distance-based rather than kernel-based losses. This finding was not the original target of the experimental program, but characterizing it precisely took roughly as much effort as the trajectory analysis that was the target, and we believe it is useful to the field as an honest negative result with an analytical diagnosis.

**Scope.** Byte-level encoding means this work tests *orthographic* compositional structure (character-class boundaries, within-class gradients) rather than *semantic* compositional structure in the SCAN/COGS sense. We consider byte-level a cleaner test of representation dynamics at small scale — primitives are discrete and enumerable, the full vocabulary is 256 items, and noise from polysemy and context-dependence is low — while acknowledging it is a weaker test of the semantic-compositional claims that motivate related research threads. The methodological observations (phase-structured formation, asymmetry-conditional anchor dynamics, gradient-vanishing pathology) generalize beyond byte-level, but we do not empirically verify that in this paper.

### Roadmap

§2 describes the model, corpus, and four methodological principles that constrain the analyses. §3 presents the trajectory findings: within-class coherence, the digit deep dive, anchor-driven convergence, two modes of peak-then-decline, trajectory smoothness, and a supplementary per-token velocity analysis. §4 describes the topographic-regularization attempt and the gradient-vanishing diagnosis. §5 discusses the findings and their connection to superposition / mechanistic-interpretability literature. §6 concludes. Code and analysis scripts are in the project repository; an appendix collects additional figures, the corpus-characterization data used in the §3.2 mechanism correction, and reproducibility notes.

---

## 2. Setup

### 2.1 Model and training

All experiments use a 3.4M-parameter transformer with the NeuroGen baseline architecture: depth 4, embedding dimension 256, four attention heads, a context length of 256 bytes, and byte-level encoding (vocab size 256). The architecture is a standard decoder-only transformer with rotary position encoding, RMSNorm, and GLU-style MLP blocks; details and training code are in the project repository. Parameter count is small enough that every token embedding can be inspected individually and the full 256×256 pairwise cosine matrix is trivial to compute.

Training is on the TinyStories corpus [@eldan2023], encoded as UTF-8 bytes rather than tokenized with a subword vocabulary. The full training set is 1.9B bytes. Byte-level encoding is a deliberate choice: it makes the vocabulary finite and fully enumerable (the 256 bytes are the model's entire set of "words"), which allows aggregate analyses at the vocabulary level to cover all representational mass without missing rare-token behavior. The trade-off — that byte-level encoding is orthographic rather than semantic — is addressed in the scope caveat at the end of §1.

All training runs use AdamW with cosine learning-rate decay (warmup 200 steps, peak LR 2e-3, final LR 2e-4), batch size 32, weight decay 0.05, and gradient clipping at norm 1.0. The single Exp 0 training run (§3) is 100,000 steps (~5h wall-clock on M1 Pro / MPS); the Exp 2 pilots (§4) are 10,000 steps each. All runs use random seed 42. On MPS hardware, floating-point non-determinism produces run-to-run variance of approximately ±0.055 val_bpb at 10K steps even with identical seeds — a variance larger than several of the effects this paper analyzes. Cross-code-path comparisons therefore require matched-null baselines (see §2.3).

### 2.2 Snapshot cadence for trajectory analysis

During the Exp 0 run, the full token embedding matrix `w_t` is saved as a snapshot every 500 steps for the first 10K steps and every 1000 steps thereafter, producing 111 snapshots over 100K steps. The dense-early cadence was chosen before any findings were available and was validated retroactively: the trajectory analyses in §3 show that most class formation (for classes that form monotonically) occurs in the first 10K–20K steps, so uniform-cadence sampling would have under-resolved the most informative region. The choice generalizes as a prior for future trajectory-analysis experiments: default to dense-early cadence, with sparser sampling after the transient phase.

Full optimizer checkpoints (model weights, optimizer state, RNG state) are also saved at coarser cadence for reproducibility; analyses in §3 use only the embedding snapshots.

### 2.3 Methodological principles

Four principles constrain how the analyses are run and interpreted. We state them here because their application is not always self-evident.

**Positions over displacements.** Many trajectory analyses naturally suggest displacement-based features (total path length, step-to-step velocity, PCA of `w_final − w_init`). These features are interpretable in principle but in practice their variance is dominated by the initialization term, not the learned final state. Concretely: the isotropic Gaussian init produces per-token `w_0` with norm ≈12.6 on our scale, and the learned displacement `w_final − w_init` has comparable norm; variance in the displacement is therefore predominantly variance in the (random) initialization term, not variance in the (structured) final term. PCA of displacement vectors shows a flat spectrum (top PC captures 1.5%, participation ratio 127.6 out of 256) — not because trajectories lack structure, but because the structure sits in the `w_final` term, not in the differences. Analyses throughout this paper therefore operate on `w_t` positions at each snapshot, not on `w_t − w_s` displacements. For displacement-based features we note this explicitly.

**Matched-null comparison.** When an experimental intervention changes parameter allocation, optimizer setup, or RNG consumption relative to a baseline, comparing the intervention to an earlier baseline (trained with a different code path) inherits MPS non-determinism as systematic bias. In §4, adding a 256×2 `grid_pos` parameter and its optimizer group is enough of a change to introduce ~0.055 val_bpb apparent improvement over the Exp 0 baseline *even with `w_topo = 0`* (no gradient flowing through the topographic term). We verified this empirically with an RNG-matched-baseline pilot (same code path, `w_topo = 0`). Consequently §4's comparisons use the matched null as baseline, not Exp 0.

**Substrate-check before interpretation.** When a finding feels clean enough to commit to in a report, we run a cheap sanity check on the substrate the finding assumes. This caught two near-overclaims in this work. The first: the flat-spectrum PCA of displacement vectors, when initially observed, looked like "no trajectory structure" — a finding that would have undercut the premise of trajectory analysis. A substrate check via pairwise cosine in `w_final` revealed the opposite: within-class similarities are strong (+0.44 for digits, +0.50 for sentence-punct in `w_final`); the flat spectrum was an artifact of isotropic init dominating the displacement variance (cf. "positions over displacements" above). The second: the digit phase-structure finding in §3.2 initially suggested the model was learning arithmetic. A corpus check revealed zero arithmetic patterns in the training data (§3.2); the within-category axis is age/count adjacency, not arithmetic. Both corrections narrowed the claims being made; neither substantively changed the headline finding.

**Scope-honest framing.** Byte-level TinyStories is a *cleaner* testbed for representation dynamics than word-level on a larger corpus — primitives are discrete and fully enumerable, combinations are enumerable, context noise is low — but it is a *weaker* test of the semantic-compositional claims that motivate a strand of the representation-learning literature (SCAN, COGS, systematic-generalization). We lead with the cleaner framing because it accurately describes what this work tests and contributes; the scope caveat is noted but does not frame the results.

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

## 4. Limits of topographic regularization: a gradient-vanishing pathology

The trajectory findings in §3 characterize how representations form under standard training. A natural follow-up question is whether imposing spatial structure on token representations — forcing co-occurring bytes to be organized onto a 2D grid in a way that mirrors their co-occurrence matrix — would alter those dynamics. This is a classical topographic-organization hypothesis [@kohonen1982], and it motivates a straightforward experimental design: add a regularization term to the training loss that penalizes arrangements where co-occurring tokens are far apart on a learned 16×16 grid of positions, and compare the resulting learning dynamics to the unregularized baseline.

This section reports four experimental attempts at such a regularizer, all of which failed — but failed in a way that, when analyzed, reveals a structural property of Gaussian-kernel-based topographic losses that generalizes beyond this project. We describe the attempts, the three distinct pathologies observed, and the unified diagnosis that accounts for all three.

### 4.1 Design

Each token in the 256-byte vocabulary is assigned a 2D position `pos_i ∈ R²` on a 16×16 grid, initialized as a shuffled lattice (every cell populated by one token, token-to-cell assignment random). The positions are learnable parameters, trained alongside the model's standard parameters via the same AdamW optimizer but with no weight decay (so the "grid" structure is not artificially collapsed by regularization).

The objective augments the language modeling loss with a topographic term:

```
L_total = L_LM + w_topo · L_topo
```

where `w_topo` is ramped from 0 to a target value over the first 10% of training, and `L_topo` is a function of the current grid positions and a precomputed 256×256 co-occurrence matrix. The matrix is computed once before training by scanning the TinyStories corpus with a window of 5 bytes, log-normalizing the raw counts (to avoid domination by space-letter bigrams, which otherwise carry 57% of the raw co-occurrence mass), and renormalizing to a probability distribution.

Control conditions use the same code path with the co-occurrence matrix permuted (random targets) or with `w_topo = 0` (the "matched null"; see §2). The matched null is necessary rather than comparing to the Exp 0 baseline because MPS non-determinism on this hardware produces ~0.055 val_bpb run-to-run variance at 10K steps — comparable to the "substantial" band of the effect-size calibration in §2. Any cross-code-path comparison therefore needs an explicit matched baseline.

The specific form of `L_topo` is where the formulations differ, and is what produces the distinct pathologies below.

### 4.2 Three formulations, three pathologies

**Formulation A — Gaussian pure attractive.** The most direct import of SOM-style dynamics into gradient-based learning: treat the co-occurrence weights as attraction strengths, and pull co-occurring pairs toward each other with a Gaussian kernel whose bandwidth σ anneals from 8 to 1 over the first 20% of training.

```
L_topo = − Σ_{i<j} cooccur(i, j) · exp(−‖pos_i − pos_j‖² / (2σ²))
```

Minimizing this loss makes co-occurring pairs have high kernel value (small pairwise distance). Three pilots varied the grid learning rate across two orders of magnitude (`grid_lr_scale ∈ {10, 1, 0.1}`, giving Adam LR `∈ {2e-2, 2e-3, 2e-4}`). Result:

- `scale = 10`: collapse to coincidence. Topographic loss saturates at its floor (−1.0, the theoretical minimum for `-E[K(d)]` when cooccur is a probability distribution) by step 500. Grid spread ratio (mean pairwise distance / initial) plateaus at 0.56 and freezes.
- `scale = 1`: same collapse, delayed. Saturation at step 3000, spread ratio 0.56.
- `scale = 0.1`: positions barely move in 10K steps. Spread ratio 0.94, topographic loss −0.05 (vs. floor −1.0). Extrapolating to the full 90K-step training schedule suggests saturation would eventually occur.

Across all three learning rates, the final equilibrium *geometry* is identical (spread 0.56 for the two that reached it), differing only in when saturation occurs. The learning rate determines the *timing* of collapse, not the *endpoint*. This is a structural property of the loss, not a calibration choice.

**Formulation B — MSE against similarity targets, with zero target for non-cooccur pairs.**

To prevent collapse, we replaced the pure-attractive loss with an MSE against a target similarity matrix: high-co-occurrence pairs should have high kernel similarity, low-co-occurrence pairs should have low similarity. Construction: target_sim(i, j) = cap × cooccur_norm(i, j), with cap = 0.9 to prevent the max-cooccurrence pair from targeting full coincidence.

```
L_topo = mean_{i<j} (exp(−‖pos_i − pos_j‖² / (2σ²)) − target_sim(i, j))²
```

Result at grid_lr_scale = 1, 10K steps: positions *expanded*. Grid spread ratio rose to 1.58 (from initial 1.0); mean pairwise distance rose to 13.2 (from initial 8.3); topographic loss decreased by only 39% (0.028 → 0.017) and then froze as gradients vanished.

Mechanism of the failure: the majority of pairs (~53,000 of 65,000) have zero co-occurrence, so their target similarity is 0. Minimizing loss for these pairs drives actual similarity toward 0 — i.e., drives the pair's distance to where `exp(−d²/2σ²) ≈ 0`, which at σ = 1 means d ≫ 3. With ~5× more "should be far" pairs than "should be close" pairs, the aggregate repulsive pressure pushes positions outward until the kernel is near-zero everywhere. Once there, loss gradient also vanishes, and the system freezes in a spread-out non-equilibrium state that encodes little topographic information.

**Formulation C — Equilibrium MSE with non-zero floor target.**

To avoid the expansion, we set the target similarity for zero-co-occurrence pairs to the *expected similarity* at uniformly-random grid placement, rather than to 0. Target construction: `target(i, j) = max(expected_random_sim(σ), cap · cooccur_norm(i, j))`. With σ = 3 fixed (no anneal), expected random similarity is ≈0.161, corresponding to an equilibrium distance of ≈5.73 — approximately the mean pairwise distance in a random placement on a 16×16 grid. High-cooccur pairs keep the higher cap-scaled target; non-cooccur pairs settle at "random-grid separation" rather than escaping to infinity.

Result at grid_lr_scale = 1, 10K steps: *bimodal failure*. High-cooccur pairs collapsed *past* the 1.38 equilibrium target to ≈0.1–0.3 grid units. Low-cooccur pairs escaped *past* the 5.73 equilibrium target to 16+ grid units. Mean pairwise distance ballooned to 16.9. Six tracked pairs:

| pair | measured d at step 10K | equilibrium target | status |
|---|---|---|---|
| ` `–`e` | 0.13 | ≈1.38 | collapsed |
| ` `–`t` | 0.25 | ≈1.38 | collapsed |
| ` `–`a` | 0.09 | ≈1.38 | collapsed |
| `.`–`!` | 1.78 | ≈1.38 | **near target** |
| `0`–`5` | 26.44 | ≈5.73 | escaped (outside nominal grid extent) |
| `(`–`[` | 19.17 | ≈5.73 | escaped |

Only the sentence-punct pair `.`–`!` landed near its equilibrium. All high-cooccur pairs collapsed past equilibrium; all low-cooccur pairs escaped past equilibrium.

Common thread across all three formulations is visible in Figure 7 (`reports/exp2/pilot_comparison.png`), which overlays grid spread, topographic loss magnitude, and fraction-of-positions-moving for each pilot. All three reach a frozen state (fraction moved → 0 by late training), but arrive there via different failure modes: collapse, escape, and bimodal.

### 4.3 Unified diagnosis: vanishing gradient at both tails of a Gaussian kernel

Formulations A, B, and C differ in their target structure and their equilibrium specifications, but share a single property: they all define the loss as a function of `K(d) = exp(−d²/2σ²)`. Writing `L = f(K(d_ij), T_ij)` for an arbitrary function f and target T_ij, the gradient with respect to a position is

```
∂L/∂pos_i = Σ_j  [∂f/∂K] · (∂K/∂d) · (∂d/∂pos_i)
         ∝ Σ_j  [∂f/∂K] · K(d_ij) · (pos_i − pos_j)
```

(absorbing the σ² factor). The gradient magnitude per term has *two* factors that can vanish: `K(d_ij)` and `(pos_i − pos_j)`.

- **d → 0**: `(pos_i − pos_j) → 0`, so the gradient vanishes regardless of what `f` specifies. Positions that coincide have no force pushing them apart, even if `f` says they shouldn't coincide.
- **d → ∞**: `K(d) → 0` exponentially, so the gradient vanishes regardless of `f`. Positions that are far apart have no force pulling them closer, even if `f` says they shouldn't be far apart.

The `[∂f/∂K]` factor can only reshape the gradient within the *active* region where `K(d)` is non-trivial — approximately d ∈ [0.5σ, 3σ]. Outside this region, the gradient envelope is effectively zero.

**Figure 8** (`reports/exp2/gradient_envelope.png`) illustrates this for three concrete losses: the pure Gaussian attractive loss (A), the Gaussian-MSE loss (B, C), and a distance-based MSE `L = (d − target_d)²`. The kernel-based losses both have gradient magnitude that rises from zero, peaks near d = σ, and decays back to zero for large d. The distance-based loss has gradient magnitude linear in distance error, unbounded above, and zero only at the target distance itself.

The consequence: for any kernel-based topographic loss, a stationary point at distance `d*` is only an *attractor* if `d*` lies within the active region of the kernel envelope. Stationary points far from σ — which includes both tight equilibria (`d* ≪ σ`) and loose equilibria (`d* ≫ σ`) — lack basins of attraction. Positions perturbed toward the zero-gradient tails cannot be pulled back.

This explains all three pathologies. In Formulation A, the only implicit stationary point is at d = 0 (coincidence), which is the one limiting point the gradient reaches — not as an attractor but as a *limit of* the vanishing gradient region. In B, the stationary point for zero-target pairs is at d → ∞, again a limiting point rather than an attractor. In C, the explicit equilibria at d* = 1.38 (high-cooccur) and d* = 5.73 (low-cooccur) are both outside the kernel's active region for the chosen σ = 3 (the kernel's active region being roughly d ∈ [1.5, 9]): the high-cooccur equilibrium sits in the d ≲ σ zero-gradient tail, and the low-cooccur equilibrium sits near the edge of the active region but just below the escape regime. Positions that drift past either equilibrium in the wrong direction cannot be pulled back.

### 4.4 What avoids the pathology

Two categories of alternative formulation avoid the gradient-vanishing envelope:

**Distance-based losses.** A loss formulated directly on `d_ij` rather than on `K(d_ij)` has gradient proportional to `(d_ij − target_d_ij)/d_ij · (pos_i − pos_j)`, which has magnitude `|d_ij − target_d_ij|` — unbounded above, zero only at the equilibrium, and not gated by a kernel envelope. Example: `L = mean((d_ij − target_d_ij)²)` with `target_d_ij` derived from co-occurrence (e.g., `target_d = d_far − (d_far − d_near) · cooccur_norm`). Globally well-behaved gradient dynamics; equilibria are true attractors.

**Non-gradient competitive-learning dynamics.** The original SOM algorithm uses a Gaussian kernel, but the kernel controls *which weights get updated* based on proximity of their *fixed grid locations* — not a gradient on learned positions. Grid positions don't move, so the vanishing-gradient issue doesn't arise. The kernel does a different job than it does in a gradient-based formulation.

A natural interpretation is that Gaussian-kernel-based gradient learning of grid positions imports a mathematical object (the kernel) from a setting where it was designed to solve one problem (competitive learning) into a setting where it must solve a different problem (gradient-based attraction-to-equilibrium). The kernel's properties — bell-shaped magnitude, fast decay — are well-matched to the first setting and pathological in the second.

We did not pilot a distance-based formulation in this project. The analytical argument is sufficient to explain the three observed pathologies, and the time cost of another pilot sequence exceeded the information it would have produced relative to the analytical diagnosis. Whether distance-based topographic regularization would produce *measurable effects on downstream learning* at this scale remains an open empirical question; the analytical argument here addresses only the *stability* of such formulations, not their *efficacy*.

### 4.5 Implications

The narrow implication is that future work attempting topographic regularization on transformer token embeddings via gradient-based losses should use distance-based formulations rather than kernel-based ones. This is a practical claim about avoiding a specific pathology, not a claim that topographic regularization works (or doesn't) at small transformer scale — the latter question we cannot answer without successful empirical trials.

The broader implication — which might be the more durable contribution of this section — is that importing mathematical objects across paradigms (SOM kernels into gradient learning) can smuggle in pathologies that the object's original setting doesn't exhibit. The Gaussian kernel in SOM is well-behaved because SOM's update rule is not gradient-based. When the same kernel is placed inside a gradient-based loss, its envelope becomes a gradient envelope, and its fast decay becomes vanishing-gradient regions. Checking whether a borrowed object retains its good properties in the new setting is worth doing analytically before committing to empirical work; the cost of not doing so here was four pilot experiments that each looked like a different calibration problem until the shared structural cause was visible.

---

## 5. Discussion

### What the trajectory findings say

The digit phase-structure finding (§3.2) is the clearest illustration of why trajectory analysis captures more than endpoint analysis. The endpoint shows a digit class with pairwise cosine +0.44 and an ordinal gradient — a snapshot that could be read as "the category formed and then partially loosened," which is correct at the level of aggregate cosine but wrong about what actually happened. The trajectory shows that the category first formed coherently (steps 0–47K), then *the ordinal axis emerged during a subsequent phase that partially decomposed the crude category*. The decomposition is information gain, not information loss. Without the trajectory we would have to infer that subsequent training "degraded" the category; with the trajectory we see that it refined into a more-informative representation.

The anchor-driven convergence finding (§3.3), once scoped to asymmetric triads, is a small but concrete trajectory-level signature: *how* a cluster forms depends on whether its members have comparable frequencies or whether one dominates. The specific mechanism — high-frequency members receive more gradient signal per step early in training and therefore reach their final region faster, creating a migration dynamic for lower-frequency members — suggests that frequency-driven "anchor" dynamics may be present in larger models with larger vocabularies, though we do not test this here. The property is worth looking for in larger-scale trajectory analyses.

The two-mode peak-then-decline observation (§3.4) generalizes the digit finding. Peak-then-decline isn't a unique digit phenomenon; it occurs for clause-punct and the `()` pair with quite different endpoint interpretations. "Ordinal refinement" and "role specialization" are two distinct mechanisms that produce superficially-similar trajectory signatures (coherence peaks, then declines), and they can be distinguished by whether the endpoint still shows class-level clustering (ordinal refinement: yes, loosely) or has diverged into anti-correlated subclasses (role specialization: no, the `()` pair has essentially decomposed by end of training). Analyses that aggregate classes without this decomposition risk missing the substantive difference between the two modes.

The flat-spectrum PCA finding (§2.3) and its resolution via substrate-check may be the most broadly transferable methodological contribution. Any trajectory-analysis work on trained networks will face the same issue: displacement vectors are dominated by init variance, not by the structure the analysis is trying to find. The recommendation — work on positions, not displacements; when displacements are genuinely the object of interest, use a matched-null baseline — applies across methodologies.

### What the topographic-regularization finding says

The narrow implication of §4 is that future topographic-regularization work on transformer embeddings should prefer distance-based losses over kernel-based losses. This is a practical point about avoiding a specific pathology.

The broader point has wider reach: importing a mathematical object across paradigms (here, the Gaussian kernel from SOM-style competitive learning into gradient-based position learning) can smuggle in pathologies that the original setting does not exhibit. The Gaussian kernel in SOM is well-behaved because SOM's update rule is not gradient-based — grid positions are fixed, the kernel controls which *weights* get updated, and the kernel's fast decay is precisely the property that produces the locality SOM is designed to achieve. When the same kernel appears inside a gradient-based loss, its fast decay becomes vanishing-gradient regions. Same object, different problem; the object works in one setting and pathologically in the other.

We were not the first to import SOM-style kernels into gradient learning, and topographic regularization on neural-network representations is an active area. The specific gradient-vanishing analysis in §4.3 did not surface in our (limited) survey of related work, though the general vanishing-gradient properties of Gaussian kernels are well-known in other contexts (saturating nonlinearities, kernel-regression gradients). We believe the analysis is useful to future researchers working in this area, but we are open to the possibility that it has been observed before and we missed it; if so, pointers are welcome.

### Connection to superposition and mechanistic-interpretability literature

The Exp 0 substrate-check diagnostics (§2.3's flat spectrum finding) show that the final embedding space is high-dimensional in a specific sense: the top principal component of displacement vectors captures only 1.5%, participation ratio is 127.6 out of 256. This is consistent with the direction-encoded-features picture in superposition work [@elhage2022] at small-model scale: features are encoded as directions, the number of features exceeds the trivially-available dimensionality, and the low-rank compression one might naively expect from "only a few categories" is not how the representations actually organize.

The two-mode peak-then-decline finding (§3.4) may connect to a specific phenomenon in mechanistic-interpretability work on larger models: the emergence of feature directions for within-category distinctions after coarse categories are in place. Work on induction heads [@olsson2022] and on the development of specific features across training [@nanda2023] describes phenomena with a similar phase-structured character, though at quite different scales and in different feature taxonomies.

The anchor-driven convergence finding (§3.3) does not have an obvious mechanistic-interpretability analogue in the literature we reviewed, but the qualitative phenomenon — a representational anchor forming early, others migrating toward it — resembles informal descriptions of how language models might bootstrap token relationships. Formalizing the anchor-ness criterion (initial cosine with final centroid, or equivalently, frequency-weighted early gradient signal) is a small methodological contribution that could be tested at larger scale.

### Scope and limits

- **Scale.** All findings are at 3.4M parameters. Whether they generalize to larger models is unknown. The trajectory-analytic methodology is scale-agnostic, but the specific phenomena (peak-then-decline on digits, anchor-driven convergence on `.!?`) may or may not recur at GPT-2-scale or larger.
- **Tokenization.** Byte-level encoding constrains what "compositional structure" means. At sub-word or word level, category structure is qualitatively different (orthographic vs. semantic), and phase-structure dynamics may manifest on different feature axes. We suspect the dynamics generalize in form, not in specifics.
- **Corpus.** TinyStories is a specific corpus with known stylistic constraints (simple grammar, restricted vocabulary). The "N-year-old" template dominating digit co-occurrence is a TinyStories-specific artifact; analogous formulaic patterns exist in all corpora but the specific templates will differ.
- **Topographic-regularization scope.** We diagnosed why Gaussian-kernel-based approaches don't work; we did not empirically test distance-based alternatives. Whether distance-based topographic regularization produces measurable effects on downstream learning at 3.4M scale is open.

### Open questions

- Do monotone-growth classes (uppercase, lowercase, consonants) eventually show their own Phase 2 at longer training (200K+ steps)? The digit trajectory suggests Phase 2 emerges around step 20K and peaks around step 47K; letters may have an equivalent timescale that hasn't played out at 100K.
- Does anchor-driven convergence at larger scale also track frequency asymmetry? Can the anchor-ness metric (init cosine with final centroid) be predicted from corpus statistics alone, without knowing the final embedding?
- Do distance-based topographic regularizers produce measurable downstream effects at this scale? This is a concrete empirical follow-up; we did not run it.
- Is there a SOM-style (non-gradient, competitive-learning) formulation that could be interposed into transformer training without the gradient-vanishing issue? The answer is probably yes but would require reformulating token embeddings as the competitive-learning weights rather than as free parameters.

---

## 6. Conclusion

Trajectory analysis of a small-scale transformer on a simple corpus surfaces phenomena that endpoint analysis misses. In a 3.4M-parameter byte-level model trained for 100K steps on TinyStories, at least two byte classes form in a two-stage dynamic — a coarse category first, then a within-category axis that partially decomposes the crude category in favor of a more-informative gradient — and certain tightly-clustered triads form via an asymmetry-gated anchor dynamic where the most-frequent member serves as the migration target for its neighbors. These are descriptive findings about how representations form, and they are visible because we inspected the full representational state at regular checkpoints rather than only at the end of training.

An attempt to impose topographic organization on these representations via gradient-based regularization produced a structural finding rather than a topographic regularizer: Gaussian-kernel-based losses have gradients that vanish in both limits (`d → 0` and `d → ∞`), so stable equilibria of such losses are not attractors. The pathology is structural, affects any loss factoring through a Gaussian kernel, and points toward distance-based formulations as the remedy for future work in this area. It is a negative result with an analytical diagnosis — useful to the field, we believe, even though it falls short of demonstrating that topographic regularization *works* at this scale.

We intend this work as a small case study in what close reading of small-model training dynamics can produce: specific, falsifiable, conditioned findings rather than sweeping claims. The methodological notes — positions over displacements, matched-null baselines, substrate-checks before committing to interpretations, dense-early snapshot cadence — are as much the contribution as the specific results.

---

## Appendix A — Methodological principles (summary)

The four principles stated in §2.3 each have a longer-form memo in the project repository under `memory/` (trajectory-methodology, matched-null, substrate-check, snapshot-cadence). The principles are:

- **Positions over displacements.** Analyze `w_t` at each checkpoint, not `w_t − w_s`. Displacement variance is dominated by isotropic init, which masks structure.
- **Matched-null.** When a new experiment changes parameter allocation or optimizer setup, the baseline must be the same code path with intervention neutralized — not an earlier experiment.
- **Substrate-check.** When a finding feels clean enough to write up, run one more substrate check. Corrected two near-overclaims in this work (flat-spectrum resolution; digit-arithmetic → context-adjacency).
- **Dense-early snapshots.** Most class formation occurs in the first 10–20% of training. Dense cadence there, sparser after.

## Appendix B — Corpus characterization for the digit axis correction

Raw digit-byte counts in a 200M-byte sample of the TinyStories training corpus:

| byte | count | fraction of corpus |
|---|---:|---:|
| `0` | 475 | 0.0002% |
| `1` | 577 | 0.0003% |
| `2` | 289 | 0.0001% |
| `3` | **5,172** | **0.0026%** |
| `4` | 155 | 0.0001% |
| `5` | 206 | 0.0001% |
| `6` | 54 | 0.0000% |
| `7` | 51 | 0.0000% |
| `8` | 52 | 0.0000% |
| `9` | 65 | 0.0000% |

Digit `3` accounts for 73% of all digit bytes. Sample of bigram contexts around digit `3`: `" 3 "` (5121 of 5172 occurrences), next-byte distribution dominated by `" "` (4338) and `"-"` (626). Sampled 20-byte windows show the pattern is overwhelmingly the "`N-year-old`" template (`"She was 3 years ol"`, `"a 3 year old boy"`, `"was only 3 years ol"`, etc.).

Arithmetic-syntax regex counts in the same sample:

| pattern | matches |
|---|---:|
| `\d\s*\+\s*\d` | 0 |
| `\d\s*-\s*\d` | 14 (mostly date ranges like `"1-2"`) |
| `\d\s*=\s*\d` | 0 |
| `\d\s*x\s*\d` | 0 |
| `\d\s*\*\s*\d` | 0 |
| literal `" + "` | 0 |
| literal `" = "` | 1 |

The corpus has essentially no arithmetic syntax; digit co-occurrence structure is driven by quantity-template contexts (ages, counts), not by numerical composition.

## Appendix C — Additional figures

- `reports/exp0/displacement_pca.png` — flat spectrum of displacement vectors referenced in §2.3. Top PC captures 1.53% of variance; participation ratio 127.6 / 256.
- `reports/exp1/within_class_coherence.png` — supplementary version of Figure 1 with zoom on first 20K steps.
- `reports/exp1/token_movement.png` — common vs rare byte per-step velocity referenced in §3.6.
- `reports/exp0/interpretation_stats.json` — nearest-neighbor tables for 21 anchor bytes (space, common letters, digits, punct, quotes, brackets), used to verify class structure was present in the substrate check for §2.3.

## Appendix D — Exp 2 pilot details

Full trajectory data and per-step diagnostics for each Exp 2 pilot:

- `runs/exp2_pilot/` — pilot 1 (Gaussian, grid_lr_scale=10). Saturated at step ~500.
- `runs/exp2_pilot2A/` — pilot 2A (Gaussian, scale=0.1). Too slow.
- `runs/exp2_pilot2B/` — pilot 2B (Gaussian, scale=1). Saturated at step ~3000.
- `runs/exp2_pilot_rngmatched/` — RNG-matched baseline (weight=0). Established ~0.055 MPS variance.
- `runs/exp2_pilot2C_mse/` — pilot 2C (MSE-simple). Escape.
- `runs/exp2_pilot2D_eq/` — pilot 2D (equilibrium MSE). Bimodal.

Full analytical derivation in `reports/exp2/gradient_vanishing_analysis.md`.

## Appendix E — Reproducibility

Code, analysis scripts, and configs:
- Model / training: `autoresearch/exp0_train.py`, `autoresearch/exp2_train.py`, `train_r4.py`
- Co-occurrence preprocessing: `autoresearch/precompute_cooccur.py`
- Analysis: `analysis/exp1_trajectories/*.py`
- Snapshots and logs: `runs/`
- Git repository: the `autoresearch/trajectory-topography` branch.

Compute: all experiments ran on an M1 Pro MacBook using PyTorch + MPS. Total wall-clock ≈ 12h across Exp 0 and Exp 2 pilots.

Random seed: 42 for all runs. MPS floating-point non-determinism produces ~0.055 val_bpb variance even with identical seeds across runs; matched-null comparisons (§2.3) control for this when it matters.

---

## Reference placeholder

- [@saxe2019] — Saxe, McClelland, Ganguli. *A mathematical theory of semantic development in deep neural networks*. PNAS 116(23), 2019. To cite for the hierarchical-feature-emergence theoretical frame in §3.2.
- Additional references TBD — Elhage et al. on superposition, Kohonen on SOM for §4.4.
