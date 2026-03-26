# Why Attention Windows Actually Work (It's Not What We Thought)

*NeuroGen Part 2: Scaling to 125M parameters and dissecting the mechanism*

---

In [Part 1](https://open.substack.com/pub/nicoveraz/p/neurogen-teaching-transformers-to), I reported that forcing transformer layers to "grow up" — restricting early layers to local attention before opening to global context — improves training by 1.5% at 3.4M parameters. Every seed beat the baseline. Throughput was identical. It was free.

The obvious next questions: Does it scale? And *why* does it work?

I spent the last week answering both. The results surprised me.

## Scaling to 125M: The Gap Keeps Growing

I rented an H100 on RunPod and trained GPT-2 Small (12 layers, 768 dim, 125M parameters) on FineWeb-Edu with the same quartic window schedule. The first thing I noticed: with Flash Attention 2's native sliding window support, the windowed model actually trains **faster** than baseline (2.84 vs 2.78 steps/sec). Free speed on top of free quality.

But the real surprise was the learning curve. At 3.4M scale, the window advantage was a constant offset — about 1.5% from early training through convergence. At 125M, the advantage **accelerates**:

| Training Step | Baseline bpb | Quartic bpb | Gap |
|:---|:---|:---|:---|
| 5,000 | 4.911 | 4.820 | +1.9% |
| 10,000 | 4.580 | 4.390 | +4.2% |
| 20,000 | 4.095 | 3.942 | +3.8% |
| 30,000 | 3.844 | 3.576 | +7.0% |
| 40,000 | 3.538 | 3.188 | +9.9% |
| 50,000 | 3.447 | 3.001 | **+12.9%** |

After 50k steps, the quartic window model is nearly 13% better — and the gap is still widening. Neither model is converged.

This isn't faster convergence. The windowed model is finding a fundamentally better optimization trajectory, and the longer it trains, the more that trajectory diverges from baseline.

**Caveat:** These are preliminary results with only 2 matched-seed pairs. The effect is large enough to be visible without statistics, but I'd want 5+ seeds and full convergence before claiming a final number. The 3.4M results (5 seeds, p=0.001) are validated. The 125M results are promising but not proven.

## The Hypothesis That Died

I had a clean hypothesis for why windows help: **gradient noise removal**. The softmax backward pass couples all positions — even positions the model ignores contribute gradient signal. By masking those positions, windows should clean the gradient, removing noise from non-attended positions.

This predicts specific, testable things:
1. Gradient noise should decrease as windows shrink
2. Non-attended positions should contribute significant gradient contamination (>30%)
3. The improvement should be replicable by any method that reduces gradient variance

I tested all three. All three failed.

## Experiment 1: Gradient Noise Is Constant

I loaded a trained baseline checkpoint (frozen weights, no training) and measured gradient properties at 10 different window sizes, from 8 tokens to 256 (full attention). For each, I ran 50 forward-backward passes on validation data and measured the gradient's signal-to-noise ratio, direction stability, and effective rank.

| Window | SNR | Signal Norm | Noise Norm | Stability |
|:---|:---|:---|:---|:---|
| 8 | 5.59 | 0.0323 | 0.0058 | 0.97 |
| 16 | 3.23 | 0.0175 | 0.0054 | 0.92 |
| 32 | 0.77 | 0.0041 | 0.0054 | 0.36 |
| 64 | 0.34 | 0.0018 | 0.0053 | 0.09 |
| 128 | 0.34 | 0.0018 | 0.0054 | 0.09 |
| 256 | 0.33 | 0.0017 | 0.0053 | 0.08 |

Look at the noise norm column. **It's constant.** 0.0053 regardless of window size. Windows don't remove noise.

What changes is the *signal*. The signal norm increases 18× from full attention (0.0017) to window 8 (0.0323). With fewer positions to attend to, the gradient points in a more consistent direction across batches. The gradient becomes more *coherent*, not cleaner.

## Experiment 2: Softmax Coupling Is Tiny

I decomposed the softmax backward pass into contributions from positions the model actually attends to (attention weight > 1/T) versus positions it ignores. If gradient noise removal is the mechanism, the "noise fraction" from ignored positions should be large.

| Layer | Noise Fraction | Natural Attention Span |
|:---|:---|:---|
| 0 | 5.0% | 12.9 tokens |
| 1 | 4.0% | 7.4 tokens |
| 2 | 6.9% | 21.3 tokens |
| 3 | 6.7% | 13.0 tokens |

Noise fraction is 4–7%. Not 30–50%. The softmax coupling that I hypothesized was corrupting gradients barely exists. The model already concentrates its attention on a small number of positions, and the gradient contributions from the rest are negligible.

The gradient noise removal hypothesis is dead.

## Experiment 3: Larger Batch Can't Replicate It

If windows improve training through variance reduction (fewer gradient sources = less variance), then simply training with a larger batch size should give the same benefit. Larger batches average over more samples, reducing gradient variance the honest way.

I trained four configurations for 2,000 steps each:

| Config | Effective Batch | Final bpb | Tokens Seen |
|:---|:---|:---|:---|
| Full attention, batch 32 | 32 | 1.242 | 16M |
| **Quartic windows, batch 32** | 32 | **1.224** | **16M** |
| Full attention, batch 128 | 128 | 1.086 | 66M |
| Full attention, batch 256 | 256 | 1.150 | 66M |

The larger-batch models look better at the same step count, but they've seen 4–8× more data. At equal token count (16M tokens), the picture inverts:

- Quartic windows: **1.224** bpb (2000 steps × batch 32)
- Baseline: 1.242 bpb (2000 steps × batch 32)
- Batch 128: 1.357 bpb (500 steps × batch 128)
- Batch 256: 1.321 bpb (500 steps × batch 256)

Larger batch is **worse** than baseline at the same token budget. Fewer gradient updates hurts more than cleaner gradients help. Variance reduction is not the mechanism.

## So What IS the Mechanism?

Process of elimination:

| Hypothesis | Prediction | Result |
|:---|:---|:---|
| Gradient noise removal | Noise decreases with smaller windows | **Dead.** Noise is constant |
| Softmax coupling contamination | >30% gradient from non-attended positions | **Dead.** Only 4–7% |
| Variance reduction | Larger batch replicates the effect | **Dead.** Can't replicate |
| **Forced specialization** | Early layers build local features that help later layers | **Consistent with all data** |

The surviving hypothesis: **forced architectural specialization**. By constraining early layers to local attention, you force them to learn compositional features — n-gram patterns, syntactic boundaries, local structure — that they might otherwise skip in favor of trying to learn everything at once. Later layers then build on these local features for more effective global integration.

This explains why:
- The gradient becomes more *coherent* (not less noisy) — the model has a simpler task at each layer
- The effect grows with depth (12 layers benefit more than 4 layers from the hierarchy)
- The gap widens over training (the compositional features compound)
- Larger batch can't replicate it (it's not about gradient quality, it's about what the model is forced to learn)

It's the difference between telling someone "learn everything at once" versus "learn the basics first, then build up." The constraint is a curriculum imposed on the architecture.

## What This Means

The practical takeaway is simple: add quartic attention window growth to your transformer. It costs nothing — zero extra parameters, and with Flash Attention it's actually faster. The theoretical window schedule is one line of code:

```python
window = base + ((layer + 1) / n_layers) ** 4 * (seq_len - base)
```

The deeper takeaway is about optimization landscapes. Full-attention transformers explore a vast parameter space where every layer competes for both local and global patterns simultaneously. Window constraints partition the search space: early layers search locally, late layers search globally. The partitioned search finds better solutions because the compositional structure is preserved by design rather than rediscovered by accident.

## What's Next

The 125M results need more seeds and longer training. The gap at 50k steps is 12.9% and still growing — I want to see where it plateaus. I also haven't tested whether the optimal exponent shifts at depth 12 (quartic was optimal at depth 4, but 12 layers might want a steeper or shallower curve).

The mechanism analysis should be repeated at 125M scale. The gradient experiments were all done at 3.4M — confirming that forced specialization operates the same way at larger scale would strengthen the story.

And the obvious question: what happens at 350M? 1B? If the effect continues growing with scale, this becomes genuinely important for efficient training.

---

*Code and data: [github.com/nicoveraz/neurogen](https://github.com/nicoveraz/neurogen)*

*Paper: [doi.org/10.5281/zenodo.19194323](https://doi.org/10.5281/zenodo.19194323)*

*Part 1: [NeuroGen: Teaching Transformers to Grow Up](https://open.substack.com/pub/nicoveraz/p/neurogen-teaching-transformers-to)*
