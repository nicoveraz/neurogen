# What 100K Steps Reveal About Developmental Attention Windows

*NeuroGen Part 3: Extended training, generation samples, and what the hierarchy looks like after 5× longer training*

---

In [Part 1](https://open.substack.com/pub/nicoveraz/p/neurogen-teaching-transformers-to), I reported that forcing early transformer layers to attend locally before opening to global context improves training by 1.5% at 3.4M parameters. In [Part 2](https://open.substack.com/pub/nicoveraz/p/why-attention-windows-actually-work), I killed three hypotheses about why it works and identified the surviving mechanism: forced gradient specialization through reduced parameter coupling.

Both posts covered models trained for 20,000 steps. The obvious question: what happens when you keep training?

## 100K Steps: The Advantage Persists

I extended both the baseline and quartic window models to 100,000 steps on TinyStories — 5× longer than the original validation runs. Same architecture (3.4M parameters, depth 4), same seed, same hardware (M1 Pro, $0).

![100K Training Curves](https://raw.githubusercontent.com/nicoveraz/neurogen/main/charts/100k_training_curve.png)

```
Steps      Baseline    Quartic     Gap
1,000      1.3394      1.3261      +0.99%
5,000      1.1216      1.1086      +1.16%
10,000     1.0512      1.0210      +2.87%  ← peak
20,000     0.9709      0.9572      +1.41%
50,000     0.8983      0.8933      +0.56%
100,000    0.8072      0.7994      +0.97%  ← persists
```

The advantage peaks at +2.87% during the early hierarchy-forming phase (10K steps), then settles to a persistent +0.97% at 100K. The quartic model reached 0.7921 bpb at its best checkpoint (step 96K). Neither model is fully converged.

This matches the developmental intuition: the constraint helps most when the model is building its initial layer hierarchy. After the hierarchy is established, the benefit becomes a stable structural improvement — the quartic model found a better solution, not just a faster path to the same solution.

## What the Models Actually Write

At 3.4M parameters, neither model writes literature. But at 100K steps, the text is coherent enough to see patterns. I generated 240 samples per model (12 prompts × 20 seeds) at temperature 0.8 with top-p 0.9 nucleus sampling.

Aggregate metrics across all 240 samples:

```
Metric                Baseline    Quartic
Mean words/sample     31.2        30.3
Vocabulary diversity  0.789       0.801 (+1.5%)
3-gram repetition     0.011       0.015
```

On average, the differences are modest — both models produce similar-quality TinyStories output at this scale. But the interesting signal is in the failure modes. When baseline fails, it tends toward repetitive phrase loops. Here are the most contrasting pairs I found (cherry-picked from the best-of-5 seeds per prompt — not every sample looks like this):

**Prompt: "The dog found a big red"**

Baseline (seed 256, 4-gram repetition: 20.8%):
> *ball. The dog was happy to play with the ball. The dog wagged his tail and licked the dog's ears. The dog wagged its tail and barked happily. The dog wagged its tail and barked happily. The dog ran to catch the dog. The dog licked their faces and bi...*

Quartic (seed 256):
> *ball of yarn in his mouth. He bit it and made a loud noise. The blue ball rolled into the ball. The ball was sad.*

**Prompt: "Mom said we could go to the"**

Baseline (seed 137, 4-gram repetition: 8.0%):
> *park and play on the swings. But they had to be quiet and careful. They wished they could go to the park and see the trees and the flowers. They were careful not to fall off the swings and the slide. They loved to play on the swings and the slide...*

Quartic (seed 137):
> *park to play with it. Tim liked the park better, so he grabbed the trunk and put it on the swingsing.*

**Prompt: "Tom was very sad because"**

Baseline (seed 137, 10 words then stops):
> *he could not play on the swings and the slide.*

Quartic (seed 137, 19 words, continues into narrative):
> *he really wanted to go to the market. His mom was very kind and said that Tom needed help.*

**Prompt: "The dog found a big red" (from the 20-seed run)**

Baseline (seed 4, 4-gram repetition: 12.5%):
> *ball and started to play with it. The dog barked and wagged its tail. The dog saw the red ball and wanted to play with it. The dog wanted to play with the ball too.*

Quartic (seed 4):
> *ball. The dog ran after the ball and jumped off the ball. The ball flew off and the dog started to chase it. The ball started to fly away. The dog was sad because the ball was broken. The dog ran after the ball, but it could not catch it.*

I want to be honest about what these show: the quartic model is slightly better, not dramatically better. At 3.4M parameters, the real evidence is the bpb improvement, not generation quality. The qualitative differences become clearer in the failure modes — baseline loops on phrases, quartic introduces new events. But both models produce recognizable TinyStories-style output, and on most random seeds the samples are hard to tell apart.

## Attention Entropy: Seeing the Hierarchy

This is the strongest piece of evidence. I measured attention entropy at each layer for both models on validation data at both 20K and 100K steps.

Entropy measures how focused each layer's attention is. Low entropy = the layer attends to a few specific positions (focused, specialized). High entropy = attention spreads broadly (diffuse, unspecialized).

![Attention Entropy Per Layer](https://raw.githubusercontent.com/nicoveraz/neurogen/main/charts/attention_entropy_per_layer.png)

![Entropy 20K vs 100K](https://raw.githubusercontent.com/nicoveraz/neurogen/main/charts/attention_entropy_20k_vs_100k.png)

```
         --- 20K Steps ---              --- 100K Steps ---
Layer    Baseline  Quartic  Change      Baseline  Quartic  Change
L0       1.994     0.852    −57.3%      1.860     0.898    −51.7%
L1       1.506     0.916    −39.2%      1.933     1.248    −35.4%
L2       2.466     2.183    −11.5%      2.722     2.489    −8.6%
L3       2.120     2.449    +15.5%      2.382     2.605    +9.4%
```

The key findings:

- **Layer 0**: 57% lower entropy at 20K, still 52% lower at 100K. The quartic model's first layer attends to only 2-3 nearby positions. The baseline's layer 0 spreads attention across many positions.
- **Layer 3** (final): Slightly *higher* entropy in the quartic model. It compensates by using its full attention span more broadly for global integration.
- **The specialization is permanent.** The entropy gap narrows only slightly from 20K to 100K. This is not a transient training artifact — it's lasting structural hierarchy.

This is what the developmental constraint creates: a gradient from focused (early) to diffuse (late) that persists through 5× longer training. The baseline never develops this clean hierarchy on its own.

## The Mechanism, Revisited

Three things now converge:

1. **Gradient measurements** (Part 2): Windows increase gradient signal 18× while noise stays constant. The optimizer gets a clear direction instead of 48 competing directions.

2. **Attention entropy** (this post): Quartic early layers are measurably more focused — 52% lower entropy at 100K steps. The specialization is real, visible, and permanent.

3. **Extended training** (this post): The advantage peaks during early hierarchy formation (+2.87% at 10K), then stabilizes at +0.97% through 100K. The constraint is a curriculum: it shapes structure early, and that structure persists.

The mechanism isn't mysterious: constrain early layers to local patterns → they learn local patterns well → later layers get clean features → they can focus on global composition. It's the difference between telling someone "learn everything at once" versus "learn the basics first, then build up."

## What I Don't Know Yet

The 125M scaling experiments from Part 2 showed the gap *growing* with scale (+8.4% at 50K steps). The 3.4M extended training shows the gap *narrowing* (from +2.87% to +0.97% at 100K steps). These aren't contradictory — more layers means more room for hierarchy, while longer training allows the baseline to partially discover its own specialization — but the full picture needs more data points.

The open questions:

- **Does the benefit survive at 1B+?** Many inductive biases help at small scale and vanish at large scale. The 125M results are encouraging but preliminary (2 seeds, not converged).
- **Can it combine with other techniques?** Mixture-of-Depths, sparse attention schedules, and hybrid architectures (like Qwen3-Next's 3:1 linear/full ratio) are doing something similar from different angles. Do they stack or conflict?
- **Is there an optimal point to remove the constraint?** The developmental analogy suggests critical periods — windows might help most during early training and become unnecessary later. My Experiment 7 (train with windows, then switch to full attention) already shows this works: removing windows at step 10K gives a *better* result than keeping them for all 20K steps.

## Independent Research, Honest Scale

This is a solo project run on a laptop (M1 Pro) and occasional H100 rentals. The 3.4M results are rigorous (5 seeds, statistical tests, 7 mechanism experiments, attention entropy analysis at two training durations). The 125M results are promising but need more seeds. The 100K extended training adds confidence that the effect is real and persistent, not a training artifact.

I don't have the compute to answer the scaling question definitively. If you do and this interests you, the code and data are open:

→ [github.com/nicoveraz/neurogen](https://github.com/nicoveraz/neurogen)

→ [Paper: doi.org/10.5281/zenodo.19194323](https://doi.org/10.5281/zenodo.19194323)

→ [Part 1: Teaching Transformers to Grow Up](https://open.substack.com/pub/nicoveraz/p/neurogen-teaching-transformers-to)

→ [Part 2: Why Attention Windows Actually Work](https://open.substack.com/pub/nicoveraz/p/why-attention-windows-actually-work)

If this kind of biologically-inspired inductive bias interests you, I'd love to hear your thoughts.

*Nicolás Vera Z. — Emergency physician, Puerto Montt, Chile*
