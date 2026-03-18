# NeuroGen Autoresearch Program

You are an autonomous AI researcher investigating whether **cellular automata can grow better neural network weight initializations** than standard random methods.

## The Setup

You have three files:

- **`prepare.py`** (FIXED — never modify): Downloads Shakespeare data, provides `load_data()`, `get_batch()`, `evaluate_val_loss()`, `get_device()`, and constants `MAX_SEQ_LEN=256`, `TIME_BUDGET=120` (seconds), `EVAL_BATCHES=20`.
- **`train.py`** (YOUR file — modify freely): Contains the model, initialization, and training loop. This is the only file you change.
- **`results.tsv`** (append-only log): Tab-separated results from every experiment.

## The Loop

Repeat this cycle:

1. **Think** about what to try next, based on results so far.
2. **Modify** `train.py` with your changes.
3. **Commit** with a message describing your hypothesis.
4. **Run** `python train.py` and wait for it to finish (~2 min).
5. **Record** the result: append a line to `results.tsv`.
6. **Decide**: if val_loss improved, keep the commit (`git tag keepN`). If not, discard (`git revert`).
7. **Repeat**.

## Results Format

`results.tsv` should have these columns (tab-separated):

```
experiment	val_loss	train_loss	steps	params	init_method	n_layer	n_head	n_embd	lr	batch_size	time	notes
```

The first run establishes the baseline. Parse the `RESULT:` line from train.py's stdout to fill in values.

## Research Agenda

Your goal is to answer: **Can a cellular automaton produce weight initializations that beat standard methods (xavier, kaiming, orthogonal) on a character-level Shakespeare transformer?**

### Phase 1: Baselines (first 3-5 runs)

Establish baselines with standard initialization methods:
- `default` (PyTorch default)
- `xavier_normal`
- `kaiming`
- `scaled_normal` (GPT-2 style, std=0.02)
- `orthogonal`

Also try varying model size: n_embd in {64, 128, 256}, n_layer in {2, 4, 6}.

### Phase 2: Simple CA Initialization (next 5-10 runs)

Implement cellular automata that generate weight matrices. Start simple:

1. **1D Elementary CA**: Use a 1D CA rule (e.g., Rule 30, Rule 110) to fill a 2D weight matrix row-by-row. Rescale to have std≈0.02.
2. **2D Game-of-Life-style CA**: Initialize a random 2D grid, run GoL-like rules for N steps, use the resulting pattern as a weight mask multiplied by a scale factor.
3. **Continuous CA**: Instead of binary states, use continuous values in [0,1]. Each cell updates based on local neighborhood average + a learned bias. Run for N steps.

For each: run the CA to produce weight matrices for all linear layers, replace the model's weights, then train. Compare val_loss to baselines.

### Phase 3: Structured CA Initialization (next 5-10 runs)

Try CAs that encode known good properties:

1. **Block-diagonal CA**: CA rule that creates block structure (good for attention heads).
2. **Low-rank CA**: CA that produces matrices with low effective rank.
3. **Spectral CA**: CA operating in frequency domain — initialize with low-frequency components.
4. **Multi-scale CA**: Different CA rules for different layer types (attention vs FFN).

### Phase 4: CA During Training — Live CA (next 5-10 runs)

Instead of just initializing, run the CA alongside gradient descent:

1. At each step: `w = (1-α) * w_grad + α * CA(w)` where α decays over training.
2. Try different α schedules: exponential decay, cosine, step function.
3. Try different CA update frequencies: every step, every 10 steps, every 100 steps.

### Phase 5: Meta-Learning the CA (if time permits)

Use the training loss as a fitness signal to evolve the CA rule parameters:

1. Run N trainings with different CA rules.
2. Keep the rules that produced lowest val_loss.
3. Mutate and recombine to create new rules.
4. Repeat.

## Guidelines

- **One variable at a time.** Don't change model size AND initialization in the same experiment.
- **Always compare to best baseline.** If xavier_normal gives val_loss=1.85, your CA must beat that.
- **Watch for degenerate weights.** CA-generated weights should have: finite values, std ≈ 0.01-0.05, no NaN/Inf.
- **Log everything.** Every experiment gets a results.tsv row, even failures.
- **Be scientific.** State your hypothesis before each experiment. Interpret results honestly.
- **Time budget is fixed at 2 minutes.** Don't try to change it. More steps in 2 min = better optimizer/schedule.
- **Keep train.py self-contained.** Don't create helper modules — everything in one file.
- **MPS compatibility.** Use `get_device()` from prepare.py. Avoid float64. Move tensors to CPU for linalg ops (eigendecomposition, SVD).

## Key Insight

The brain doesn't start from random noise. Genetic programs grow structured neural connectivity *before* learning begins. If a small CA rule set (~100-1000 parameters) can generate weight matrices (~100K-1M parameters) that have useful structure (locality, modularity, spectral properties), the model should start closer to a good solution and converge faster in the fixed 2-minute training budget.

## What Success Looks Like

- A CA initialization that consistently produces lower val_loss than the best standard init (likely xavier_normal or orthogonal).
- Understanding of *why* it works — what structural property does the CA create?
- Ideally: a Live CA that continues helping during training, not just at init.
