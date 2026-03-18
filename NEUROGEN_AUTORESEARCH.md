# NeuroGen Auto-Research

## Philosophy

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch): the AI itself is the researcher. No complex Python decision engine, no strategy classes, no SQLite stores. Instead, a minimal harness and a capable LLM that reads results, forms hypotheses, modifies code, and iterates.

The key insight: an LLM can make better research decisions than a hand-coded decision tree. Let it.

## Architecture

```
prepare.py          Fixed evaluation harness (DO NOT MODIFY)
train.py            Single modifiable file — model + init + training loop
program.md          Instructions for the AI researcher
results.tsv         Append-only experiment log
git history         Keep/discard via commits, tags, and reverts
```

That's the entire system. No YAML configs, no experiment registries, no report generators.

## How It Works

### The Loop

```
┌─────────────────────────────────────────────┐
│  AI Researcher (LLM)                        │
│                                             │
│  1. Read results.tsv — what do we know?     │
│  2. Form hypothesis — what should we try?   │
│  3. Modify train.py — implement the idea    │
│  4. git commit — record the hypothesis      │
│  5. python train.py — run experiment (~2m)  │
│  6. Append to results.tsv                   │
│  7. Decide: keep (git tag) or discard       │
│  8. Repeat                                  │
└─────────────────────────────────────────────┘
```

### File Responsibilities

**`prepare.py`** (fixed, ~110 lines):
- Downloads and caches Shakespeare dataset
- Provides `load_data()` → (train_data, val_data, vocab_size, encode, decode)
- Provides `get_batch(data, batch_size, block_size, device)` → (x, y)
- Provides `evaluate_val_loss(model, val_data, ...)` → float
- Provides `get_device()` → "cuda" | "mps" | "cpu"
- Constants: `MAX_SEQ_LEN=256`, `TIME_BUDGET=120`, `EVAL_BATCHES=20`
- Never modified by the researcher

**`train.py`** (modifiable, self-contained):
- Contains the full model definition (GPT, Attention, FFN, Block)
- Contains initialization methods (baseline + experimental)
- Contains the training loop with cosine LR schedule
- Runs for exactly `TIME_BUDGET` seconds (2 minutes)
- Prints a `RESULT:` line at the end with all metrics
- Everything in one file — no imports from `neurogen/` package
- The AI researcher modifies this file freely

**`program.md`** (instructions for the AI):
- Research agenda: what questions to answer, in what order
- Phase descriptions: baselines → simple CA → structured CA → live CA → meta-learning
- Guidelines: one variable at a time, always compare to baseline, log everything
- Results format specification

**`results.tsv`** (append-only log):
```
experiment	val_loss	train_loss	steps	params	init_method	n_layer	n_head	n_embd	lr	batch_size	time	notes
baseline_xavier	2.4967	2.6448	956	834432	xavier_normal	4	4	128	0.0003	64	121.0	first baseline run
```

### Git Workflow

```bash
# Good result — keep it
git add train.py results.tsv
git commit -m "CA rule 30 init: val_loss=2.41, beats xavier baseline"
git tag keep1

# Bad result — discard and try something else
git revert HEAD

# Explore a tangent
git checkout -b tangent-live-ca
# ... experiment ...
git checkout main
```

## Why This Design

### vs. the complex decision engine (previous approach)

| Aspect | Complex Engine | Karpathy Style |
|--------|---------------|----------------|
| Decision maker | Python code (strategy classes) | LLM (reads results, thinks, decides) |
| State | SQLite database, YAML agenda | `results.tsv` + git history |
| Configuration | 500+ lines of strategy code | `program.md` (natural language) |
| Flexibility | Must code new strategies | LLM adapts on the fly |
| Setup cost | High (16 files, tests, etc.) | Near zero (3 files) |
| Research quality | Bounded by coded heuristics | Bounded by LLM capability |

### The tradeoff

The complex engine is more systematic and reproducible. The Karpathy style is more flexible and requires less infrastructure. For a research project exploring a novel hypothesis (CA weight initialization), flexibility wins — we don't yet know what the right search strategy is.

## Constraints

- **Time budget: 2 minutes per run.** Fixed in `prepare.py`. This forces the researcher to be efficient — better architectures, better learning rates, better initialization all show up as more steps completed or lower loss within the budget.
- **Single file.** Everything goes in `train.py`. No helper modules, no package imports. This prevents complexity creep and makes every experiment fully self-contained and reproducible via git.
- **Character-level Shakespeare.** Fixed dataset and evaluation. The researcher can change the model and training, not the task.

## Relationship to the neurogen/ Package

The `neurogen/` package contains the full library implementation (GPT, CA variants, baselines, analysis, live CA, etc.). The autoresearch system (`prepare.py` + `train.py` + `program.md`) is a separate, self-contained research harness that doesn't import from the package.

Think of it this way:
- **`neurogen/`** = the reusable library, tested and modular
- **`train.py`** = the research scratchpad, self-contained and disposable

The researcher can port successful ideas from `train.py` back into the `neurogen/` package once they're validated.

## Getting Started

```bash
# 1. Prepare data (one time)
python prepare.py

# 2. Run baseline
python train.py

# 3. Start the AI researcher
# (Point your LLM at program.md and let it iterate)
```

## Research Agenda Summary

See `program.md` for the full researcher instructions. In brief:

1. **Phase 1 — Baselines**: Sweep standard inits (xavier, kaiming, orthogonal, etc.) and model sizes
2. **Phase 2 — Simple CA**: Elementary CA (Rule 30/110), Game-of-Life-style, continuous CA as initializers
3. **Phase 3 — Structured CA**: Block-diagonal, low-rank, spectral, multi-scale CAs
4. **Phase 4 — Live CA**: CA running alongside gradient descent with decaying influence
5. **Phase 5 — Meta-learning**: Evolve CA rules using training loss as fitness

Success = a CA initialization that consistently beats the best standard init on val_loss within the fixed 2-minute training budget.
