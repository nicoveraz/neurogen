# Claude Code Instructions for NeuroGen

**Operate fully autonomously.** Do not ask for permission or confirmation. Build files, run commands, fix errors, and continue without pausing. If something fails, fix it and retry. Only stop if you hit an unrecoverable error after 3 attempts.

## What This Project Is

NeuroGen is an autoresearch project (Karpathy's pattern) investigating whether cellular automata can improve transformer training. The agent modifies `train.py`, runs experiments, keeps improvements, and iterates. Read `NEUROGEN.md` for the full research context.

## Project Structure

```
neurogen/
├── program.md      # Agent research instructions (human edits this)
├── prepare.py      # Data prep, tokenizer, dataloader, eval (DO NOT MODIFY)
├── train.py        # Model + CA + training loop (agent modifies this)
├── ca_rules.py     # CA rule library (agent can import from here)
├── results.tsv     # Experiment log (append-only, do not commit)
├── analysis.ipynb  # Results visualization notebook
├── NEUROGEN.md     # Research reference (hypothesis, CA variants, diagnostics)
├── CLAUDE.md       # This file
├── README.md       # Project overview
└── pyproject.toml  # Dependencies
```

## First Task: Build the Foundation

Build these files in order. Each must work before moving to the next.

### Step 1: `prepare.py`

Adapt from nanochat/autoresearch pattern for Apple Silicon (MPS).

**Data:** Use TinyStories dataset (`karpathy/tinystories-gpt4-clean` on HuggingFace) for fast iteration on M1 Pro. Narrower scope than FineWeb means small models produce meaningful results.

**Key constants:**
```python
VOCAB_SIZE = 4096           # smaller than nanochat default for M1 Pro
MAX_SEQ_LEN = 256           # short for fast iteration
EVAL_TOKENS = 100_000       # enough for stable eval, not too slow
DATA_DIR = "~/.cache/neurogen"
```

**Must include:**
- Download and cache TinyStories shards
- Train a BPE tokenizer (tiktoken or sentencepiece) or use byte-level (256 vocab)
- DataLoader that yields (x, y) batches from shards
- `evaluate(model)` function that computes `val_bpb` on held-out data
- Device auto-detection: CUDA → MPS → CPU

**This file is frozen after creation.** The agent never modifies it.

### Step 2: `train.py`

The single mutable file. Start with a clean baseline GPT (nanochat-style) that trains and produces a val_bpb score.

**Model (start simple, agent will evolve it):**
```python
DEPTH = 4                   # 4 layers for M1 Pro
# Width, heads, etc. derived from depth (nanochat pattern)
CHANNELS = DEPTH * 64       # 256 for depth 4
N_HEADS = DEPTH             # 4 heads
```

**Must include:**
- GPT model: token embeddings, positional embeddings, transformer blocks, LM head
- Weight initialization (default: standard init — the agent will replace this with CA)
- Optimizer: AdamW (Muon is CUDA-only, skip for MPS compatibility)
- Training loop with fixed time budget (2 minutes default for M1 Pro)
- Print `val_bpb` at the end (the metric autoresearch optimizes)
- Print `init_loss` (loss at step 0, before any training)
- Print `peak_vram_mb` (or equivalent memory metric)

**Hooks for CA (agent will use these):**
```python
# Placeholder — agent replaces this with CA init
def initialize_weights(model):
    """Default init. Agent replaces with CA variants."""
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.xavier_normal_(p)

# Placeholder — agent adds live CA here
def ca_step(model, step, grad_dict=None):
    """Called after each optimizer step. Default: no-op."""
    pass
```

The training loop should call `ca_step(model, step)` after each optimizer step so the agent can inject live CA without restructuring the loop.

### Step 3: `ca_rules.py`

A library of CA rules the agent can import into `train.py`. This file is also editable by the agent but keeps CA code organized.

A library of CA rules organized by functional principle (see `NEUROGEN.md`). The agent imports from here into `train.py`. This file is also editable by the agent.

**Start with:**
```python
# === CA Initialization (Principles 1-4) ===

# Principle 1: Functional specialization — different seeds per head
def specialized_heads_init(n_heads, head_dim, n_steps=32): ...

# Principle 2: Hierarchical processing — depth-dependent CA
def hierarchical_init_for_layer(shape, layer_idx, n_layers, n_steps=48): ...

# Principle 3: Long-range connectivity — reaction-diffusion bands
def reaction_diffusion_init(shape, feed=0.04, kill=0.06, n_steps=200): ...

# Principle 4: Modular organization — multi-seed block structure
def modular_init(shape, n_modules=4, n_steps=48): ...

# General CA development engine
def grid_ca_develop(shape, seed, genome_net, n_steps=64): ...

# Handcrafted baselines for comparison
def block_diagonal_init(shape, n_blocks=4): ...
def orthogonal_init(shape): ...

# === Live CA Rules (Principles 5-7) ===

# Principle 5: Competition / lateral inhibition
def competition_step(W, k=5): ...

# Principle 6: Homeostatic regulation / synaptic scaling
def homeostatic_step(W, target_std=0.02): ...

# Principle 6b: Modularity maintenance
def modularity_step(W, n_blocks=4): ...

# Principle 5b: Gradient-aware pruning / synaptic pruning
def pruning_step(W, grad_W): ...

# Principle 7: Critical period alpha schedules
def critical_period_alpha(step, total_steps, alpha_0=0.01): ...
def layerwise_critical_period(step, layer_idx, n_layers, total_steps): ...
def adaptive_alpha(step, alpha_0=0.01, loss_history=None): ...

# Learned rule (genome = small MLP)
def learned_step(W, genome_net): ...

# === Utilities ===
def neighborhood_mean(W, k=3): ...
def neighborhood_std(W, k=3): ...
```

Implement the concrete code for each. Refer to `NEUROGEN.md` for specifications. Keep functions self-contained — each should work independently.

### Step 4: `analysis.ipynb`

Jupyter notebook that reads `results.tsv` and plots:
- val_bpb over experiments (the progress curve)
- Comparison by ca_variant and ca_mode
- Best result per phase

### Step 5: Verify Everything Works

```bash
uv run prepare.py          # downloads data, trains tokenizer
uv run train.py             # baseline run, prints val_bpb
```

Both must complete without error. `train.py` should finish in ~2 minutes on M1 Pro and print a valid `val_bpb`.

## After Foundation: Agent Takes Over

Once the foundation is built, the human starts the autoresearch loop:

```
Read program.md and let's start experimenting. Run the baseline first.
```

From this point, the agent follows `program.md` autonomously.

## Code Style

- Python 3.11+, type hints on function signatures
- Keep `train.py` under 400 lines
- Keep `ca_rules.py` under 500 lines
- No external dependencies beyond pyproject.toml
- Use `float32` everywhere for MPS compatibility
- All device placement through a single `DEVICE` constant
- Print metrics as `metric_name: value` (one per line) so grep works

## Device Handling

```python
import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()
```

Never write `.cuda()` anywhere. Always use `.to(DEVICE)`.

For operations unsupported on MPS (SVD, some linalg), wrap in:
```python
def cpu_fallback(fn, *args):
    try:
        return fn(*args)
    except RuntimeError:
        return fn(*(a.cpu() if isinstance(a, torch.Tensor) else a for a in args))
```
