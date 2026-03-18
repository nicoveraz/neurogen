# Claude Code Instructions for NeuroGen

## What This Project Is

NeuroGen is a research project testing whether cellular automata can generate better weight initializations for transformers than random initialization. Read `NEUROGEN_PROJECT.md` for the full specification.

## How to Work on This Project

### Always follow the sprint order in NEUROGEN_PROJECT.md
Do not skip ahead. Each sprint depends on the previous one. A working `train.py` matters more than a fancy CA.

### Code style
- Python 3.11+, type hints everywhere
- Use `dataclasses` for configs, not dicts
- Docstrings on all public functions (Google style)
- Keep files under 300 lines — split if larger
- Format with `black`, lint with `ruff`

### Testing approach
- Read `NEUROGEN_TESTING.md` for the full test and benchmark specification
- Every module gets a test file in `tests/`
- Test with tiny configs (n_layer=2, n_embd=64, block_size=32) so tests run in seconds
- Use `pytest` with fixtures for model and data setup (see conftest.py spec in NEUROGEN_TESTING.md)
- Smoke tests first, correctness tests second
- Use markers: `@pytest.mark.slow` for >30s tests, `@pytest.mark.gpu` for CUDA tests
- Run fast tests: `pytest tests/ -v -m "not slow"`
- Each sprint has a validation checklist in NEUROGEN_TESTING.md — verify before marking complete

### Benchmarking approach
- Benchmarks (BM1-BM8) are separate from tests — they measure research questions, not correctness
- Each benchmark has a defined protocol, output format, and report template
- Quick suite (~5 min CPU) runs on every PR via CI
- Standard suite (~2-4h GPU) runs for milestone validations
- All benchmark results go to `outputs/benchmarks/` with raw data, figures, and markdown reports

### Key interfaces (do not break these)
```python
# Model weight interface
model.get_weight_tensors() -> dict[str, torch.Tensor]
model.set_weight_tensors(weights: dict[str, torch.Tensor]) -> None

# Initializer interface (baselines and CA must both follow this)
def initialize(model: GPT, config: dict) -> dict[str, torch.Tensor]

# CA genome interface
genome.develop(seed, target_shape, n_steps) -> torch.Tensor

# Device detection (ALWAYS use this, never hardcode devices)
get_device() -> str  # returns "cuda", "mps", or "cpu"
```

### Device handling (critical for Apple Silicon)
- **NEVER** write `torch.device("cuda")` or `.cuda()` directly in any module
- **ALWAYS** use `get_device()` from `neurogen/config.py`
- Analysis functions (SVD, spectral norm, eigendecomposition) must move tensors to CPU before computing — MPS doesn't support all linalg ops
- Use `float32` everywhere, never `float64` (MPS has inconsistent float64 support)
- `torch.compile()` should be gated behind `if device == "cuda"` — it's limited on MPS
- For `torch.multinomial` in text generation, add a try/except that falls back to CPU
- All tests should run on any device — use the `device` fixture from conftest.py
- Every CLI script must accept `--device` flag to override auto-detection

### When implementing CA variants
1. Start with the simplest version that produces valid weight tensors
2. Verify: correct shape, finite values, reasonable magnitude (std ~0.02)
3. Visualize the developed weights as heatmaps before training with them
4. Only then integrate with the training pipeline

### Experiment YAML format
Follow the schema in NEUROGEN_PROJECT.md Phase 4.1. Every experiment must specify: name, hypothesis, model config, dataset, init method, training config, metrics, and random seeds.

### Git workflow
- Feature branches: `sprint-N/description`
- Commit messages: `[sprint-N] description`
- Tag working checkpoints: `v0.1-sprint1-complete`

### When in doubt
- Simpler is better
- Match Karpathy's nanoGPT conventions where applicable
- CPU-runnable defaults with GPU as opt-in
- Log more than you think you need
