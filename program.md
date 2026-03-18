# NeuroGen Research Program

You are an autonomous ML researcher investigating whether cellular automata can improve transformer training. You modify `train.py`, run experiments, and keep improvements. You do not touch `prepare.py`.

## Setup (first time only)

1. Create the branch: `git checkout -b neurogen/<tag>` from current master.
2. Read the repo files for context:
   - `README.md` — project overview
   - `NEUROGEN.md` — full CA research reference (hypothesis, variants, rules, diagnostics). **Read this carefully before your first experiment.**
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. **Do not modify.**
   - `train.py` — model, optimizer, training loop, CA code. **This is the file you modify.**
3. Verify data exists: check that `~/.cache/neurogen/` contains data shards and tokenizer. If not, tell the human to run `uv run prepare.py`.
4. Initialize `results.tsv` with header: `experiment\ttag\tval_bpb\tinit_loss\tca_variant\tca_mode\tvs_baseline_pct\ttotal_flops\tnotes`
5. Confirm setup looks good. Once confirmed, begin experimentation.

## Experiment Loop

Each experiment:

1. **Hypothesis.** Write a one-line hypothesis in your message before touching code. Example: "Grid CA init with 64 development steps will produce lower init_loss than Xavier."

2. **Modify `train.py`.** Implement your idea. Keep changes focused — one idea per experiment. You may add CA initialization functions, live CA rules, alpha schedules, diagnostic metrics, or modify architecture/hyperparameters. All changes go in `train.py` only.

3. **Commit.** `git add train.py && git commit -m "exp: <short description>"`

4. **Run.** `uv run train.py > run.log 2>&1`

5. **Read results.** `grep "^val_bpb:\|^init_loss:\|^ca_delta_norm:\|^ca_grad_alignment:\|^peak_vram_mb:" run.log`
   - If grep is empty, the run crashed. Run `tail -n 50 run.log` to read the error. Fix and retry. If you can't fix after 3 attempts, revert and try a different idea.

6. **Record.** Append a row to `results.tsv` with the experiment results.

7. **Decision.**
   - If `val_bpb` improved (lower): keep the commit. This is the new baseline.
   - If `val_bpb` is equal or worse: `git reset --hard HEAD~1` to discard.
   - Exception: if `init_loss` improved significantly but `val_bpb` didn't, note this — it suggests the init is better but training dynamics need tuning.

8. **Next.** Based on results so far, form your next hypothesis and repeat.

## Research Phases

Follow this order. Move to the next phase when you have clear evidence (positive or negative) about the current one.

### Phase 1: Establish Baselines
Run the unmodified `train.py` to get the reference `val_bpb`. Then try standard structured initializations (Xavier, orthogonal, identity-like, block-diagonal) to understand how much init matters at all. If structured init already helps, that's signal for CA.

> Before moving to Phase 2, run: `uv run benchmark.py --compare "default,xavier,orthogonal,block_diagonal" --seeds 5`
> This gives you multi-seed baselines with statistical significance. Note which init diagnostics (head_diversity, block_diag_ratio) correlate with better val_bpb.

### Phase 2: CA Initialization — Functional Structure
Implement CA variants from `NEUROGEN.md`. The key idea: don't generate random patterns, generate patterns that encode **functional principles** from neuroscience (modularity, specialization, hierarchy, long-range connectivity). Start with modular multi-seed grid CA. Key measurements:
- `init_loss` (loss at step 0) — directly measures init quality
- `val_bpb` — does better init translate to better final model?
- `head_diversity` — are attention heads starting differentiated?

Try the functional principles in priority order from `NEUROGEN.md`:
1. Modular init (Principle 4 — cortical columns: independent processing units)
2. Specialized heads (Principle 1 — different CA seed per head, like Broca ≠ Wernicke)
3. Hierarchical init (Principle 2 — depth-dependent CA: local-to-global processing)
4. Long-range connectivity (Principle 3 — reaction-diffusion bands: arcuate fasciculus analog)

For each, try 2-3 CA step counts (16, 64, 256). Keep the best.

> Before moving to Phase 3, run: `uv run benchmark.py --compare "default,best_baseline,best_ca_variant" --seeds 5`
> Only proceed if the CA improvement is statistically significant (p < 0.05) or the trend is clear across all seeds. If not significant, try more CA variants before moving on.

### Phase 3: Live CA During Training — Ongoing Development
Add CA rules from `NEUROGEN.md` Principles 5-7. The brain's developmental programs don't stop when learning starts — synaptic scaling, pruning, and lateral inhibition continue alongside Hebbian learning. Start with homeostatic normalization (Principle 6, safest). Then competition (Principle 5). Key measurements:
- `ca_delta_norm` — is the CA changing weights?
- `ca_grad_alignment` — cooperation or competition with gradient?
- `val_bpb` — does it help?

Implement critical period alpha schedules (Principle 7) — strong CA early, fading as training progresses. Try layerwise critical periods where earlier layers close first (like phonological before syntactic critical periods).

### Phase 4: Combine Init + Live
Best CA init (Phase 2) + best live rule (Phase 3). The biological analog: genetic programs build structure (init), ongoing developmental processes maintain it (live CA), experience-driven learning refines it (gradient descent). All three simultaneously.

### Phase 5: Learned CA Rules (Genome Evolution)
Implement a learned CA rule (small MLP genome). The autoresearch loop *is* evolution — each experiment that improves val_bpb = one generation of selection. The git history accumulates the genome's evolutionary trajectory. Start with tiny genome (2-layer MLP, 32 hidden).

### Phase 6: Advanced
Only if earlier phases show promise:
- Multi-timescale CA (fast homeostatic + slow structural, mimicking biological timescales)
- Per-layer scope (competition for attention Q/K, modularity for V/O, pruning for FFN)
- CA as learned optimizer (CA sees gradients — neuromodulation analog)

## Rules

- **One idea per experiment.** Don't combine multiple untested changes.
- **If it crashes, fix it.** Common issues: shape mismatch (CA output wrong size), NaN (CA magnitude too large, add clamping), OOM (CA overhead too high, reduce neighborhood size or update frequency).
- **Watch the overhead.** If your CA code makes training >30% slower, optimize it or reduce the CA update frequency.
- **Print diagnostics.** Always print `init_loss` (loss at step 0). For live CA, print `ca_delta_norm` and `ca_grad_alignment` at eval intervals.
- **The genome must be small.** Any CA rule's parameters should be <1% of model parameters. If your CA genome is bigger than that, simplify it. The whole point is compression: small program → structured weights.
- **Always compare against baseline.** Every CA result must be reported as "val_bpb X (baseline Y, improvement Z%)". Raw val_bpb without comparison is meaningless. Run `uv run benchmark.py --baseline` once to establish the reference.
- **Account for CA compute.** If your CA init takes 5 seconds to develop weights, report total wall time including development. The baseline gets those 5 extra seconds of training.
- **Check quality on significant improvements.** When val_bpb improves by more than 2%, also run: `uv run evaluate_quality.py`. A model with better val_bpb but higher repetition or lower diversity is suspicious — it might be overfitting or degenerating in a way that loss doesn't capture.
- **Negative results are valuable.** If CA init consistently doesn't beat Xavier, that's a finding. Record it clearly and move on.
- **Don't get stuck.** If a variant shows no promise after 3 experiments, try a different variant. If a whole axis (init/live/combined) shows no promise after 8 experiments, document the finding and move to the next axis.

## Notes for M1 Pro / MPS

- Use `float32` everywhere (MPS has inconsistent float64/bfloat16 support)
- For SVD/spectral analysis, move tensors to CPU first
- `torch.compile` doesn't work well on MPS — disable it or guard with device check
- If `torch.multinomial` errors during sampling, add CPU fallback
- Keep model small: depth 4-6, lower batch size if OOM
- Target ~2 min per experiment for fast iteration (~30 experiments/hour)
