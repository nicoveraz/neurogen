# Auto-Research Implementation Instructions

## Problem

The current `research/engine.py` is a dumb experiment runner: it executes pre-written YAML configs and generates reports. A human must read each report, decide what to do next, and manually write the next experiment YAML. The loop is open.

## Goal

Close the loop. After each experiment (or batch of experiments), the engine must:
1. Analyze results against defined success criteria
2. Decide what to run next (without human input)
3. Generate and queue the next experiment(s)
4. Repeat until budget exhausted or questions answered

## What to Build

### 1. Research Agenda File (`research/agenda.yaml`)

Replace the collection of individual experiment YAMLs with a single agenda file. This is the only human-authored input. It defines questions, not experiments.

```yaml
name: "NeuroGen CA Investigation"
hardware: "macbook_m1pro_16gb"
total_budget_hours: 80
report_every_n_experiments: 5

questions:
  - id: Q1_baselines
    question: "What are the convergence profiles of standard initializations?"
    success: "All 9 baselines profiled with loss curves and weight stats"
    metric: completeness  # not a loss metric — just run all baselines
    priority: 1
    depends_on: []
    max_hours: 2
    max_experiments: 30

  - id: Q2_ca_viability
    question: "Can any CA variant produce weights that train at all?"
    success: "At least one CA variant reaches val_loss < 4.0 within 1000 steps"
    metric: val_loss
    threshold: 4.0
    comparison: best_baseline
    priority: 2
    depends_on: [Q1_baselines]
    max_hours: 3
    max_experiments: 50

  - id: Q3_ca_beats_baseline
    question: "Does any CA init outperform the best baseline?"
    success: "CA init reaches baseline val_loss in ≥20% fewer steps OR achieves ≥5% lower final loss"
    metric: convergence_speed_ratio  # steps_ca / steps_baseline
    threshold: 0.8
    comparison: best_baseline
    priority: 3
    depends_on: [Q2_ca_viability]
    max_hours: 15
    max_experiments: 200

  - id: Q4_live_ca
    question: "Does live CA during training improve over init-only CA?"
    success: "Live CA achieves lower val_loss than init-only with same rule"
    metric: val_loss_improvement
    threshold: 0.0  # any improvement
    comparison: best_init_only_ca
    priority: 4
    depends_on: [Q3_ca_beats_baseline]
    max_hours: 20
    max_experiments: 300

  - id: Q5_meta_learned
    question: "Can meta-learning find a better CA genome than random?"
    success: "Meta-learned genome outperforms random genome by ≥10%"
    metric: val_loss_ratio
    threshold: 0.9
    comparison: random_genome_same_variant
    priority: 5
    depends_on: [Q3_ca_beats_baseline]
    max_hours: 30
    max_experiments: 500

  - id: Q6_transfer
    question: "Does a genome learned on tiny model help a larger model?"
    success: "Transferred genome outperforms random init on larger model"
    metric: val_loss
    comparison: xavier_on_larger_model
    priority: 6
    depends_on: [Q5_meta_learned]
    max_hours: 10
    max_experiments: 50
```

### 2. Results Store (`research/results_store.py`)

A persistent store (SQLite or JSON-lines file) that accumulates every experiment result. The decision engine queries this to make choices.

```python
class ResultsStore:
    def __init__(self, path: str = "outputs/results.db"):
        ...

    def record(self, experiment_id: str, config: dict, metrics: dict, 
               question_id: str, duration_seconds: float):
        """Store one experiment's results."""

    def query(self, question_id: str = None, variant: str = None,
              metric: str = None, top_k: int = None) -> list[dict]:
        """Retrieve results filtered and sorted."""

    def best_result(self, question_id: str, metric: str) -> dict:
        """Best result for a question by a given metric."""

    def baseline_results(self) -> dict:
        """All Q1 baseline results for comparison."""

    def budget_used(self, question_id: str = None) -> float:
        """Hours consumed, optionally filtered by question."""

    def experiment_count(self, question_id: str = None) -> int:
        """Number of experiments run."""
```

### 3. Decision Engine (`research/decision_engine.py`)

This is the core of auto-research. It takes results so far and the agenda, and produces the next batch of experiments to run.

**Decision logic by question type:**

**Q1 (baselines) — Exhaustive sweep, no decisions needed:**
- Generate one experiment per baseline × 3 seeds
- Mark complete when all finish

**Q2 (CA viability) — Explore broadly, abandon failures fast:**
- Start with 5 configs per CA variant (Stage 1 survey from NEUROGEN_EXPLORATION.md)
- After each batch:
  - If a variant has >3 configs that produce NaN or loss > 8.0 → abandon that variant
  - If a variant has any config with val_loss < 4.0 → mark variant as viable
  - For viable variants, try 5 more configs to confirm it's not a fluke
- Stop when: all variants classified as viable/abandoned, or budget exhausted

**Q3 (CA beats baseline) — Focused search on viable variants:**
- Pull the best baseline result from Q1 (target to beat)
- For each viable variant from Q2:
  - Run Optuna-style search over architectural hyperparams (Stage 2 from NEUROGEN_EXPLORATION.md)
  - After every 10 trials, check: is the best CA result within 20% of baseline?
    - If yes → increase training budget per trial (more steps for finer signal)
    - If no after 30 trials → deprioritize this variant, try next
  - If any config beats baseline → run 5 seeds to confirm statistical significance
- Stop when: significance confirmed, all variants exhausted, or budget exhausted

**Q4 (live CA) — Compare init-only vs live:**
- Take the best init CA config from Q3
- Test each live CA rule with that config
- After each rule tested (3 seeds):
  - Compare against init-only result from Q3
  - If live is better → try more alpha schedules for that rule
  - If live is worse → try next rule
- Then test multi-timescale combinations of the best rules
- Stop when: best live setup found or all rules tested

**Q5 (meta-learning) — CMA-ES with checkpointing:**
- Use best variant + architecture from Q3
- Run CMA-ES with progressive evaluation (from NEUROGEN_EXPLORATION.md)
- After every 10 generations:
  - Is meta-loss still decreasing? → continue
  - Has it plateaued for 20 generations? → stop, report best
  - Is best genome already beating random by >10%? → early success
- Stop when: success, plateau, or budget exhausted

**Q6 (transfer) — Simple validation:**
- Take best genome from Q5
- Apply to 2× and 4× model sizes
- 3 seeds each, compare against xavier at same size
- Binary outcome: does transfer work or not?

```python
class DecisionEngine:
    def __init__(self, agenda: ResearchAgenda, store: ResultsStore):
        self.agenda = agenda
        self.store = store

    def next_experiments(self, batch_size: int = 5) -> list[ExperimentConfig]:
        """Decide what to run next. Returns a batch of experiment configs."""
        # 1. Find the highest-priority active question
        question = self._get_active_question()
        if question is None:
            return []  # all done

        # 2. Check budget
        if self._is_over_budget(question):
            self._mark_abandoned(question, reason="budget_exhausted")
            return self.next_experiments(batch_size)

        # 3. Dispatch to question-specific strategy
        strategy = self._get_strategy(question)
        experiments = strategy.propose_next(
            question=question,
            results_so_far=self.store.query(question_id=question.id),
            baseline_results=self.store.baseline_results(),
            batch_size=batch_size,
        )

        # 4. Check for completion
        if strategy.is_question_answered(question, self.store):
            self._mark_completed(question)
            return self.next_experiments(batch_size)

        return experiments

    def _get_active_question(self) -> ResearchQuestion | None:
        """Highest priority question whose dependencies are met."""
        for q in sorted(self.agenda.questions, key=lambda q: q.priority):
            if q.status != "pending" and q.status != "active":
                continue
            if all(self._is_completed(dep) for dep in q.depends_on):
                q.status = "active"
                return q
        return None

    def _get_strategy(self, question) -> QuestionStrategy:
        """Return the decision strategy for this question type."""
        strategies = {
            "Q1_baselines": BaselineSweepStrategy(),
            "Q2_ca_viability": ViabilitySearchStrategy(),
            "Q3_ca_beats_baseline": FocusedOptimizationStrategy(),
            "Q4_live_ca": LiveCAComparisonStrategy(),
            "Q5_meta_learned": MetaLearningStrategy(),
            "Q6_transfer": TransferValidationStrategy(),
        }
        return strategies[question.id]
```

### 4. Question Strategies (`research/strategies/`)

Each question type gets a strategy class that encapsulates its decision logic.

```python
# research/strategies/base.py
class QuestionStrategy(ABC):
    @abstractmethod
    def propose_next(self, question, results_so_far, baseline_results, 
                     batch_size) -> list[ExperimentConfig]:
        """Generate the next batch of experiments."""

    @abstractmethod
    def is_question_answered(self, question, store) -> bool:
        """Check if we have enough evidence to answer the question."""

# research/strategies/viability.py
class ViabilitySearchStrategy(QuestionStrategy):
    """Q2: Explore broadly, abandon failures fast."""

    def propose_next(self, question, results_so_far, baseline_results, batch_size):
        # Which variants haven't been tested yet?
        tested_variants = {r["variant"] for r in results_so_far}
        all_variants = ["grid_ca", "neural_ca", "spectral_ca", "topo_ca", "reaction_diffusion"]
        abandoned = self._get_abandoned_variants(results_so_far)

        experiments = []
        for variant in all_variants:
            if variant in abandoned:
                continue

            variant_results = [r for r in results_so_far if r["variant"] == variant]

            if len(variant_results) == 0:
                # Never tested — start with 5 random configs
                for cfg in sample_random_ca_configs(variant, n=min(5, batch_size)):
                    experiments.append(make_experiment(variant, cfg, steps=1000))

            elif self._should_abandon(variant_results):
                abandoned.add(variant)
                continue

            elif self._is_viable(variant_results) and self._needs_confirmation(variant_results):
                # Viable but need more seeds to confirm
                best_cfg = self._best_config(variant_results)
                for seed in self._unused_seeds(variant_results, best_cfg):
                    experiments.append(make_experiment(variant, best_cfg, seed=seed, steps=1000))

            else:
                # Tested but not yet viable — try more configs
                for cfg in sample_random_ca_configs(variant, n=min(3, batch_size)):
                    experiments.append(make_experiment(variant, cfg, steps=1000))

            if len(experiments) >= batch_size:
                break

        return experiments[:batch_size]

    def _should_abandon(self, variant_results) -> bool:
        failures = sum(1 for r in variant_results 
                       if r["val_loss"] > 8.0 or math.isnan(r["val_loss"]))
        return failures > 3 and len(variant_results) >= 5

    def _is_viable(self, variant_results) -> bool:
        return any(r["val_loss"] < 4.0 for r in variant_results)

    def is_question_answered(self, question, store) -> bool:
        results = store.query(question_id=question.id)
        all_variants = ["grid_ca", "neural_ca", "spectral_ca", "topo_ca", "reaction_diffusion"]
        for variant in all_variants:
            vr = [r for r in results if r["variant"] == variant]
            if len(vr) == 0:
                return False  # untested variant
            if not (self._is_viable(vr) or self._should_abandon(vr)):
                return False  # undecided variant
        return True

# research/strategies/focused_optimization.py
class FocusedOptimizationStrategy(QuestionStrategy):
    """Q3: Bayesian optimization on viable variants."""

    def propose_next(self, question, results_so_far, baseline_results, batch_size):
        baseline_target = min(r["val_loss"] for r in baseline_results.values())
        viable_variants = self._get_viable_variants(results_so_far)

        experiments = []
        for variant in viable_variants:
            variant_results = [r for r in results_so_far if r["variant"] == variant]

            if len(variant_results) < 10:
                # Still in random exploration phase — propose random configs
                for cfg in sample_random_ca_configs(variant, n=batch_size):
                    experiments.append(make_experiment(variant, cfg, steps=3000))
            else:
                # Enough data for Bayesian optimization — propose BO suggestions
                study = self._get_or_create_optuna_study(variant, variant_results)
                for _ in range(batch_size):
                    trial = study.ask()
                    cfg = trial_to_config(trial, variant)
                    experiments.append(make_experiment(variant, cfg, steps=3000))

            # Check if any result already beats baseline
            best = min((r["val_loss"] for r in variant_results), default=float("inf"))
            if best < baseline_target:
                # Promising — add confirmation runs with more seeds
                best_cfg = self._best_config(variant_results)
                experiments.extend(self._confirmation_runs(variant, best_cfg))

        return experiments[:batch_size]

    def is_question_answered(self, question, store) -> bool:
        results = store.query(question_id=question.id)
        baseline_target = min(r["val_loss"] for r in store.baseline_results().values())

        for variant in self._get_viable_variants(results):
            vr = [r for r in results if r["variant"] == variant]
            best_cfg_results = self._results_for_best_config(vr)

            # Need at least 3 seeds of the best config
            if len(best_cfg_results) >= 3:
                mean_loss = np.mean([r["val_loss"] for r in best_cfg_results])
                if mean_loss < baseline_target * 0.95:  # 5% better
                    return True  # found a winner

                # Check convergence speed
                mean_steps = np.mean([r.get("steps_to_target", float("inf")) 
                                     for r in best_cfg_results])
                baseline_steps = np.mean([r.get("steps_to_target", float("inf"))
                                         for r in store.baseline_results().values()])
                if mean_steps < baseline_steps * 0.8:  # 20% faster
                    return True

        # Answered negatively if all variants exhausted their trial budget
        total_trials = len(results)
        if total_trials >= question.max_experiments:
            return True  # answered: "no, CA doesn't beat baseline"

        return False
```

### 5. Main Loop (`research/auto_research.py`)

The top-level runner that ties everything together.

```python
# research/auto_research.py

class AutoResearch:
    def __init__(self, agenda_path: str):
        self.agenda = load_agenda(agenda_path)
        self.store = ResultsStore()
        self.decision = DecisionEngine(self.agenda, self.store)
        self.runner = ExperimentRunner()
        self.reporter = ReportGenerator(self.store)

    def run(self):
        """Main auto-research loop."""
        print(f"Starting auto-research: {self.agenda.name}")
        print(f"Budget: {self.agenda.total_budget_hours}h")
        print(f"Questions: {len(self.agenda.questions)}")

        cycle = 0
        while True:
            cycle += 1

            # 1. Ask the decision engine what to run next
            experiments = self.decision.next_experiments(batch_size=5)

            if not experiments:
                print("All questions answered or budget exhausted.")
                break

            # 2. Run the batch
            print(f"\n--- Cycle {cycle}: running {len(experiments)} experiments ---")
            for exp in experiments:
                print(f"  Running: {exp.name} ({exp.variant}, {exp.steps} steps)")
                try:
                    result = self.runner.run(exp)
                    self.store.record(
                        experiment_id=exp.id,
                        config=exp.to_dict(),
                        metrics=result.metrics,
                        question_id=exp.question_id,
                        duration_seconds=result.duration,
                    )
                except Exception as e:
                    self.store.record(
                        experiment_id=exp.id,
                        config=exp.to_dict(),
                        metrics={"error": str(e), "val_loss": float("nan")},
                        question_id=exp.question_id,
                        duration_seconds=0,
                    )

            # 3. Check budget
            total_hours = self.store.budget_used()
            if total_hours >= self.agenda.total_budget_hours:
                print(f"Global budget exhausted ({total_hours:.1f}h used)")
                break

            # 4. Generate periodic report
            if cycle % self.agenda.report_every_n_experiments == 0:
                self.reporter.generate_progress_report(cycle)

            # 5. Print status
            self._print_status()

        # Final report
        self.reporter.generate_final_report()

    def _print_status(self):
        hours = self.store.budget_used()
        n_exp = self.store.experiment_count()
        for q in self.agenda.questions:
            qcount = self.store.experiment_count(q.id)
            print(f"  {q.id}: {q.status} ({qcount} experiments)")
        print(f"  Total: {n_exp} experiments, {hours:.1f}h used / {self.agenda.total_budget_hours}h budget")
```

### 6. CLI Entry Point

```python
# scripts/run_auto_research.py
"""
Usage:
    python scripts/run_auto_research.py                          # uses default agenda
    python scripts/run_auto_research.py --agenda my_agenda.yaml  # custom agenda
    python scripts/run_auto_research.py --resume                 # continue from last state
    python scripts/run_auto_research.py --status                 # print current status
    python scripts/run_auto_research.py --report                 # generate report from existing results
"""
```

---

## Files to Create

```
research/
├── agenda.yaml              # default research agenda (the one above)
├── agenda.py                # ResearchAgenda, ResearchQuestion dataclasses
├── auto_research.py         # main loop (AutoResearch class)
├── results_store.py         # persistent experiment results (SQLite)
├── decision_engine.py       # decides what to run next
├── strategies/
│   ├── __init__.py
│   ├── base.py              # QuestionStrategy ABC
│   ├── baseline_sweep.py    # Q1: exhaustive sweep
│   ├── viability.py         # Q2: broad search, fast abandonment
│   ├── focused_optimization.py  # Q3: Bayesian optimization
│   ├── live_ca_comparison.py    # Q4: live CA rules comparison
│   ├── meta_learning.py     # Q5: CMA-ES with checkpointing
│   └── transfer_validation.py   # Q6: simple scale-up test
├── experiment_generator.py  # creates ExperimentConfig from strategy decisions
└── report.py                # (already exists, extend for auto-research reports)

scripts/
├── run_auto_research.py     # CLI entry point
```

## Files to Modify

- **`research/engine.py`** — Refactor the existing runner to implement the `ExperimentRunner` interface. The runner should accept an `ExperimentConfig` programmatically (not just YAML path). Keep YAML support for manual runs but add `runner.run(config: ExperimentConfig) -> ExperimentResult`.

- **`NEUROGEN_PROJECT.md`** — Replace Phase descriptions (1-8) with a note that phases are now auto-managed by the research agenda. Keep the phase content as reference for what the engine does internally.

- **`pyproject.toml`** — No new dependencies needed. Optuna is already in `[search]` extras. SQLite is stdlib.

## Implementation Order for Claude Code

1. `research/agenda.py` — dataclasses for agenda and questions
2. `research/results_store.py` — SQLite-backed results store
3. `research/strategies/base.py` — ABC for question strategies
4. `research/strategies/baseline_sweep.py` — simplest strategy (Q1)
5. `research/experiment_generator.py` — creates configs from strategy decisions
6. Refactor `research/engine.py` — add programmatic `run(config)` interface
7. `research/decision_engine.py` — dispatches to strategies
8. `research/strategies/viability.py` — Q2 strategy
9. `research/strategies/focused_optimization.py` — Q3 strategy
10. `research/auto_research.py` — main loop
11. `scripts/run_auto_research.py` — CLI
12. `research/strategies/live_ca_comparison.py` — Q4
13. `research/strategies/meta_learning.py` — Q5
14. `research/strategies/transfer_validation.py` — Q6
15. `research/agenda.yaml` — default agenda
16. Tests for all of the above

## Key Design Rules

- **The decision engine never invents new model architectures or CA variants.** It only selects configurations from the defined search spaces. The human defines the space; the engine searches it.
- **Every decision is logged.** The results store records not just metrics but also *why* each experiment was chosen (which strategy, what it was comparing against, what the current best is).
- **Resumable by default.** The results store is persistent. If you kill the process and restart, it picks up where it left off by querying the store for what's already been done.
- **Budget is a hard constraint.** The engine tracks wall-clock time per experiment and refuses to start new runs if the remaining budget can't cover the estimated cost.
- **Questions have dependencies.** Q3 can't start until Q2 is answered. This is enforced in the decision engine, not by the human manually sequencing things.
- **Negative results are real answers.** If Q3 exhausts its budget without finding a CA that beats baseline, that's a valid answer: "No, with this search budget, no CA init beats standard initialization." The engine marks it completed, not failed.

## Tests to Add

```
tests/test_auto_research.py

test_agenda_loading
test_results_store_record_and_query
test_results_store_budget_tracking
test_decision_engine_respects_dependencies
test_decision_engine_baseline_sweep_generates_all
test_decision_engine_viability_abandons_bad_variants
test_decision_engine_stops_when_budget_exhausted
test_decision_engine_stops_when_questions_answered
test_strategy_viability_propose_next
test_strategy_focused_optuna_integration
test_auto_research_loop_tiny_agenda  # 2 questions, 10 experiments, <60s
test_resume_from_partial_results
```
