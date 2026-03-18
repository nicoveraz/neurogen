"""Tests for the auto-research system: agenda, results store, decision engine,
experiment generator, and question strategies.

All tests run without network access, use tiny configs, and rely on tmp_path
for file operations.
"""

from __future__ import annotations

import math

import pytest
import yaml

from research.agenda import ResearchAgenda, ResearchQuestion, load_agenda
from research.results_store import ResultsStore
from research.experiment_generator import (
    ALL_BASELINES,
    ALL_CA_VARIANTS,
    DEFAULT_SEEDS,
    make_experiment,
    make_experiment_id,
    sample_random_ca_configs,
)
from research.decision_engine import DecisionEngine
from research.strategies.baseline_sweep import BaselineSweepStrategy
from research.strategies.viability import ViabilitySearchStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_two_question_agenda(
    q1_status: str = "pending",
    q2_status: str = "pending",
    total_budget_hours: float = 80.0,
) -> ResearchAgenda:
    """Create a minimal 2-question agenda (Q1 + Q2 depends_on Q1)."""
    return ResearchAgenda(
        name="Test Agenda",
        hardware="test",
        total_budget_hours=total_budget_hours,
        questions=[
            ResearchQuestion(
                id="Q1_baselines",
                question="Baseline sweep",
                priority=1,
                depends_on=[],
                max_hours=2.0,
                max_experiments=30,
                status=q1_status,
            ),
            ResearchQuestion(
                id="Q2_ca_viability",
                question="CA viability",
                priority=2,
                depends_on=["Q1_baselines"],
                max_hours=3.0,
                max_experiments=50,
                status=q2_status,
            ),
        ],
    )


def _make_store(tmp_path) -> ResultsStore:
    """Create a ResultsStore backed by a temporary database."""
    return ResultsStore(path=str(tmp_path / "test.db"))


def _record_baseline(
    store: ResultsStore,
    init_method: str,
    seed: int,
    val_loss: float,
    duration: float = 60.0,
) -> None:
    """Record a single baseline experiment result."""
    store.record(
        experiment_id=make_experiment_id(),
        config={"init_method": init_method, "seed": seed},
        metrics={"val_loss": val_loss},
        question_id="Q1_baselines",
        duration_seconds=duration,
    )


def _record_ca(
    store: ResultsStore,
    variant: str,
    seed: int,
    val_loss: float | None,
    question_id: str = "Q2_ca_viability",
    duration: float = 60.0,
    ca_config: dict | None = None,
) -> None:
    """Record a single CA experiment result."""
    metrics: dict = {}
    if val_loss is not None:
        metrics["val_loss"] = val_loss
    store.record(
        experiment_id=make_experiment_id(),
        config={
            "init_method": variant,
            "variant": variant,
            "seed": seed,
            "ca_config": ca_config or {},
        },
        metrics=metrics,
        question_id=question_id,
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# TestAgenda
# ---------------------------------------------------------------------------

class TestAgenda:
    def test_agenda_loading(self, tmp_path):
        """Load the default agenda.yaml and verify structure."""
        import os
        agenda_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "research", "agenda.yaml",
        )
        if not os.path.exists(agenda_path):
            pytest.skip("Default agenda.yaml not found")

        agenda = load_agenda(agenda_path)

        assert isinstance(agenda, ResearchAgenda)
        assert agenda.name == "NeuroGen CA Investigation"
        assert agenda.hardware == "macbook_m1pro_16gb"
        assert agenda.total_budget_hours == 80.0
        assert len(agenda.questions) == 6

        # Verify Q1 basics
        q1 = agenda.questions[0]
        assert q1.id == "Q1_baselines"
        assert q1.priority == 1
        assert q1.depends_on == []
        assert q1.max_hours == 2

        # Verify Q2 depends on Q1
        q2 = agenda.questions[1]
        assert q2.id == "Q2_ca_viability"
        assert q2.depends_on == ["Q1_baselines"]

    def test_agenda_loading_custom(self, tmp_path):
        """Write a minimal custom agenda YAML, load it, verify."""
        agenda_data = {
            "name": "Custom Test Agenda",
            "hardware": "test_hw",
            "total_budget_hours": 10,
            "report_every_n_experiments": 3,
            "questions": [
                {
                    "id": "QA",
                    "question": "Is this a test?",
                    "success": "Yes",
                    "metric": "val_loss",
                    "priority": 1,
                    "depends_on": [],
                    "max_hours": 1.0,
                    "max_experiments": 5,
                    "threshold": 2.0,
                },
            ],
        }
        path = tmp_path / "custom_agenda.yaml"
        with open(path, "w") as f:
            yaml.dump(agenda_data, f)

        agenda = load_agenda(str(path))
        assert agenda.name == "Custom Test Agenda"
        assert agenda.hardware == "test_hw"
        assert agenda.total_budget_hours == 10
        assert agenda.report_every_n_cycles == 3
        assert len(agenda.questions) == 1

        q = agenda.questions[0]
        assert q.id == "QA"
        assert q.question == "Is this a test?"
        assert q.threshold == 2.0
        assert q.depends_on == []
        assert q.max_experiments == 5

    def test_agenda_missing_file(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_agenda("/nonexistent/path/agenda.yaml")

    def test_research_question_defaults(self):
        """ResearchQuestion has correct defaults."""
        q = ResearchQuestion()
        assert q.id == ""
        assert q.question == ""
        assert q.success == ""
        assert q.metric == "val_loss"
        assert q.priority == 1
        assert q.depends_on == []
        assert q.max_hours == 2.0
        assert q.max_experiments == 30
        assert q.threshold is None
        assert q.comparison is None
        assert q.status == "pending"
        assert q.answer == ""


# ---------------------------------------------------------------------------
# TestResultsStore
# ---------------------------------------------------------------------------

class TestResultsStore:
    def test_record_and_query(self, tmp_path):
        """Record an experiment, query it back."""
        store = _make_store(tmp_path)
        store.record(
            experiment_id="exp001",
            config={"init_method": "xavier_normal", "seed": 42},
            metrics={"val_loss": 2.5, "train_loss": 2.0},
            question_id="Q1_baselines",
            duration_seconds=120.0,
        )

        results = store.query(question_id="Q1_baselines")
        assert len(results) == 1

        r = results[0]
        assert r["experiment_id"] == "exp001"
        assert r["question_id"] == "Q1_baselines"
        assert r["val_loss"] == 2.5
        assert r["train_loss"] == 2.0
        assert r["init_method"] == "xavier_normal"
        assert r["seed"] == 42
        assert r["duration_seconds"] == 120.0
        assert r["config"]["init_method"] == "xavier_normal"
        assert r["metrics"]["val_loss"] == 2.5
        store.close()

    def test_budget_tracking(self, tmp_path):
        """budget_used() returns correct hours."""
        store = _make_store(tmp_path)

        # No experiments yet
        assert store.budget_used() == 0.0
        assert store.budget_used("Q1_baselines") == 0.0

        # Record 3600 seconds = 1 hour for Q1
        store.record(
            experiment_id="exp_budget_1",
            config={"init_method": "xavier_normal", "seed": 42},
            metrics={"val_loss": 2.5},
            question_id="Q1_baselines",
            duration_seconds=3600.0,
        )
        assert abs(store.budget_used("Q1_baselines") - 1.0) < 1e-6
        assert abs(store.budget_used() - 1.0) < 1e-6

        # Record 1800 seconds = 0.5 hours for Q2
        store.record(
            experiment_id="exp_budget_2",
            config={"init_method": "grid_ca", "seed": 42},
            metrics={"val_loss": 3.0},
            question_id="Q2_ca_viability",
            duration_seconds=1800.0,
        )
        assert abs(store.budget_used("Q2_ca_viability") - 0.5) < 1e-6
        assert abs(store.budget_used() - 1.5) < 1e-6
        store.close()

    def test_experiment_count(self, tmp_path):
        """experiment_count() with and without question filter."""
        store = _make_store(tmp_path)
        assert store.experiment_count() == 0
        assert store.experiment_count("Q1_baselines") == 0

        _record_baseline(store, "xavier_normal", 42, 2.5)
        _record_baseline(store, "kaiming_normal", 42, 2.6)
        _record_ca(store, "grid_ca", 42, 3.0)

        assert store.experiment_count() == 3
        assert store.experiment_count("Q1_baselines") == 2
        assert store.experiment_count("Q2_ca_viability") == 1
        store.close()

    def test_best_result(self, tmp_path):
        """best_result() returns lowest val_loss."""
        store = _make_store(tmp_path)

        # No results yet
        assert store.best_result("Q1_baselines") is None

        _record_baseline(store, "xavier_normal", 42, 2.5)
        _record_baseline(store, "kaiming_normal", 42, 2.2)
        _record_baseline(store, "orthogonal", 42, 2.8)

        best = store.best_result("Q1_baselines", metric="val_loss")
        assert best is not None
        assert best["val_loss"] == 2.2
        assert best["init_method"] == "kaiming_normal"
        store.close()

    def test_best_baseline_loss(self, tmp_path):
        """best_baseline_loss() from Q1 results."""
        store = _make_store(tmp_path)

        # No baselines yet -- should return inf
        assert store.best_baseline_loss() == float("inf")

        _record_baseline(store, "xavier_normal", 42, 2.5)
        _record_baseline(store, "kaiming_normal", 42, 2.1)
        _record_baseline(store, "orthogonal", 42, 2.3)

        assert store.best_baseline_loss() == 2.1
        store.close()

    def test_query_by_variant(self, tmp_path):
        """Query filtered by variant."""
        store = _make_store(tmp_path)
        _record_ca(store, "grid_ca", 42, 3.0)
        _record_ca(store, "neural_ca", 42, 3.5)
        _record_ca(store, "grid_ca", 137, 2.8)

        grid_results = store.query(variant="grid_ca")
        assert len(grid_results) == 2
        for r in grid_results:
            assert r["variant"] == "grid_ca"

        neural_results = store.query(variant="neural_ca")
        assert len(neural_results) == 1
        assert neural_results[0]["variant"] == "neural_ca"
        store.close()

    def test_nan_handling(self, tmp_path):
        """NaN val_loss is stored as NULL and handled correctly."""
        store = _make_store(tmp_path)
        store.record(
            experiment_id="exp_nan",
            config={"init_method": "grid_ca", "seed": 42},
            metrics={"val_loss": float("nan")},
            question_id="Q2_ca_viability",
            duration_seconds=60.0,
        )

        results = store.query(question_id="Q2_ca_viability")
        assert len(results) == 1
        # NaN val_loss should be stored as None (SQL NULL)
        assert results[0]["val_loss"] is None

        # best_baseline_loss should not be affected by NaN
        assert store.best_baseline_loss() == float("inf")
        store.close()

    def test_persistence(self, tmp_path):
        """Close and reopen store, data persists."""
        db_path = str(tmp_path / "persist.db")

        store1 = ResultsStore(path=db_path)
        store1.record(
            experiment_id="exp_persist",
            config={"init_method": "xavier_normal", "seed": 42},
            metrics={"val_loss": 2.5},
            question_id="Q1_baselines",
            duration_seconds=100.0,
        )
        store1.close()

        # Reopen and verify data is still there
        store2 = ResultsStore(path=db_path)
        assert store2.experiment_count() == 1
        results = store2.query(question_id="Q1_baselines")
        assert len(results) == 1
        assert results[0]["experiment_id"] == "exp_persist"
        assert results[0]["val_loss"] == 2.5
        store2.close()


# ---------------------------------------------------------------------------
# TestExperimentGenerator
# ---------------------------------------------------------------------------

class TestExperimentGenerator:
    def test_make_experiment(self):
        """make_experiment() produces valid config dict."""
        exp = make_experiment(
            init_method="xavier_normal",
            question_id="Q1_baselines",
            steps=500,
            seed=42,
            reason="test",
        )
        assert exp["init_method"] == "xavier_normal"
        assert exp["question_id"] == "Q1_baselines"
        assert exp["seed"] == 42
        assert exp["steps"] == 500
        assert exp["variant"] == "xavier_normal"
        assert exp["reason"] == "test"
        assert "experiment_id" in exp
        assert len(exp["experiment_id"]) == 12
        assert exp["model_config"] is not None
        assert exp["train_config"] is not None
        assert isinstance(exp["ca_config"], dict)

    def test_make_experiment_id_unique(self):
        """IDs are unique."""
        ids = {make_experiment_id() for _ in range(100)}
        assert len(ids) == 100, "All 100 generated IDs should be unique"

    def test_sample_random_ca_configs(self):
        """sample_random_ca_configs produces N configs with correct variant."""
        n = 7
        configs = sample_random_ca_configs(
            variant="neural_ca",
            n=n,
            question_id="Q2_ca_viability",
        )
        assert len(configs) == n
        for cfg in configs:
            assert cfg["init_method"] == "neural_ca"
            assert cfg["variant"] == "neural_ca"
            assert cfg["question_id"] == "Q2_ca_viability"
            assert "ca_config" in cfg
            # neural_ca should have n_channels and p_update
            assert "n_channels" in cfg["ca_config"]
            assert "p_update" in cfg["ca_config"]

    def test_all_baselines_listed(self):
        """ALL_BASELINES has 9 entries."""
        assert len(ALL_BASELINES) == 9
        expected = {
            "xavier_uniform", "xavier_normal", "kaiming_uniform",
            "kaiming_normal", "orthogonal", "sparse", "fixup",
            "mimetic", "spectral_delta",
        }
        assert set(ALL_BASELINES) == expected

    def test_all_ca_variants_listed(self):
        """ALL_CA_VARIANTS has 5 entries."""
        assert len(ALL_CA_VARIANTS) == 5
        expected = {
            "grid_ca", "neural_ca", "spectral_ca",
            "topo_ca", "reaction_diffusion",
        }
        assert set(ALL_CA_VARIANTS) == expected


# ---------------------------------------------------------------------------
# TestDecisionEngine
# ---------------------------------------------------------------------------

class TestDecisionEngine:
    def test_respects_dependencies(self, tmp_path):
        """Q2 doesn't start before Q1 is completed."""
        agenda = _make_two_question_agenda()
        store = _make_store(tmp_path)
        engine = DecisionEngine(agenda, store)

        # First call should target Q1, not Q2
        experiments = engine.next_experiments(batch_size=3)
        assert len(experiments) > 0
        for exp in experiments:
            assert exp["question_id"] == "Q1_baselines"

        # Q2 should still be pending because Q1 is not completed
        q2 = agenda.questions[1]
        assert q2.status == "pending"
        store.close()

    def test_baseline_sweep_generates_all(self, tmp_path):
        """Q1 strategy generates experiments for all baselines x seeds."""
        agenda = _make_two_question_agenda()
        store = _make_store(tmp_path)
        engine = DecisionEngine(agenda, store)

        # Collect all proposed experiments across multiple batches
        all_experiments = []
        for _ in range(100):  # generous iteration limit
            batch = engine.next_experiments(batch_size=10)
            if not batch:
                break
            for exp in batch:
                if exp["question_id"] != "Q1_baselines":
                    break
                # Simulate recording the result
                store.record(
                    experiment_id=exp["experiment_id"],
                    config={
                        "init_method": exp["init_method"],
                        "seed": exp["seed"],
                    },
                    metrics={"val_loss": 2.5},
                    question_id=exp["question_id"],
                    duration_seconds=10.0,
                )
                all_experiments.append(exp)

        # Should have proposed all 9 baselines x 3 seeds = 27
        combos = {(e["init_method"], e["seed"]) for e in all_experiments}
        expected_total = len(ALL_BASELINES) * len(DEFAULT_SEEDS)
        assert len(combos) == expected_total, (
            f"Expected {expected_total} unique (init, seed) combos, got {len(combos)}"
        )
        store.close()

    def test_stops_when_budget_exhausted(self, tmp_path):
        """Returns empty list when budget is used up."""
        agenda = _make_two_question_agenda(total_budget_hours=0.001)
        store = _make_store(tmp_path)
        engine = DecisionEngine(agenda, store)

        # Record enough to exceed the tiny budget
        store.record(
            experiment_id="exp_budget",
            config={"init_method": "xavier_normal", "seed": 42},
            metrics={"val_loss": 2.5},
            question_id="Q1_baselines",
            duration_seconds=36000.0,  # 10 hours
        )

        experiments = engine.next_experiments()
        assert experiments == [], "Should return empty when global budget exhausted"
        store.close()

    def test_stops_when_questions_answered(self, tmp_path):
        """Returns empty list when all questions completed."""
        agenda = _make_two_question_agenda(
            q1_status="completed",
            q2_status="completed",
        )
        store = _make_store(tmp_path)
        engine = DecisionEngine(agenda, store)

        experiments = engine.next_experiments()
        assert experiments == [], "Should return empty when all questions completed"
        store.close()

    def test_marks_completed_automatically(self, tmp_path):
        """When a strategy says answered, question gets marked completed."""
        agenda = _make_two_question_agenda()
        store = _make_store(tmp_path)
        engine = DecisionEngine(agenda, store)

        # Record all 9 baselines x 3 seeds to complete Q1
        for init_method in ALL_BASELINES:
            for seed in DEFAULT_SEEDS:
                _record_baseline(store, init_method, seed, 2.5, duration=1.0)

        # Next call should see Q1 answered and mark it completed,
        # then move on to Q2
        experiments = engine.next_experiments(batch_size=3)

        q1 = agenda.questions[0]
        assert q1.status == "completed", (
            f"Q1 should be marked completed, got '{q1.status}'"
        )

        # Should now be proposing Q2 experiments (or empty if Q2 also done)
        if experiments:
            assert experiments[0]["question_id"] == "Q2_ca_viability"
        store.close()


# ---------------------------------------------------------------------------
# TestViabilityStrategy
# ---------------------------------------------------------------------------

class TestViabilityStrategy:
    def test_propose_initial(self):
        """First call proposes configs for untested variants."""
        strategy = ViabilitySearchStrategy()
        question = ResearchQuestion(
            id="Q2_ca_viability",
            priority=2,
            max_experiments=50,
        )

        experiments = strategy.propose_next(
            question=question,
            results_so_far=[],
            baseline_results=[],
            batch_size=10,
        )

        assert len(experiments) > 0
        # Should propose experiments for untested variants
        proposed_variants = {e["variant"] for e in experiments}
        assert len(proposed_variants) >= 1
        # All proposed variants should be from the known list
        for v in proposed_variants:
            assert v in ALL_CA_VARIANTS

    def test_abandons_bad_variants(self):
        """Variant with >3 failures gets abandoned."""
        strategy = ViabilitySearchStrategy()
        question = ResearchQuestion(
            id="Q2_ca_viability",
            priority=2,
            max_experiments=50,
        )

        # Create results where grid_ca has 5 experiments, all with loss > 8
        bad_results = []
        for i in range(5):
            bad_results.append({
                "variant": "grid_ca",
                "init_method": "grid_ca",
                "val_loss": 10.0,
                "seed": i,
                "config": {"ca_config": {}},
                "metrics": {"val_loss": 10.0},
            })

        experiments = strategy.propose_next(
            question=question,
            results_so_far=bad_results,
            baseline_results=[],
            batch_size=10,
        )

        # grid_ca should be abandoned, so no new experiments for it
        grid_ca_exps = [e for e in experiments if e["variant"] == "grid_ca"]
        assert len(grid_ca_exps) == 0, (
            "Abandoned variant should not get new experiments"
        )

    def test_confirms_viable(self):
        """Viable variant gets confirmation runs."""
        strategy = ViabilitySearchStrategy()
        question = ResearchQuestion(
            id="Q2_ca_viability",
            priority=2,
            max_experiments=50,
        )

        # One good result for neural_ca (below VIABILITY_LOSS=4.0)
        viable_results = [
            {
                "variant": "neural_ca",
                "init_method": "neural_ca",
                "val_loss": 3.0,
                "seed": 999,
                "config": {"ca_config": {"hidden_dim": 64}},
                "metrics": {"val_loss": 3.0},
                "ca_config": {"hidden_dim": 64},
            },
        ]

        experiments = strategy.propose_next(
            question=question,
            results_so_far=viable_results,
            baseline_results=[],
            batch_size=10,
        )

        # Should include confirmation experiments for neural_ca
        neural_exps = [e for e in experiments if e["variant"] == "neural_ca"]
        assert len(neural_exps) > 0, (
            "Viable variant should get confirmation experiments"
        )


# ---------------------------------------------------------------------------
# TestBaselineSweepStrategy
# ---------------------------------------------------------------------------

class TestBaselineSweepStrategy:
    def test_proposes_all_baselines(self):
        """Proposes all 9 baselines x 3 seeds."""
        strategy = BaselineSweepStrategy()
        question = ResearchQuestion(
            id="Q1_baselines",
            priority=1,
            max_experiments=30,
        )

        # Collect all proposals across multiple batches
        all_experiments = []
        results_so_far: list[dict] = []
        for _ in range(30):
            batch = strategy.propose_next(
                question=question,
                results_so_far=results_so_far,
                baseline_results=results_so_far,
                batch_size=5,
            )
            if not batch:
                break
            for exp in batch:
                results_so_far.append({
                    "init_method": exp["init_method"],
                    "seed": exp["seed"],
                })
                all_experiments.append(exp)

        combos = {(e["init_method"], e["seed"]) for e in all_experiments}
        expected = len(ALL_BASELINES) * len(DEFAULT_SEEDS)
        assert len(combos) == expected, (
            f"Expected {expected} unique combos, got {len(combos)}"
        )

    def test_skips_completed(self):
        """Doesn't re-propose already completed experiments."""
        strategy = BaselineSweepStrategy()
        question = ResearchQuestion(
            id="Q1_baselines",
            priority=1,
            max_experiments=30,
        )

        # Simulate that xavier_normal with seed 42 is already done
        done = [{"init_method": "xavier_normal", "seed": 42}]

        experiments = strategy.propose_next(
            question=question,
            results_so_far=done,
            baseline_results=done,
            batch_size=30,
        )

        proposed_combos = {(e["init_method"], e["seed"]) for e in experiments}
        assert ("xavier_normal", 42) not in proposed_combos, (
            "Already completed experiment should not be re-proposed"
        )

        # Total should be all combos minus the one already done
        total_expected = len(ALL_BASELINES) * len(DEFAULT_SEEDS) - 1
        assert len(proposed_combos) == total_expected

    def test_is_answered_when_all_done(self, tmp_path):
        """is_question_answered True when all baselines x seeds are done."""
        strategy = BaselineSweepStrategy()
        question = ResearchQuestion(
            id="Q1_baselines",
            priority=1,
            max_experiments=30,
        )
        store = _make_store(tmp_path)

        # Not answered yet -- empty store
        assert not strategy.is_question_answered(question, store)

        # Record all 9 baselines x 3 seeds
        for init_method in ALL_BASELINES:
            for seed in DEFAULT_SEEDS:
                _record_baseline(store, init_method, seed, 2.5)

        assert strategy.is_question_answered(question, store), (
            "Should be answered when all baseline x seed combos are recorded"
        )
        store.close()
