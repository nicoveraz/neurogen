"""Q4: Live CA comparison strategy.

Tests each live CA rule (applied during training) against the
init-only baseline from Q3. Explores different alpha schedules
for promising rules.
"""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

from research.strategies.base import QuestionStrategy
from research.experiment_generator import DEFAULT_SEEDS, make_live_ca_experiment

if TYPE_CHECKING:
    from research.agenda import ResearchQuestion
    from research.results_store import ResultsStore


ALL_LIVE_CA_RULES: list[str] = [
    "local_norm",
    "modularity",
    "pruning",
    "competition",
    "learned",
]

ALL_ALPHA_SCHEDULES: list[str] = [
    "exponential_decay",
    "cosine",
    "phased",
    "adaptive",
    "cyclic",
]


class LiveCAComparisonStrategy(QuestionStrategy):
    """Test each live CA rule against init-only baseline."""

    def propose_next(
        self,
        question: ResearchQuestion,
        results_so_far: list[dict],
        baseline_results: list[dict],
        batch_size: int = 5,
    ) -> list[dict]:
        """Propose live CA experiments: test rules, then alpha schedules.

        First tests each live CA rule with default alpha schedule and
        multiple seeds. For promising rules, explores different alpha
        schedules.

        Args:
            question: The research question being investigated.
            results_so_far: All results for this question from the store.
            baseline_results: Q1 baseline results for comparison.
            batch_size: Maximum number of experiments to propose.

        Returns:
            List of experiment config dicts for the next batch.
        """
        # Find best init-only CA method from Q3
        best_init = self._best_init_method(baseline_results, results_so_far)

        experiments: list[dict] = []
        for rule in ALL_LIVE_CA_RULES:
            rule_results = [
                r
                for r in results_so_far
                if r.get("config", {}).get("ca_config", {}).get("live_ca_rule")
                == rule
            ]

            if len(rule_results) == 0:
                # Never tested -- try it with multiple seeds
                for seed in DEFAULT_SEEDS[
                    : min(3, batch_size - len(experiments))
                ]:
                    experiments.append(make_live_ca_experiment(
                        ca_rule=rule,
                        init_method=best_init,
                        question_id=question.id,
                        steps=3000,
                        seed=seed,
                        reason=f"testing live CA rule: {rule}",
                    ))
            elif self._is_promising(rule_results):
                # Try different alpha schedules
                tested_schedules = {
                    r.get("config", {})
                    .get("ca_config", {})
                    .get("alpha_schedule")
                    for r in rule_results
                }
                for sched in ALL_ALPHA_SCHEDULES:
                    if (
                        sched not in tested_schedules
                        and len(experiments) < batch_size
                    ):
                        experiments.append(make_live_ca_experiment(
                            ca_rule=rule,
                            init_method=best_init,
                            question_id=question.id,
                            alpha_schedule=sched,
                            steps=3000,
                            seed=42,
                            reason=f"testing {rule} with {sched} schedule",
                        ))

            if len(experiments) >= batch_size:
                break

        return experiments[:batch_size]

    def is_question_answered(
        self,
        question: ResearchQuestion,
        store: ResultsStore,
    ) -> bool:
        """Check if all live CA rules have been tested.

        Answered when every rule has at least one result, or the
        experiment budget is exhausted.

        Args:
            question: The research question.
            store: The results store for querying.

        Returns:
            True if the question can be considered answered.
        """
        results = store.query(question_id=question.id)
        if not results:
            return False

        tested_rules = {
            r.get("config", {}).get("ca_config", {}).get("live_ca_rule")
            for r in results
        }

        if len(tested_rules) >= len(ALL_LIVE_CA_RULES):
            return True
        if store.experiment_count(question.id) >= question.max_experiments:
            return True
        return False

    def _best_init_method(
        self,
        baseline_results: list[dict],
        q3_results: list[dict],
    ) -> str:
        """Find best init method from Q3 or fall back to best baseline.

        Args:
            baseline_results: Q1 baseline result dicts.
            q3_results: Q3 focused optimization result dicts.

        Returns:
            Name of the best initialization method found.
        """
        valid_q3 = [
            r
            for r in q3_results
            if r.get("val_loss") is not None
            and isinstance(r.get("val_loss"), (int, float))
            and not math.isnan(r["val_loss"])
        ]
        if valid_q3:
            best = min(valid_q3, key=lambda r: r["val_loss"])
            return best.get("init_method", "xavier_normal")

        valid_bl = [
            r
            for r in baseline_results
            if r.get("val_loss") is not None
            and isinstance(r.get("val_loss"), (int, float))
            and not math.isnan(r["val_loss"])
        ]
        if valid_bl:
            best = min(valid_bl, key=lambda r: r["val_loss"])
            return best.get("init_method", "xavier_normal")

        return "xavier_normal"

    def _is_promising(self, rule_results: list[dict]) -> bool:
        """Check if a live CA rule shows promise (loss below 4.0).

        Args:
            rule_results: Results for a single live CA rule.

        Returns:
            True if at least one result has val_loss below 4.0.
        """
        valid = [
            r
            for r in rule_results
            if r.get("val_loss") is not None
            and isinstance(r.get("val_loss"), (int, float))
            and not math.isnan(r["val_loss"])
        ]
        if not valid:
            return False
        best = min(r["val_loss"] for r in valid)
        return best < 4.0
