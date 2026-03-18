"""Q3: Focused optimization to beat baselines.

Takes viable CA variants from Q2 and optimizes their hyperparameters
to beat the best baseline from Q1. Uses random search (with Optuna
as a future upgrade path). Confirms winners with multiple seeds.
"""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

from research.strategies.base import QuestionStrategy
from research.experiment_generator import (
    ALL_CA_VARIANTS,
    DEFAULT_SEEDS,
    make_experiment,
    sample_random_ca_configs,
)

if TYPE_CHECKING:
    from research.agenda import ResearchQuestion
    from research.results_store import ResultsStore


class FocusedOptimizationStrategy(QuestionStrategy):
    """Bayesian/random optimization on viable variants to beat baseline."""

    def propose_next(
        self,
        question: ResearchQuestion,
        results_so_far: list[dict],
        baseline_results: list[dict],
        batch_size: int = 5,
    ) -> list[dict]:
        """Propose experiments that focus on beating the best baseline.

        For each viable variant, checks if any config already beats
        the baseline. If so, runs confirmation seeds. Otherwise,
        samples new random configs with longer training runs.

        Args:
            question: The research question being investigated.
            results_so_far: All results for this question from the store.
            baseline_results: Q1 baseline results for comparison.
            batch_size: Maximum number of experiments to propose.

        Returns:
            List of experiment config dicts for the next batch.
        """
        baseline_target = self._best_baseline_loss(baseline_results)
        viable = self._get_viable_variants(results_so_far, baseline_results)

        if not viable:
            # Fall back to first few CA variants
            viable = ALL_CA_VARIANTS[:3]

        experiments: list[dict] = []
        for variant in viable:
            variant_results = [
                r for r in results_so_far if r.get("variant") == variant
            ]

            # Check if any result already beats baseline
            valid_results = [
                r
                for r in variant_results
                if r.get("val_loss") is not None
                and isinstance(r.get("val_loss"), (int, float))
                and not math.isnan(r["val_loss"])
            ]

            if valid_results:
                best = min(valid_results, key=lambda r: r["val_loss"])
                if best["val_loss"] < baseline_target:
                    # Promising -- add confirmation runs with different seeds
                    used_seeds = {r.get("seed") for r in variant_results}
                    for seed in DEFAULT_SEEDS:
                        if seed not in used_seeds and len(experiments) < batch_size:
                            experiments.append(make_experiment(
                                init_method=variant,
                                question_id=question.id,
                                steps=3000,
                                seed=seed,
                                ca_config=best.get(
                                    "ca_config",
                                    best.get("config", {}).get("ca_config", {}),
                                ),
                                reason=f"confirming {variant} beats baseline",
                            ))

            # Also propose new random configs (more steps than Q2)
            remaining = batch_size - len(experiments)
            if remaining > 0:
                new_exps = sample_random_ca_configs(
                    variant,
                    n=min(remaining, 3),
                    question_id=question.id,
                )
                for e in new_exps:
                    e["steps"] = 3000
                    e["train_config"].max_steps = 3000
                    e["train_config"].eval_interval = 300
                experiments.extend(new_exps)

            if len(experiments) >= batch_size:
                break

        return experiments[:batch_size]

    def is_question_answered(
        self,
        question: ResearchQuestion,
        store: ResultsStore,
    ) -> bool:
        """Check if a CA variant statistically beats the best baseline.

        Answered positively if any variant's mean loss (over 3+ seeds) is
        at least 5% better than the best baseline. Answered negatively if
        the experiment budget is exhausted without meeting this criterion.

        Args:
            question: The research question.
            store: The results store for querying.

        Returns:
            True if the question can be considered answered.
        """
        results = store.query(question_id=question.id)
        baseline_target = store.best_baseline_loss()

        if baseline_target == float("inf"):
            return False

        # Check if any variant beats baseline with statistical confirmation
        for variant in ALL_CA_VARIANTS:
            vr = [r for r in results if r.get("variant") == variant]
            valid = [
                r
                for r in vr
                if r.get("val_loss") is not None
                and isinstance(r.get("val_loss"), (int, float))
                and not math.isnan(r["val_loss"])
            ]

            if len(valid) >= 3:
                mean_loss = sum(r["val_loss"] for r in valid) / len(valid)
                if mean_loss < baseline_target * 0.95:  # 5% better
                    return True

        # Answered negatively if budget exhausted
        if store.experiment_count(question.id) >= question.max_experiments:
            return True

        return False

    def _best_baseline_loss(self, baseline_results: list[dict]) -> float:
        """Find the best val_loss from baseline results.

        Args:
            baseline_results: Q1 baseline result dicts.

        Returns:
            Lowest val_loss, or inf if no valid results.
        """
        valid = [
            r.get("val_loss")
            for r in baseline_results
            if r.get("val_loss") is not None
            and isinstance(r.get("val_loss"), (int, float))
            and not math.isnan(r.get("val_loss", float("nan")))
        ]
        return min(valid) if valid else float("inf")

    def _get_viable_variants(
        self,
        results_so_far: list[dict],
        baseline_results: list[dict],
    ) -> list[str]:
        """Get CA variants that showed promise in Q2/Q3.

        A variant is viable if any result has val_loss < 4.0.

        Args:
            results_so_far: Results from the current question (Q3).
            baseline_results: Q1 baseline results (not used for viability).

        Returns:
            List of viable variant names, or top-3 variants as fallback.
        """
        viable: set[str] = set()
        for r in results_so_far:
            vl = r.get("val_loss")
            if (
                vl is not None
                and isinstance(vl, (int, float))
                and not math.isnan(vl)
                and vl < 4.0
            ):
                v = r.get("variant")
                if v:
                    viable.add(v)
        return list(viable) if viable else ALL_CA_VARIANTS[:3]
