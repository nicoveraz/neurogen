"""Q5: Meta-learning strategy -- CMA-ES genome optimization.

Optimizes CA genomes using progressive evaluation: early generations
get short training runs, later generations get longer runs. Detects
plateaus to stop early.
"""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

from research.strategies.base import QuestionStrategy
from research.experiment_generator import ALL_CA_VARIANTS, make_experiment

if TYPE_CHECKING:
    from research.agenda import ResearchQuestion
    from research.results_store import ResultsStore


class MetaLearningStrategy(QuestionStrategy):
    """CMA-ES meta-learning with progressive evaluation."""

    def propose_next(
        self,
        question: ResearchQuestion,
        results_so_far: list[dict],
        baseline_results: list[dict],
        batch_size: int = 5,
    ) -> list[dict]:
        """Propose meta-learning experiments with progressive step counts.

        Early generations use short training runs (500 steps) for fast
        evaluation. As more results accumulate, training runs get longer
        (up to 3000 steps) for more accurate fitness estimates.

        Args:
            question: The research question being investigated.
            results_so_far: All results for this question from the store.
            baseline_results: Q1 baseline results for comparison.
            batch_size: Maximum number of experiments to propose.

        Returns:
            List of experiment config dicts for the next batch.
        """
        # Use best variant from prior questions
        best_variant = self._best_variant(results_so_far, baseline_results)

        experiments: list[dict] = []
        n_done = len(results_so_far)

        # Progressive evaluation: more steps as we go deeper
        if n_done < 20:
            steps = 500
        elif n_done < 50:
            steps = 1000
        elif n_done < 100:
            steps = 2000
        else:
            steps = 3000

        for i in range(batch_size):
            experiments.append(make_experiment(
                init_method=best_variant,
                question_id=question.id,
                steps=steps,
                ca_config={"meta_generation": n_done + i},
                reason=f"meta-learning generation {n_done + i}, {steps} steps",
            ))

        return experiments[:batch_size]

    def is_question_answered(
        self,
        question: ResearchQuestion,
        store: ResultsStore,
    ) -> bool:
        """Check if meta-learning has converged or plateaued.

        Answered positively if the best meta-learned genome is 10% better
        than the early (random) genomes. Also answered if the recent
        improvement has plateaued (< 1% change over last 20 experiments)
        or the budget is exhausted.

        Args:
            question: The research question.
            store: The results store for querying.

        Returns:
            True if the question can be considered answered.
        """
        results = store.query(question_id=question.id)
        if not results:
            return False

        valid = [
            r
            for r in results
            if r.get("val_loss") is not None
            and isinstance(r.get("val_loss"), (int, float))
            and not math.isnan(r["val_loss"])
        ]

        if not valid:
            return False

        # Check if meta-learned beats random genomes by 10%
        best_meta = min(r["val_loss"] for r in valid)
        early = valid[:5]
        if early:
            early_mean = sum(r["val_loss"] for r in early) / len(early)
            if best_meta < early_mean * 0.9:
                return True  # 10% improvement achieved

        # Check for plateau in recent results
        if len(valid) >= 30:
            recent = [r["val_loss"] for r in valid[-10:]]
            older = [r["val_loss"] for r in valid[-20:-10]]
            if recent and older:
                recent_mean = sum(recent) / len(recent)
                older_mean = sum(older) / len(older)
                if abs(recent_mean - older_mean) / max(older_mean, 1e-8) < 0.01:
                    return True  # plateaued

        if store.experiment_count(question.id) >= question.max_experiments:
            return True

        return False

    def _best_variant(
        self,
        results_so_far: list[dict],
        baseline_results: list[dict],
    ) -> str:
        """Find the CA variant with the best val_loss from prior questions.

        Args:
            results_so_far: Results from prior questions.
            baseline_results: Q1 baseline results.

        Returns:
            Name of the best CA variant, defaulting to "grid_ca".
        """
        all_results = results_so_far + baseline_results
        ca_results = [
            r for r in all_results if r.get("variant") in ALL_CA_VARIANTS
        ]
        valid = [
            r
            for r in ca_results
            if r.get("val_loss") is not None
            and isinstance(r.get("val_loss"), (int, float))
            and not math.isnan(r["val_loss"])
        ]
        if valid:
            best = min(valid, key=lambda r: r["val_loss"])
            return best.get("variant", "grid_ca")
        return "grid_ca"
