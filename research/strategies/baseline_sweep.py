"""Q1: Baseline sweep strategy -- exhaustive enumeration.

Runs all 9 baselines x 3 seeds. No adaptive decisions needed;
this is a simple exhaustive sweep that produces the reference
measurements for all subsequent questions.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from research.strategies.base import QuestionStrategy
from research.experiment_generator import ALL_BASELINES, DEFAULT_SEEDS, make_experiment

if TYPE_CHECKING:
    from research.agenda import ResearchQuestion
    from research.results_store import ResultsStore


class BaselineSweepStrategy(QuestionStrategy):
    """Run all 9 baselines x 3 seeds. No decisions needed."""

    def propose_next(
        self,
        question: ResearchQuestion,
        results_so_far: list[dict],
        baseline_results: list[dict],
        batch_size: int = 5,
    ) -> list[dict]:
        """Propose the next batch of baseline experiments.

        Enumerates every (init_method, seed) pair that has not yet been
        run, returning up to ``batch_size`` experiments.

        Args:
            question: The research question being investigated.
            results_so_far: All results for this question from the store.
            baseline_results: Q1 baseline results (same as results_so_far here).
            batch_size: Maximum number of experiments to propose.

        Returns:
            List of experiment config dicts for unfinished baseline runs.
        """
        done: set[tuple[str | None, int | None]] = {
            (r.get("init_method"), r.get("seed")) for r in results_so_far
        }

        experiments: list[dict] = []
        for init_name in ALL_BASELINES:
            for seed in DEFAULT_SEEDS:
                if (init_name, seed) not in done:
                    experiments.append(make_experiment(
                        init_method=init_name,
                        question_id=question.id,
                        steps=5000,
                        seed=seed,
                        reason=f"baseline sweep: {init_name} seed={seed}",
                    ))
                if len(experiments) >= batch_size:
                    return experiments

        return experiments[:batch_size]

    def is_question_answered(
        self,
        question: ResearchQuestion,
        store: ResultsStore,
    ) -> bool:
        """Check if all baseline x seed combinations have been run.

        Args:
            question: The research question.
            store: The results store for querying.

        Returns:
            True if every (baseline, seed) pair has a recorded result.
        """
        results = store.query(question_id=question.id)
        done: set[tuple[str | None, int | None]] = {
            (r.get("init_method"), r.get("seed")) for r in results
        }
        total_needed = len(ALL_BASELINES) * len(DEFAULT_SEEDS)
        return len(done) >= total_needed
