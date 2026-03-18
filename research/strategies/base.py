"""Abstract base class for question strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from research.agenda import ResearchQuestion
    from research.results_store import ResultsStore


class QuestionStrategy(ABC):
    """Base class for research question strategies.

    Each question type gets a strategy that decides what experiments
    to run next based on results so far.
    """

    @abstractmethod
    def propose_next(
        self,
        question: ResearchQuestion,
        results_so_far: list[dict],
        baseline_results: list[dict],
        batch_size: int = 5,
    ) -> list[dict]:
        """Generate the next batch of experiment configs.

        Args:
            question: The research question being investigated.
            results_so_far: All results for this question from the store.
            baseline_results: Q1 baseline results for comparison.
            batch_size: Maximum number of experiments to propose.

        Returns:
            List of experiment config dicts ready for the runner.
        """

    @abstractmethod
    def is_question_answered(
        self,
        question: ResearchQuestion,
        store: ResultsStore,
    ) -> bool:
        """Check if we have enough evidence to answer the question.

        Args:
            question: The research question.
            store: The results store for querying.

        Returns:
            True if the question can be considered answered.
        """
