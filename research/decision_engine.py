"""Decision engine for auto-research: decides what experiments to run next.

Takes the research agenda and results store, finds the highest-priority
active question, and delegates to its strategy to propose experiments.
"""

from __future__ import annotations

import logging
from typing import Any

from research.agenda import ResearchAgenda, ResearchQuestion
from research.results_store import ResultsStore
from research.strategies import STRATEGIES

logger = logging.getLogger(__name__)


class DecisionEngine:
    """Decides what experiments to run next based on agenda and results.

    The engine respects question dependencies, budget constraints,
    and delegates to question-specific strategies for decision logic.

    Args:
        agenda: The research agenda with all questions.
        store: Persistent results store.
    """

    def __init__(self, agenda: ResearchAgenda, store: ResultsStore) -> None:
        self.agenda = agenda
        self.store = store
        self._strategy_instances: dict[str, Any] = {}

    def next_experiments(self, batch_size: int = 5) -> list[dict]:
        """Decide what to run next. Returns a batch of experiment configs.

        1. Find the highest-priority active question
        2. Check budget constraints
        3. Dispatch to question-specific strategy
        4. Check for completion

        Args:
            batch_size: Max experiments to propose.

        Returns:
            List of experiment config dicts, empty if all done.
        """
        question = self._get_active_question()
        if question is None:
            return []

        # Check question budget
        if self._is_over_budget(question):
            logger.info(f"Question {question.id} over budget, marking abandoned")
            question.status = "abandoned"
            question.answer = "Budget exhausted before answer found"
            return self.next_experiments(batch_size)

        # Check global budget
        if self.store.budget_used() >= self.agenda.total_budget_hours:
            logger.info("Global budget exhausted")
            return []

        # Get strategy for this question
        strategy = self._get_strategy(question)

        # Check if already answered
        if strategy.is_question_answered(question, self.store):
            logger.info(f"Question {question.id} answered")
            question.status = "completed"
            # Try next question
            return self.next_experiments(batch_size)

        # Get results for context
        results_so_far = self.store.query(question_id=question.id)
        baseline_results = self.store.baseline_results()

        # Propose experiments
        experiments = strategy.propose_next(
            question=question,
            results_so_far=results_so_far,
            baseline_results=baseline_results,
            batch_size=batch_size,
        )

        if not experiments:
            # Strategy has nothing to propose -- mark answered
            question.status = "completed"
            question.answer = "No more experiments to propose"
            return self.next_experiments(batch_size)

        return experiments

    def _get_active_question(self) -> ResearchQuestion | None:
        """Find highest priority question whose dependencies are met."""
        for q in sorted(self.agenda.questions, key=lambda q: q.priority):
            if q.status in ("completed", "abandoned"):
                continue

            # Check dependencies
            deps_met = all(
                self._is_completed(dep) for dep in q.depends_on
            )
            if deps_met:
                q.status = "active"
                return q

        return None

    def _is_completed(self, question_id: str) -> bool:
        """Check if a question is completed.

        Args:
            question_id: ID of the question to check.

        Returns:
            True if the question has status "completed".
        """
        for q in self.agenda.questions:
            if q.id == question_id:
                return q.status == "completed"
        return False

    def _is_over_budget(self, question: ResearchQuestion) -> bool:
        """Check if a question has exceeded its budget.

        Args:
            question: The research question to check.

        Returns:
            True if the question is over its hour or experiment budget.
        """
        hours = self.store.budget_used(question.id)
        count = self.store.experiment_count(question.id)
        return hours >= question.max_hours or count >= question.max_experiments

    def _get_strategy(self, question: ResearchQuestion) -> Any:
        """Get or create the strategy instance for a question.

        Args:
            question: The research question needing a strategy.

        Returns:
            An instantiated QuestionStrategy for this question.

        Raises:
            ValueError: If no strategy is registered for the question ID.
        """
        if question.id not in self._strategy_instances:
            strategy_cls = STRATEGIES.get(question.id)
            if strategy_cls is None:
                raise ValueError(f"No strategy registered for {question.id}")
            self._strategy_instances[question.id] = strategy_cls()
        return self._strategy_instances[question.id]

    def get_status(self) -> dict[str, Any]:
        """Get current status of all questions.

        Returns:
            Dict mapping question ID to status info including experiment
            counts, hours used, and any answer found.
        """
        status: dict[str, Any] = {}
        for q in self.agenda.questions:
            status[q.id] = {
                "status": q.status,
                "question": q.question,
                "experiments": self.store.experiment_count(q.id),
                "max_experiments": q.max_experiments,
                "hours_used": round(self.store.budget_used(q.id), 2),
                "max_hours": q.max_hours,
                "answer": q.answer,
            }
        return status
