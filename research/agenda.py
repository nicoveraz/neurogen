"""Research agenda and question dataclasses for auto-research."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ResearchQuestion:
    """A single research question in the agenda.

    Attributes:
        id: Unique identifier (e.g. "Q1_baselines").
        question: Human-readable question text.
        success: Success criteria description.
        metric: Primary metric to evaluate (e.g. "val_loss", "completeness").
        priority: Execution order (lower = higher priority).
        depends_on: List of question IDs that must complete first.
        max_hours: Maximum compute hours for this question.
        max_experiments: Maximum number of experiments to run.
        threshold: Numeric threshold for the metric (optional).
        comparison: What to compare against (e.g. "best_baseline").
        status: Current status: "pending", "active", "completed", "abandoned".
        answer: Summary of the answer once completed.
    """

    id: str = ""
    question: str = ""
    success: str = ""
    metric: str = "val_loss"
    priority: int = 1
    depends_on: list[str] = field(default_factory=list)
    max_hours: float = 2.0
    max_experiments: int = 30
    threshold: float | None = None
    comparison: str | None = None
    status: str = "pending"
    answer: str = ""


@dataclass
class ResearchAgenda:
    """Full research agenda defining all questions to investigate.

    Attributes:
        name: Name of the research program.
        hardware: Hardware profile identifier.
        total_budget_hours: Global compute budget.
        report_every_n_cycles: Generate report every N cycles.
        questions: List of research questions.
    """

    name: str = ""
    hardware: str = "macbook_m1pro_16gb"
    total_budget_hours: float = 80.0
    report_every_n_cycles: int = 5
    questions: list[ResearchQuestion] = field(default_factory=list)


def load_agenda(path: str) -> ResearchAgenda:
    """Load a research agenda from a YAML file.

    Args:
        path: Path to the agenda YAML file.

    Returns:
        Populated ResearchAgenda.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required fields are missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Agenda file not found: {path}")

    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Empty agenda file: {path}")

    questions = []
    for q_raw in raw.get("questions", []):
        questions.append(
            ResearchQuestion(
                id=q_raw["id"],
                question=q_raw.get("question", ""),
                success=q_raw.get("success", ""),
                metric=q_raw.get("metric", "val_loss"),
                priority=q_raw.get("priority", 1),
                depends_on=q_raw.get("depends_on", []),
                max_hours=q_raw.get("max_hours", 2.0),
                max_experiments=q_raw.get("max_experiments", 30),
                threshold=q_raw.get("threshold"),
                comparison=q_raw.get("comparison"),
            )
        )

    return ResearchAgenda(
        name=raw.get("name", "Unnamed"),
        hardware=raw.get("hardware", "macbook_m1pro_16gb"),
        total_budget_hours=raw.get("total_budget_hours", 80.0),
        report_every_n_cycles=raw.get("report_every_n_experiments", 5),
        questions=questions,
    )
