"""Main auto-research loop: closed-loop experiment execution.

Ties together the decision engine, experiment runner, results store,
and report generator into an autonomous research loop.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from research.agenda import load_agenda, ResearchAgenda
from research.decision_engine import DecisionEngine
from research.results_store import ResultsStore

logger = logging.getLogger(__name__)


class AutoResearch:
    """Autonomous research loop.

    Repeatedly asks the decision engine what to run, runs it,
    records results, and generates periodic reports.

    Args:
        agenda_path: Path to the research agenda YAML.
        output_dir: Base directory for outputs.
        device: Device override (None for auto-detect).
    """

    def __init__(
        self,
        agenda_path: str = "research/agenda.yaml",
        output_dir: str = "outputs/auto_research",
        device: str | None = None,
    ) -> None:
        self.agenda = load_agenda(agenda_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.store = ResultsStore(
            path=str(self.output_dir / "results.db")
        )
        self.decision = DecisionEngine(self.agenda, self.store)
        self.device = device

    def run(self, max_cycles: int | None = None) -> dict[str, Any]:
        """Main auto-research loop.

        Args:
            max_cycles: Optional limit on number of cycles (for testing).

        Returns:
            Final status dict.
        """
        from research.engine import run_experiment

        print(f"=== Starting Auto-Research: {self.agenda.name} ===")
        print(f"Budget: {self.agenda.total_budget_hours}h")
        print(f"Questions: {len(self.agenda.questions)}")
        print()

        cycle = 0
        while True:
            cycle += 1
            if max_cycles is not None and cycle > max_cycles:
                print(f"Max cycles ({max_cycles}) reached.")
                break

            # 1. Ask decision engine what to run
            experiments = self.decision.next_experiments(batch_size=5)

            if not experiments:
                print("All questions answered or budget exhausted.")
                break

            # 2. Run the batch
            active_q = self._active_question_id(experiments)
            print(
                f"\n--- Cycle {cycle}: running {len(experiments)} "
                f"experiments for {active_q} ---"
            )

            for exp in experiments:
                name = exp.get("init_method", "unknown")
                seed = exp.get("seed", 0)
                steps = exp.get("steps", 0)
                reason = exp.get("reason", "")
                print(
                    f"  Running: {name} (seed={seed}, "
                    f"{steps} steps) [{reason}]"
                )

                start = time.time()
                try:
                    result = run_experiment(exp, device=self.device)
                    duration = time.time() - start

                    self.store.record(
                        experiment_id=exp.get(
                            "experiment_id", f"exp_{cycle}"
                        ),
                        config=_config_to_dict(exp),
                        metrics=result,
                        question_id=exp.get("question_id", ""),
                        duration_seconds=duration,
                        decision_reason=reason,
                    )

                    vl = result.get(
                        "val_loss",
                        result.get("final_val_loss", "?"),
                    )
                    print(f"    -> val_loss={vl}, took {duration:.1f}s")

                except Exception as e:
                    duration = time.time() - start
                    logger.error(f"Experiment failed: {e}")
                    self.store.record(
                        experiment_id=exp.get(
                            "experiment_id", f"exp_{cycle}"
                        ),
                        config=_config_to_dict(exp),
                        metrics={
                            "error": str(e),
                            "val_loss": float("nan"),
                        },
                        question_id=exp.get("question_id", ""),
                        duration_seconds=duration,
                        decision_reason=reason,
                    )
                    print(f"    -> FAILED: {e}")

            # 3. Check global budget
            total_hours = self.store.budget_used()
            if total_hours >= self.agenda.total_budget_hours:
                print(
                    f"\nGlobal budget exhausted ({total_hours:.1f}h used)"
                )
                break

            # 4. Periodic report
            if cycle % self.agenda.report_every_n_cycles == 0:
                self._generate_report(cycle)

            # 5. Print status
            self._print_status()

        # Final report
        self._generate_report(cycle, final=True)

        return self.decision.get_status()

    def status(self) -> dict[str, Any]:
        """Get current research status without running anything.

        Returns:
            Dict mapping question ID to status info.
        """
        return self.decision.get_status()

    def _active_question_id(self, experiments: list[dict]) -> str:
        """Extract question ID from a batch of experiments.

        Args:
            experiments: List of experiment config dicts.

        Returns:
            The question_id from the first experiment, or "unknown".
        """
        if experiments:
            return experiments[0].get("question_id", "unknown")
        return "unknown"

    def _print_status(self) -> None:
        """Print current status of all questions."""
        hours = self.store.budget_used()
        n_exp = self.store.experiment_count()
        print(
            f"\n  Status ({n_exp} experiments, "
            f"{hours:.1f}h / {self.agenda.total_budget_hours}h):"
        )
        for q in self.agenda.questions:
            qcount = self.store.experiment_count(q.id)
            qhours = self.store.budget_used(q.id)
            best = self.store.best_result(q.id)
            best_loss = best.get("val_loss", "N/A") if best else "N/A"
            print(
                f"    {q.id}: {q.status} "
                f"({qcount}/{q.max_experiments} exps, "
                f"{qhours:.1f}/{q.max_hours}h, best={best_loss})"
            )

    def _generate_report(
        self, cycle: int, final: bool = False
    ) -> None:
        """Generate a progress or final report.

        Args:
            cycle: Current cycle number.
            final: Whether this is the final report.
        """
        report_name = (
            "final_report.md" if final else f"report_cycle_{cycle}.md"
        )
        report_path = self.output_dir / report_name

        status = self.decision.get_status()
        lines = [
            f"# Auto-Research {'Final ' if final else ''}"
            f"Report (Cycle {cycle})",
            "",
            f"**Program:** {self.agenda.name}",
            f"**Total experiments:** {self.store.experiment_count()}",
            f"**Total compute:** {self.store.budget_used():.1f}h "
            f"/ {self.agenda.total_budget_hours}h",
            "",
            "## Question Status",
            "",
            "| Question | Status | Experiments | Hours "
            "| Best Val Loss |",
            "|----------|--------|-------------|-------"
            "|---------------|",
        ]

        for qid, info in status.items():
            best = self.store.best_result(qid)
            if best and best.get("val_loss"):
                best_loss = f"{best['val_loss']:.4f}"
            else:
                best_loss = "N/A"
            lines.append(
                f"| {qid} | {info['status']} "
                f"| {info['experiments']}/{info['max_experiments']} "
                f"| {info['hours_used']:.1f}/{info['max_hours']} "
                f"| {best_loss} |"
            )

        lines.extend(["", "## Answers", ""])
        for qid, info in status.items():
            if info["answer"]:
                lines.append(f"**{qid}:** {info['answer']}")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nReport written to {report_path}")


def _config_to_dict(exp: dict) -> dict:
    """Convert experiment config to a serializable dict.

    Handles dataclass fields by extracting their attributes into
    plain dicts.

    Args:
        exp: Experiment config dict, possibly containing dataclasses.

    Returns:
        A fully serializable dict.
    """
    result: dict[str, Any] = {}
    for k, v in exp.items():
        if hasattr(v, "__dataclass_fields__"):
            result[k] = {
                f: getattr(v, f) for f in v.__dataclass_fields__
            }
        else:
            result[k] = v
    return result
