"""Experiment registry: tracks status, results paths, and enables resumption."""

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class ExperimentRecord:
    """Record for a single experiment run."""

    experiment_id: str
    name: str
    status: str = ExperimentStatus.PENDING
    config_path: str = ""
    results_dir: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    error_message: str = ""
    metrics_summary: dict = field(default_factory=dict)


class ExperimentRegistry:
    """Tracks experiment status and results.

    Args:
        registry_path: Path to the registry JSON file.
    """

    def __init__(self, registry_path: str | Path = "outputs/registry.json") -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.experiments: dict[str, ExperimentRecord] = {}
        if self.registry_path.exists():
            self._load()

    def _load(self) -> None:
        """Load registry from disk."""
        with open(self.registry_path) as f:
            data = json.load(f)
        for exp_id, exp_data in data.items():
            self.experiments[exp_id] = ExperimentRecord(**exp_data)

    def _save(self) -> None:
        """Save registry to disk."""
        data = {
            exp_id: asdict(record)
            for exp_id, record in self.experiments.items()
        }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def register(
        self,
        experiment_id: str,
        name: str,
        config_path: str = "",
        results_dir: str = "",
    ) -> ExperimentRecord:
        """Register a new experiment."""
        record = ExperimentRecord(
            experiment_id=experiment_id,
            name=name,
            config_path=config_path,
            results_dir=results_dir,
        )
        self.experiments[experiment_id] = record
        self._save()
        return record

    def mark_running(self, experiment_id: str) -> None:
        """Mark an experiment as running."""
        record = self.experiments[experiment_id]
        record.status = ExperimentStatus.RUNNING
        record.started_at = time.time()
        self._save()

    def mark_complete(
        self, experiment_id: str, metrics_summary: dict | None = None
    ) -> None:
        """Mark an experiment as complete."""
        record = self.experiments[experiment_id]
        record.status = ExperimentStatus.COMPLETE
        record.completed_at = time.time()
        if metrics_summary:
            record.metrics_summary = metrics_summary
        self._save()

    def mark_failed(self, experiment_id: str, error: str) -> None:
        """Mark an experiment as failed."""
        record = self.experiments[experiment_id]
        record.status = ExperimentStatus.FAILED
        record.completed_at = time.time()
        record.error_message = error
        self._save()

    def get_status(self, experiment_id: str) -> str:
        """Get the status of an experiment."""
        return self.experiments[experiment_id].status

    def get_record(self, experiment_id: str) -> ExperimentRecord:
        """Get the full record for an experiment."""
        return self.experiments[experiment_id]

    def get_by_status(self, status: str) -> list[ExperimentRecord]:
        """Get all experiments with a given status."""
        return [r for r in self.experiments.values() if r.status == status]

    def list_all(self) -> list[ExperimentRecord]:
        """List all registered experiments."""
        return list(self.experiments.values())
