"""Experiment registry with JSON-based persistence.

Tracks experiment status (pending, running, complete, failed) and
results paths. Enables experiment management and resumption.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class ExperimentRegistry:
    """JSON-based experiment registry for tracking experiment lifecycle.

    Persists experiment records (config, status, results) to a JSON file.
    Supports registering new experiments, updating status, querying by
    status, and retrieving results.

    Args:
        registry_path: Path to the JSON registry file. Created if absent.
    """

    VALID_STATUSES = {"pending", "running", "complete", "failed"}

    def __init__(self, registry_path: str) -> None:
        """Initialize the registry, loading from disk if the file exists.

        Args:
            registry_path: Path to the JSON registry file.
        """
        self._path = Path(registry_path)
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load registry data from the JSON file if it exists."""
        if self._path.exists() and self._path.stat().st_size > 0:
            with open(self._path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                self._data = json.loads(content) if content else {}
        else:
            self._data = {}

    def _save(self) -> None:
        """Persist current registry data to the JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=str)

    def register(
        self,
        experiment_id: str,
        config: dict[str, Any],
        status: str = "pending",
    ) -> None:
        """Register a new experiment in the registry.

        Args:
            experiment_id: Unique identifier for the experiment.
            config: Experiment configuration dictionary.
            status: Initial status (default "pending").

        Raises:
            ValueError: If the experiment_id already exists or status is invalid.
        """
        if experiment_id in self._data:
            raise ValueError(
                f"Experiment '{experiment_id}' already registered. "
                f"Use update_status() to modify it."
            )
        if status not in self.VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. "
                f"Must be one of: {self.VALID_STATUSES}"
            )

        self._data[experiment_id] = {
            "config": config,
            "status": status,
            "results": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._save()

    def update_status(
        self,
        experiment_id: str,
        status: str,
        results: dict[str, Any] | None = None,
    ) -> None:
        """Update the status (and optionally results) of an experiment.

        Args:
            experiment_id: Unique identifier for the experiment.
            status: New status string.
            results: Optional results dictionary to store.

        Raises:
            KeyError: If experiment_id is not registered.
            ValueError: If status is invalid.
        """
        if experiment_id not in self._data:
            raise KeyError(
                f"Experiment '{experiment_id}' not found in registry."
            )
        if status not in self.VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. "
                f"Must be one of: {self.VALID_STATUSES}"
            )

        self._data[experiment_id]["status"] = status
        self._data[experiment_id]["updated_at"] = time.time()

        if results is not None:
            self._data[experiment_id]["results"] = results

        self._save()

    def get_status(self, experiment_id: str) -> str:
        """Get the current status of an experiment.

        Args:
            experiment_id: Unique identifier for the experiment.

        Returns:
            Current status string.

        Raises:
            KeyError: If experiment_id is not registered.
        """
        if experiment_id not in self._data:
            raise KeyError(
                f"Experiment '{experiment_id}' not found in registry."
            )
        return self._data[experiment_id]["status"]

    def list_experiments(
        self, status: str | None = None
    ) -> list[dict[str, Any]]:
        """List experiments, optionally filtered by status.

        Args:
            status: If provided, only return experiments with this status.

        Returns:
            List of dicts with keys: id, status, config, created_at, updated_at.
        """
        results: list[dict[str, Any]] = []
        for exp_id, record in self._data.items():
            if status is not None and record["status"] != status:
                continue
            results.append({
                "id": exp_id,
                "status": record["status"],
                "config": record.get("config", {}),
                "created_at": record.get("created_at"),
                "updated_at": record.get("updated_at"),
            })
        return results

    def get_results(self, experiment_id: str) -> dict[str, Any]:
        """Get stored results for an experiment.

        Args:
            experiment_id: Unique identifier for the experiment.

        Returns:
            Results dictionary, or empty dict if no results stored.

        Raises:
            KeyError: If experiment_id is not registered.
        """
        if experiment_id not in self._data:
            raise KeyError(
                f"Experiment '{experiment_id}' not found in registry."
            )
        results = self._data[experiment_id].get("results")
        return results if results is not None else {}

    def get_config(self, experiment_id: str) -> dict[str, Any]:
        """Get stored config for an experiment.

        Args:
            experiment_id: Unique identifier for the experiment.

        Returns:
            Config dictionary.

        Raises:
            KeyError: If experiment_id is not registered.
        """
        if experiment_id not in self._data:
            raise KeyError(
                f"Experiment '{experiment_id}' not found in registry."
            )
        return self._data[experiment_id].get("config", {})

    def delete(self, experiment_id: str) -> None:
        """Remove an experiment from the registry.

        Args:
            experiment_id: Unique identifier for the experiment.

        Raises:
            KeyError: If experiment_id is not registered.
        """
        if experiment_id not in self._data:
            raise KeyError(
                f"Experiment '{experiment_id}' not found in registry."
            )
        del self._data[experiment_id]
        self._save()

    def reset_failed(self) -> int:
        """Reset all failed experiments back to pending status.

        Returns:
            Number of experiments reset.
        """
        count = 0
        for exp_id, record in self._data.items():
            if record["status"] == "failed":
                record["status"] = "pending"
                record["updated_at"] = time.time()
                count += 1
        if count > 0:
            self._save()
        return count

    def __len__(self) -> int:
        """Return the number of registered experiments."""
        return len(self._data)

    def __contains__(self, experiment_id: str) -> bool:
        """Check if an experiment is registered."""
        return experiment_id in self._data
