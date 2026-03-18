"""SQLite-backed persistent store for experiment results."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any


class ResultsStore:
    """Persistent store for experiment results using SQLite.

    Records experiment configs, metrics, and timing. Supports
    querying by question, variant, metric, and budget tracking.
    """

    def __init__(self, path: str = "outputs/results.db") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create the experiments table and indexes if they don't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                question_id TEXT NOT NULL,
                config_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                variant TEXT,
                init_method TEXT,
                seed INTEGER,
                val_loss REAL,
                train_loss REAL,
                steps_to_target REAL,
                duration_seconds REAL NOT NULL,
                created_at REAL NOT NULL,
                decision_reason TEXT
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_question ON experiments(question_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_variant ON experiments(variant)
        """)
        self._conn.commit()

    def record(
        self,
        experiment_id: str,
        config: dict,
        metrics: dict,
        question_id: str,
        duration_seconds: float,
        decision_reason: str = "",
    ) -> None:
        """Store one experiment's results.

        Args:
            experiment_id: Unique ID for this experiment run.
            config: Full experiment configuration dict.
            metrics: Dict of all metrics collected during the run.
            question_id: Which research question this addresses.
            duration_seconds: Wall-clock time for the experiment.
            decision_reason: Why the decision engine chose this experiment.
        """
        # Extract key metrics for indexed columns
        val_loss = metrics.get(
            "final_val_loss", metrics.get("val_loss", metrics.get("best_val_loss"))
        )
        if val_loss is not None and isinstance(val_loss, float) and math.isnan(val_loss):
            val_loss = None
        train_loss = metrics.get("final_train_loss", metrics.get("train_loss"))
        if (
            train_loss is not None
            and isinstance(train_loss, float)
            and math.isnan(train_loss)
        ):
            train_loss = None

        variant = config.get("variant") or config.get("init_method", "")
        init_method = config.get("init_method", variant)
        seed = config.get("seed")
        steps_to_target = metrics.get("steps_to_target")

        self._conn.execute(
            """INSERT OR REPLACE INTO experiments
               (experiment_id, question_id, config_json, metrics_json,
                variant, init_method, seed, val_loss, train_loss,
                steps_to_target, duration_seconds, created_at, decision_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                experiment_id,
                question_id,
                json.dumps(config),
                json.dumps(_make_serializable(metrics)),
                variant,
                init_method,
                seed,
                val_loss,
                train_loss,
                steps_to_target,
                duration_seconds,
                time.time(),
                decision_reason,
            ),
        )
        self._conn.commit()

    def query(
        self,
        question_id: str | None = None,
        variant: str | None = None,
        init_method: str | None = None,
        metric: str | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """Retrieve results filtered and sorted.

        Args:
            question_id: Filter by research question ID.
            variant: Filter by CA variant name.
            init_method: Filter by initialization method.
            metric: Sort by this metric (currently sorts by val_loss ASC).
            top_k: Return only the top K results.

        Returns:
            List of experiment result dicts with parsed config/metrics.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if question_id:
            conditions.append("question_id = ?")
            params.append(question_id)
        if variant:
            conditions.append("variant = ?")
            params.append(variant)
        if init_method:
            conditions.append("init_method = ?")
            params.append(init_method)

        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        order = ""
        if metric == "val_loss":
            order = " ORDER BY val_loss ASC"
        elif metric:
            order = " ORDER BY val_loss ASC"

        limit = f" LIMIT {top_k}" if top_k else ""

        rows = self._conn.execute(
            f"SELECT * FROM experiments{where}{order}{limit}", params
        ).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def best_result(self, question_id: str, metric: str = "val_loss") -> dict | None:
        """Best result for a question by a given metric.

        Args:
            question_id: Research question to look up.
            metric: Metric to rank by (default: val_loss).

        Returns:
            Dict of the best experiment, or None if no results exist.
        """
        results = self.query(question_id=question_id, metric=metric, top_k=1)
        return results[0] if results else None

    def baseline_results(self) -> list[dict]:
        """All Q1 baseline results for comparison.

        Returns:
            List of all experiment dicts for question Q1_baselines.
        """
        return self.query(question_id="Q1_baselines")

    def best_baseline_loss(self) -> float:
        """Best val_loss achieved by any baseline.

        Returns:
            Lowest val_loss from Q1_baselines, or inf if none recorded.
        """
        row = self._conn.execute(
            "SELECT MIN(val_loss) FROM experiments "
            "WHERE question_id = 'Q1_baselines' AND val_loss IS NOT NULL"
        ).fetchone()
        return row[0] if row and row[0] is not None else float("inf")

    def budget_used(self, question_id: str | None = None) -> float:
        """Hours consumed, optionally filtered by question.

        Args:
            question_id: If provided, only count hours for this question.

        Returns:
            Total compute hours used.
        """
        if question_id:
            row = self._conn.execute(
                "SELECT SUM(duration_seconds) FROM experiments WHERE question_id = ?",
                (question_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT SUM(duration_seconds) FROM experiments"
            ).fetchone()
        total_sec = row[0] if row and row[0] else 0.0
        return total_sec / 3600.0

    def experiment_count(self, question_id: str | None = None) -> int:
        """Number of experiments run.

        Args:
            question_id: If provided, only count for this question.

        Returns:
            Total experiment count.
        """
        if question_id:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE question_id = ?",
                (question_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM experiments"
            ).fetchone()
        return row[0] if row else 0

    def variants_tested(self, question_id: str) -> set[str]:
        """Set of CA variants that have been tested for a question.

        Args:
            question_id: Research question to look up.

        Returns:
            Set of variant name strings.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT variant FROM experiments WHERE question_id = ?",
            (question_id,),
        ).fetchall()
        return {r[0] for r in rows if r[0]}

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a database row to a dict with parsed JSON fields.

        Args:
            row: A sqlite3.Row from a query.

        Returns:
            Dict with config_json/metrics_json parsed into config/metrics.
        """
        d = dict(row)
        d["config"] = json.loads(d.pop("config_json", "{}"))
        d["metrics"] = json.loads(d.pop("metrics_json", "{}"))
        return d

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


def _make_serializable(obj: Any) -> Any:
    """Make an object JSON-serializable.

    Converts NaN to None and infinity to string representations.

    Args:
        obj: Any Python object to make serializable.

    Returns:
        A JSON-safe version of the object.
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return str(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj
