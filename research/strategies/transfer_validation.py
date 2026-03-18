"""Q6: Transfer validation -- does a learned genome help larger models?

Tests whether genomes optimized on small models (Q5) transfer to
larger models. Runs CA init and xavier baseline at 2x and 4x scale,
comparing val_loss to determine if the benefit scales.
"""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

from neurogen.config import GPTConfig
from research.strategies.base import QuestionStrategy
from research.experiment_generator import DEFAULT_SEEDS, make_experiment

if TYPE_CHECKING:
    from research.agenda import ResearchQuestion
    from research.results_store import ResultsStore


# Scale-up configs: 2x and 4x the default model
SCALE_CONFIGS: dict[str, GPTConfig] = {
    "2x": GPTConfig(
        block_size=256, n_layer=8, n_head=8, n_embd=512, dropout=0.2,
    ),
    "4x": GPTConfig(
        block_size=256, n_layer=12, n_head=12, n_embd=768, dropout=0.2,
    ),
}


class TransferValidationStrategy(QuestionStrategy):
    """Test if genome from Q5 transfers to larger models."""

    def propose_next(
        self,
        question: ResearchQuestion,
        results_so_far: list[dict],
        baseline_results: list[dict],
        batch_size: int = 5,
    ) -> list[dict]:
        """Propose transfer experiments at each scale.

        For each scale (2x, 4x), runs the best CA variant and a xavier
        baseline, each with multiple seeds.

        Args:
            question: The research question being investigated.
            results_so_far: All results for this question from the store.
            baseline_results: Q1 baseline results for comparison.
            batch_size: Maximum number of experiments to propose.

        Returns:
            List of experiment config dicts for the next batch.
        """
        best_variant = self._best_variant(baseline_results)

        experiments: list[dict] = []
        tested: set[tuple[Any, str | None, int | None]] = {
            (
                r.get("config", {}).get("ca_config", {}).get("scale"),
                r.get("init_method"),
                r.get("seed"),
            )
            for r in results_so_far
        }

        for scale_name, scale_config in SCALE_CONFIGS.items():
            # Test CA variant at this scale
            for seed in DEFAULT_SEEDS:
                if (
                    (scale_name, best_variant, seed) not in tested
                    and len(experiments) < batch_size
                ):
                    experiments.append(make_experiment(
                        init_method=best_variant,
                        question_id=question.id,
                        steps=3000,
                        seed=seed,
                        model_config=scale_config,
                        ca_config={"scale": scale_name},
                        reason=(
                            f"transfer test: {best_variant} "
                            f"at {scale_name} scale"
                        ),
                    ))

            # Also test xavier baseline at this scale for comparison
            for seed in DEFAULT_SEEDS:
                if (
                    (scale_name, "xavier_normal", seed) not in tested
                    and len(experiments) < batch_size
                ):
                    experiments.append(make_experiment(
                        init_method="xavier_normal",
                        question_id=question.id,
                        steps=3000,
                        seed=seed,
                        model_config=scale_config,
                        ca_config={"scale": scale_name},
                        reason=(
                            f"transfer baseline: xavier "
                            f"at {scale_name} scale"
                        ),
                    ))

        return experiments[:batch_size]

    def is_question_answered(
        self,
        question: ResearchQuestion,
        store: ResultsStore,
    ) -> bool:
        """Check if transfer has been tested at all scales.

        Requires at least 3 results at each scale. Also answered if
        total experiment count meets the expected full set (2 scales
        x 3 seeds x 2 methods = 12 experiments).

        Args:
            question: The research question.
            store: The results store for querying.

        Returns:
            True if the question can be considered answered.
        """
        results = store.query(question_id=question.id)
        if not results:
            return False

        # Need at least some results at each scale
        for scale_name in SCALE_CONFIGS:
            scale_results = [
                r
                for r in results
                if r.get("config", {}).get("ca_config", {}).get("scale")
                == scale_name
            ]
            if len(scale_results) < 3:
                return False

        if store.experiment_count(question.id) >= question.max_experiments:
            return True

        # Check if we have enough data for both scales
        return len(results) >= len(SCALE_CONFIGS) * len(DEFAULT_SEEDS) * 2

    def _best_variant(self, baseline_results: list[dict]) -> str:
        """Find the best CA variant from prior results.

        Args:
            baseline_results: Results from prior questions containing
                CA variant results.

        Returns:
            Name of the best CA variant, defaulting to "grid_ca".
        """
        from research.experiment_generator import ALL_CA_VARIANTS

        ca = [
            r
            for r in baseline_results
            if r.get("variant") in ALL_CA_VARIANTS
        ]
        valid = [
            r
            for r in ca
            if r.get("val_loss") is not None
            and isinstance(r.get("val_loss"), (int, float))
            and not math.isnan(r["val_loss"])
        ]
        if valid:
            return min(valid, key=lambda r: r["val_loss"]).get(
                "variant", "grid_ca"
            )
        return "grid_ca"
