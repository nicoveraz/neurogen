"""Q2: CA viability strategy -- broad search, fast abandonment.

Explores CA variants broadly to classify each as viable or abandoned:
- Start with 5 random configs per CA variant.
- If a variant has >3 configs that produce NaN or loss > 8.0, abandon it.
- If a variant has any config with val_loss < 4.0, mark viable and
  confirm with more seeds.
- Stop when all variants are classified as viable or abandoned.
"""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

from research.strategies.base import QuestionStrategy
from research.experiment_generator import (
    ALL_CA_VARIANTS,
    DEFAULT_SEEDS,
    make_experiment,
    sample_random_ca_configs,
)

if TYPE_CHECKING:
    from research.agenda import ResearchQuestion
    from research.results_store import ResultsStore


class ViabilitySearchStrategy(QuestionStrategy):
    """Explore CA variants broadly, abandon failures quickly."""

    ABANDON_THRESHOLD: int = 3      # failures before abandoning a variant
    VIABILITY_LOSS: float = 4.0     # val_loss below this = viable
    INITIAL_CONFIGS: int = 5        # configs to try first per variant
    CONFIRM_SEEDS: int = 3          # seeds needed to confirm viability

    def propose_next(
        self,
        question: ResearchQuestion,
        results_so_far: list[dict],
        baseline_results: list[dict],
        batch_size: int = 5,
    ) -> list[dict]:
        """Propose experiments: explore untested variants, confirm viable ones.

        Args:
            question: The research question being investigated.
            results_so_far: All results for this question from the store.
            baseline_results: Q1 baseline results for comparison.
            batch_size: Maximum number of experiments to propose.

        Returns:
            List of experiment config dicts for the next batch.
        """
        abandoned = self._get_abandoned(results_so_far)
        experiments: list[dict] = []

        for variant in ALL_CA_VARIANTS:
            if variant in abandoned:
                continue

            variant_results = [
                r for r in results_so_far if r.get("variant") == variant
            ]

            if len(variant_results) == 0:
                # Never tested -- start with random configs
                new_exps = sample_random_ca_configs(
                    variant,
                    n=min(self.INITIAL_CONFIGS, batch_size - len(experiments)),
                    question_id=question.id,
                )
                experiments.extend(new_exps)

            elif self._is_viable(variant_results) and self._needs_confirmation(
                variant_results
            ):
                # Viable but need more seeds to confirm
                best_cfg = self._best_config(variant_results)
                used_seeds = {
                    r.get("seed")
                    for r in variant_results
                    if self._configs_match(r, best_cfg)
                }
                for seed in DEFAULT_SEEDS:
                    if seed not in used_seeds and len(experiments) < batch_size:
                        experiments.append(make_experiment(
                            init_method=variant,
                            question_id=question.id,
                            steps=1000,
                            seed=seed,
                            ca_config=best_cfg.get(
                                "ca_config",
                                best_cfg.get("config", {}).get("ca_config", {}),
                            ),
                            reason=f"confirming viability of {variant}",
                        ))

            elif not self._is_viable(variant_results):
                # Not yet viable -- try more configs
                new_exps = sample_random_ca_configs(
                    variant,
                    n=min(3, batch_size - len(experiments)),
                    question_id=question.id,
                )
                experiments.extend(new_exps)

            if len(experiments) >= batch_size:
                break

        return experiments[:batch_size]

    def is_question_answered(
        self,
        question: ResearchQuestion,
        store: ResultsStore,
    ) -> bool:
        """Check if all CA variants have been classified.

        A variant is classified when it is either abandoned (too many
        failures) or confirmed viable (enough seeds below threshold).
        The question is answered when all variants are classified or
        the experiment budget is exhausted.

        Args:
            question: The research question.
            store: The results store for querying.

        Returns:
            True if the question can be considered answered.
        """
        results = store.query(question_id=question.id)
        if not results:
            return False

        abandoned = self._get_abandoned(results)
        for variant in ALL_CA_VARIANTS:
            variant_results = [
                r for r in results if r.get("variant") == variant
            ]
            if len(variant_results) == 0:
                return False  # untested variant remains
            if variant not in abandoned and not self._is_viable(variant_results):
                # Still exploring this variant
                if len(variant_results) < self.INITIAL_CONFIGS + 5:
                    return False

        # Also answered if we hit max experiments
        if store.experiment_count(question.id) >= question.max_experiments:
            return True

        return True

    def _get_abandoned(self, results: list[dict]) -> set[str]:
        """Return set of variant names that should be abandoned.

        Args:
            results: All results for this question.

        Returns:
            Set of variant names with too many failures.
        """
        abandoned: set[str] = set()
        for variant in ALL_CA_VARIANTS:
            vr = [r for r in results if r.get("variant") == variant]
            if self._should_abandon(vr):
                abandoned.add(variant)
        return abandoned

    def _should_abandon(self, variant_results: list[dict]) -> bool:
        """Check if a variant should be abandoned due to repeated failures.

        Args:
            variant_results: Results for a single variant.

        Returns:
            True if the variant has too many failures.
        """
        if len(variant_results) < self.INITIAL_CONFIGS:
            return False
        failures = sum(
            1
            for r in variant_results
            if r.get("val_loss") is None
            or (
                isinstance(r.get("val_loss"), float)
                and (math.isnan(r["val_loss"]) or r["val_loss"] > 8.0)
            )
        )
        return failures > self.ABANDON_THRESHOLD

    def _is_viable(self, variant_results: list[dict]) -> bool:
        """Check if any result shows viability (loss below threshold).

        Args:
            variant_results: Results for a single variant.

        Returns:
            True if at least one result has val_loss below VIABILITY_LOSS.
        """
        return any(
            r.get("val_loss") is not None
            and isinstance(r.get("val_loss"), (int, float))
            and not math.isnan(r["val_loss"])
            and r["val_loss"] < self.VIABILITY_LOSS
            for r in variant_results
        )

    def _needs_confirmation(self, variant_results: list[dict]) -> bool:
        """Check if a viable variant needs more seed confirmation.

        Args:
            variant_results: Results for a single variant.

        Returns:
            True if fewer than CONFIRM_SEEDS viable results exist.
        """
        viable = [
            r
            for r in variant_results
            if r.get("val_loss") is not None
            and isinstance(r.get("val_loss"), (int, float))
            and not math.isnan(r.get("val_loss", float("nan")))
            and r["val_loss"] < self.VIABILITY_LOSS
        ]
        return len(viable) < self.CONFIRM_SEEDS

    def _best_config(self, variant_results: list[dict]) -> dict:
        """Find the config with the lowest val_loss.

        Args:
            variant_results: Results for a single variant.

        Returns:
            The result dict with the best (lowest) val_loss.
        """
        valid = [
            r
            for r in variant_results
            if r.get("val_loss") is not None
            and isinstance(r.get("val_loss"), (int, float))
            and not math.isnan(r.get("val_loss", float("nan")))
        ]
        if not valid:
            return variant_results[0] if variant_results else {}
        return min(valid, key=lambda r: r["val_loss"])

    def _configs_match(self, r: dict, best: dict) -> bool:
        """Check if two results share the same CA config.

        Args:
            r: First result dict.
            best: Second result dict.

        Returns:
            True if the ca_config fields are equal.
        """
        rc = r.get("ca_config", r.get("config", {}).get("ca_config", {}))
        bc = best.get("ca_config", best.get("config", {}).get("ca_config", {}))
        return rc == bc
