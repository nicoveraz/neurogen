"""Auto-research strategy system for NeuroGen.

Each research question gets a strategy that decides what experiments
to run next based on results so far. Strategies implement adaptive
experiment selection: broad exploration, fast abandonment, focused
optimization, and statistical confirmation.
"""

from research.strategies.base import QuestionStrategy
from research.strategies.baseline_sweep import BaselineSweepStrategy
from research.strategies.viability import ViabilitySearchStrategy
from research.strategies.focused_optimization import FocusedOptimizationStrategy
from research.strategies.live_ca_comparison import LiveCAComparisonStrategy
from research.strategies.meta_learning import MetaLearningStrategy
from research.strategies.transfer_validation import TransferValidationStrategy

STRATEGIES: dict[str, type[QuestionStrategy]] = {
    "Q1_baselines": BaselineSweepStrategy,
    "Q2_ca_viability": ViabilitySearchStrategy,
    "Q3_ca_beats_baseline": FocusedOptimizationStrategy,
    "Q4_live_ca": LiveCAComparisonStrategy,
    "Q5_meta_learned": MetaLearningStrategy,
    "Q6_transfer": TransferValidationStrategy,
}

__all__ = [
    "QuestionStrategy",
    "BaselineSweepStrategy",
    "ViabilitySearchStrategy",
    "FocusedOptimizationStrategy",
    "LiveCAComparisonStrategy",
    "MetaLearningStrategy",
    "TransferValidationStrategy",
    "STRATEGIES",
]
