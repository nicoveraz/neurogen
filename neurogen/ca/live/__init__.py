"""Live CA rules that operate during training alongside gradient descent."""

from neurogen.ca.live.base import LiveCA
from neurogen.ca.live.local_norm import LocalNormCA
from neurogen.ca.live.modularity import ModularityCA
from neurogen.ca.live.pruning import PruningCA
from neurogen.ca.live.competition import CompetitionCA
from neurogen.ca.live.learned import LearnedCA
from neurogen.ca.live.multi_timescale import MultiTimescaleCA
from neurogen.ca.live.ca_optimizer import CAOptimizer
from neurogen.ca.live.alpha_schedule import AlphaSchedule

LIVE_CA_RULES = {
    "local_norm": LocalNormCA,
    "modularity": ModularityCA,
    "pruning": PruningCA,
    "competition": CompetitionCA,
    "learned": LearnedCA,
}

__all__ = [
    "LiveCA", "LocalNormCA", "ModularityCA", "PruningCA",
    "CompetitionCA", "LearnedCA", "MultiTimescaleCA",
    "CAOptimizer", "AlphaSchedule", "LIVE_CA_RULES",
]
