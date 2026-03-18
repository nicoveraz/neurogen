"""Creates ExperimentConfig from strategy decisions.

Bridges the decision engine (which thinks in terms of questions, variants,
and hyperparameter ranges) with the experiment runner (which needs fully
specified ExperimentConfig objects).
"""

from __future__ import annotations

import random
import uuid
from typing import Any

from neurogen.config import ExperimentConfig, GPTConfig, TrainConfig, CAConfig


def make_experiment_id() -> str:
    """Generate a unique experiment ID."""
    return uuid.uuid4().hex[:12]


def make_experiment(
    init_method: str,
    question_id: str,
    steps: int = 1000,
    seed: int | None = None,
    model_config: GPTConfig | None = None,
    train_config: TrainConfig | None = None,
    ca_config: dict | None = None,
    reason: str = "",
) -> dict:
    """Create a fully specified experiment dict.

    Args:
        init_method: Initialization method name (baseline or CA variant).
        question_id: Which research question this serves.
        steps: Training steps.
        seed: Random seed (random if None).
        model_config: Model config (default tiny if None).
        train_config: Training config (default if None).
        ca_config: CA-specific config overrides.
        reason: Why this experiment was chosen (for logging).

    Returns:
        Dict with all fields needed to run the experiment.
    """
    if seed is None:
        seed = random.randint(0, 2**31)

    if model_config is None:
        model_config = GPTConfig(
            block_size=256, vocab_size=0,
            n_layer=6, n_head=6, n_embd=384, dropout=0.2,
        )

    if train_config is None:
        train_config = TrainConfig(max_steps=steps, eval_interval=max(steps // 10, 1))
    else:
        train_config.max_steps = steps

    return {
        "experiment_id": make_experiment_id(),
        "question_id": question_id,
        "init_method": init_method,
        "variant": init_method,
        "seed": seed,
        "steps": steps,
        "model_config": model_config,
        "train_config": train_config,
        "ca_config": ca_config or {},
        "reason": reason,
    }


# All 9 baselines from the spec
ALL_BASELINES = [
    "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal",
    "orthogonal", "sparse", "fixup", "mimetic", "spectral_delta",
]

# All 5 CA variants
ALL_CA_VARIANTS = [
    "grid_ca", "neural_ca", "spectral_ca", "topo_ca", "reaction_diffusion",
]

# Default seeds for multi-seed runs
DEFAULT_SEEDS = [42, 137, 256]


def sample_random_ca_configs(
    variant: str,
    n: int = 5,
    question_id: str = "",
) -> list[dict]:
    """Sample random CA configurations for exploration.

    Varies hidden_dim, n_steps, neighborhood_size, seed_pattern etc.
    Different hyperparameter ranges per variant.

    Args:
        variant: CA variant name (e.g. "neural_ca").
        n: Number of random configs to sample.
        question_id: Research question ID for tagging.

    Returns:
        List of experiment dicts with varied CA hyperparameters.
    """
    configs = []
    for _ in range(n):
        # Common hyperparameters
        hidden_dim = random.choice([32, 64, 128])
        n_steps = random.choice([32, 64, 128])
        seed_pattern = random.choice(["center", "random"])
        seed = random.randint(0, 2**31)

        ca_cfg: dict[str, Any] = {
            "hidden_dim": hidden_dim,
            "n_steps": n_steps,
            "seed_pattern": seed_pattern,
        }

        # Variant-specific params
        if variant == "neural_ca":
            ca_cfg["n_channels"] = random.choice([8, 16, 32])
            ca_cfg["p_update"] = random.choice([0.3, 0.5, 0.7])
        elif variant == "spectral_ca":
            ca_cfg["n_freq"] = random.choice([8, 16, 32])
        elif variant == "topo_ca":
            ca_cfg["topology"] = random.choice(["small_world", "block", "random"])
        elif variant == "reaction_diffusion":
            ca_cfg["model_type"] = random.choice([
                "gray_scott", "fitzhugh_nagumo", "brusselator",
            ])
            ca_cfg["dt"] = random.choice([0.1, 0.5, 1.0])

        configs.append(make_experiment(
            init_method=variant,
            question_id=question_id,
            steps=1000,
            seed=seed,
            ca_config=ca_cfg,
            reason=f"random exploration of {variant}",
        ))

    return configs


def make_live_ca_experiment(
    ca_rule: str,
    init_method: str,
    question_id: str,
    alpha_schedule: str = "exponential_decay",
    alpha_0: float = 0.01,
    steps: int = 3000,
    seed: int | None = None,
    reason: str = "",
) -> dict:
    """Create a live CA experiment config.

    Args:
        ca_rule: Name of the live CA rule (e.g. "local_norm").
        init_method: Base initialization method before live CA.
        question_id: Research question ID.
        alpha_schedule: Alpha decay schedule type.
        alpha_0: Initial alpha blending factor.
        steps: Training steps.
        seed: Random seed (random if None).
        reason: Why this experiment was chosen.

    Returns:
        Experiment dict configured for live CA training.
    """
    return make_experiment(
        init_method=init_method,
        question_id=question_id,
        steps=steps,
        seed=seed,
        ca_config={
            "live_ca_rule": ca_rule,
            "alpha_schedule": alpha_schedule,
            "alpha_0": alpha_0,
            "live": True,
        },
        reason=reason,
    )
