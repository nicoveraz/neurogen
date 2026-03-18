"""Configuration dataclasses and device utilities for NeuroGen."""

from dataclasses import dataclass, field

import torch


def get_device() -> str:
    """Auto-detect best available device.

    Returns:
        "cuda", "mps", or "cpu" depending on hardware availability.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@dataclass
class GPTConfig:
    """Configuration for the MicroGPT model."""

    block_size: int = 256
    vocab_size: int = 0  # set by dataset
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )


@dataclass
class TrainConfig:
    """Configuration for the training loop."""

    max_steps: int = 5000
    eval_interval: int = 250
    eval_steps: int = 20
    batch_size: int = 64
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    device: str = ""
    checkpoint_dir: str = "outputs/checkpoints"
    log_interval: int = 10

    def __post_init__(self) -> None:
        if not self.device:
            self.device = get_device()


@dataclass
class CAConfig:
    """Configuration for cellular automata weight development."""

    variant: str = "grid_ca"
    n_steps: int = 64
    hidden_dim: int = 64
    n_channels: int = 16
    seed_pattern: str = "center_block"
    neighborhood: str = "moore_3x3"
    boundary: str = "periodic"
    stochastic_rate: float = 0.5


@dataclass
class ExperimentConfig:
    """Configuration for a full experiment run."""

    name: str = ""
    hypothesis: str = ""
    model: GPTConfig = field(default_factory=GPTConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ca: CAConfig = field(default_factory=CAConfig)
    init_method: str = "xavier_normal"
    dataset: str = "shakespeare_char"
    seeds: list[int] = field(default_factory=lambda: [42, 137, 256])


@dataclass
class BenchmarkConfig:
    """Configuration for running benchmarks."""

    quick_steps: int = 500
    standard_steps: int = 5000
    extended_steps: int = 20000
    n_seeds: int = 3
    n_seeds_full: int = 5
    target_losses: list[float] = field(
        default_factory=lambda: [4.0, 3.5, 3.0, 2.5, 2.0]
    )
    eval_interval_quick: int = 50
    eval_interval_standard: int = 250


HARDWARE_PROFILES: dict[str, dict] = {
    "macbook_m1pro_16gb": {
        "description": "MacBook Pro M1 Pro, 16GB unified memory",
        "max_config": {
            "n_layer": 6,
            "n_head": 6,
            "n_embd": 384,
            "block_size": 256,
        },
        "safe_batch_size": 32,
        "meta_learning_population": 10,
        "meta_learning_inner_steps": 300,
    },
    "macbook_m1pro_32gb": {
        "description": "MacBook Pro M1 Pro, 32GB unified memory",
        "max_config": {
            "n_layer": 8,
            "n_head": 8,
            "n_embd": 512,
            "block_size": 512,
        },
        "safe_batch_size": 64,
        "meta_learning_population": 20,
        "meta_learning_inner_steps": 500,
    },
    "cpu_only": {
        "description": "Any machine, CPU-only fallback",
        "max_config": {
            "n_layer": 4,
            "n_head": 4,
            "n_embd": 128,
            "block_size": 128,
        },
        "safe_batch_size": 16,
        "meta_learning_population": 5,
        "meta_learning_inner_steps": 100,
    },
}
