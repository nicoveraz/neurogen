"""Configuration dataclasses and device detection for NeuroGen."""

import torch
from dataclasses import dataclass, field
from typing import Optional, Any


def get_device() -> str:
    """Detect the best available device.

    Returns:
        "cuda" if NVIDIA GPU available, "mps" if Apple Silicon, else "cpu".
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class GPTConfig:
    """Configuration for the MicroGPT model.

    Attributes:
        block_size: Maximum sequence length (context window).
        vocab_size: Number of tokens in the vocabulary (set by dataset).
        n_layer: Number of transformer blocks.
        n_head: Number of attention heads.
        n_embd: Embedding dimension.
        dropout: Dropout probability.
    """
    block_size: int = 256
    vocab_size: int = 0
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2


@dataclass
class TrainConfig:
    """Configuration for the training loop.

    Attributes:
        max_steps: Total number of training steps.
        eval_interval: Steps between evaluations.
        batch_size: Number of sequences per batch.
        lr: Peak learning rate.
        weight_decay: L2 regularization coefficient.
        grad_clip: Maximum gradient norm for clipping.
        warmup_steps: Number of linear warmup steps.
        min_lr: Minimum learning rate for cosine schedule.
        device: Device string or "auto" for auto-detection.
    """
    max_steps: int = 5000
    eval_interval: int = 250
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 100
    min_lr: float = 1e-5
    device: str = "auto"


@dataclass
class CAConfig:
    """Configuration for cellular automata weight generation.

    Attributes:
        variant: CA variant name (e.g. "grid_ca").
        n_steps: Number of CA development steps.
        hidden_dim: Hidden dimension of the CA rule network.
        n_channels: Number of channels per CA cell.
        neighborhood_size: Size of the local neighborhood kernel.
        seed_pattern: How to seed the initial CA grid.
    """
    variant: str = "grid_ca"
    n_steps: int = 64
    hidden_dim: int = 64
    n_channels: int = 16
    neighborhood_size: int = 3
    seed_pattern: str = "center"


@dataclass
class ExperimentConfig:
    """Full experiment specification.

    Attributes:
        name: Human-readable experiment name.
        hypothesis: What this experiment tests.
        model_config: GPT model configuration.
        train_config: Training configuration.
        ca_config: Optional CA configuration (None for baselines).
        init_method: Weight initialization method name.
        dataset: Dataset identifier string.
        metrics: List of metric names to track.
        seeds: List of random seeds for reproducibility.
    """
    name: str = ""
    hypothesis: str = ""
    model_config: GPTConfig = field(default_factory=GPTConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    ca_config: Optional[CAConfig] = None
    init_method: str = "xavier_normal"
    dataset: str = "shakespeare"
    metrics: list[str] = field(default_factory=lambda: ["train_loss", "val_loss"])
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs.

    Attributes:
        name: Benchmark identifier (e.g. "BM1").
        suite: Suite name ("quick" or "standard").
        n_seeds: Number of random seeds to run.
        max_steps: Maximum training steps for the benchmark.
    """
    name: str = ""
    suite: str = "quick"
    n_seeds: int = 3
    max_steps: int = 500


@dataclass
class LiveCAConfig:
    """Configuration for live CA that runs during training.

    Attributes:
        ca_type: Type of live CA rule (e.g. "local_norm").
        ca_interval: Apply CA every N training steps.
        alpha_schedule: How alpha decays ("exponential_decay", "linear", "constant").
        alpha_0: Initial blending factor for CA weight updates.
        decay: Decay rate for alpha schedule.
        total_steps: Total training steps (for schedule computation).
        ca_sees_gradients: Whether the CA rule can access gradient info.
        clamp_weights: Whether to clamp weight magnitudes after CA step.
        max_weight: Maximum absolute weight value if clamping.
        target_params: Which parameters the CA modifies ("all" or a list).
    """
    ca_type: str = "local_norm"
    ca_interval: int = 1
    alpha_schedule: str = "exponential_decay"
    alpha_0: float = 0.01
    decay: float = 0.001
    total_steps: int = 5000
    ca_sees_gradients: bool = True
    clamp_weights: bool = True
    max_weight: float = 3.0
    target_params: str = "all"


HARDWARE_PROFILES: dict[str, dict[str, Any]] = {
    "macbook_m1pro_16gb": {
        "device": "mps",
        "max_batch_size": 32,
        "max_n_embd": 384,
        "max_n_layer": 6,
        "max_block_size": 256,
        "compile": False,
        "dtype": "float32",
    },
    "macbook_m1pro_32gb": {
        "device": "mps",
        "max_batch_size": 64,
        "max_n_embd": 512,
        "max_n_layer": 8,
        "max_block_size": 512,
        "compile": False,
        "dtype": "float32",
    },
    "cpu_only": {
        "device": "cpu",
        "max_batch_size": 16,
        "max_n_embd": 256,
        "max_n_layer": 4,
        "max_block_size": 128,
        "compile": False,
        "dtype": "float32",
    },
}
