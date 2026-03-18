"""CA Weight Engine: dispatches to CA variants and handles full model initialization.

Provides the CAWeightEngine class that creates CA-developed weight tensors
for all weight matrices in a GPT model, using any registered CA variant.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from neurogen.ca.genome import CAGenome
from neurogen.ca.grid_ca import GridCAGenome
from neurogen.ca.neural_ca import NeuralCAGenome
from neurogen.ca.spectral_ca import SpectralCAGenome
from neurogen.ca.topo_ca import TopologicalCAGenome
from neurogen.ca.reaction_diffusion import ReactionDiffusionGenome
from neurogen.config import get_device


# Registry mapping variant name to genome class
CA_VARIANTS: dict[str, type[CAGenome]] = {
    "grid_ca": GridCAGenome,
    "neural_ca": NeuralCAGenome,
    "spectral_ca": SpectralCAGenome,
    "topo_ca": TopologicalCAGenome,
    "reaction_diffusion": ReactionDiffusionGenome,
}

# Default configs for each variant
_DEFAULT_CONFIGS: dict[str, dict[str, Any]] = {
    "grid_ca": {
        "hidden_dim": 64,
        "seed_pattern": "center",
    },
    "neural_ca": {
        "n_channels": 16,
        "hidden_dim": 64,
        "seed_pattern": "center",
        "p_update": 0.5,
    },
    "spectral_ca": {
        "n_freq": 16,
        "hidden_dim": 64,
        "seed_pattern": "center",
    },
    "topo_ca": {
        "hidden_dim": 64,
        "topology": "small_world",
        "seed_pattern": "center",
    },
    "reaction_diffusion": {
        "model_type": "gray_scott",
        "dt": 0.5,
        "seed_pattern": "center",
    },
}


class CAWeightEngine:
    """Engine that uses CA genomes to develop weight matrices for a GPT model.

    Creates a CA genome of the specified variant, then develops weight
    matrices for all weight parameters in the model. Tracks development
    time and provides compression ratio statistics.

    This is the main entry point for CA-based weight initialization.
    It can be used interchangeably with baseline initializers via the
    initialize() method.

    Args:
        variant: CA variant name (e.g. "grid_ca", "neural_ca").
        genome_config: Configuration dict passed to the genome constructor.
        device: Device string. Auto-detected if None.

    Raises:
        ValueError: If variant is not recognized.
    """

    def __init__(
        self,
        variant: str = "grid_ca",
        genome_config: dict[str, Any] | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize the CAWeightEngine.

        Args:
            variant: CA variant name.
            genome_config: Optional config dict for the genome constructor.
            device: Device string override. Auto-detected if None.

        Raises:
            ValueError: If variant is not in CA_VARIANTS.
        """
        if variant not in CA_VARIANTS:
            raise ValueError(
                f"Unknown CA variant '{variant}'. "
                f"Available: {sorted(CA_VARIANTS.keys())}"
            )

        self.variant = variant
        self.device = device or get_device()
        self._genome_config = genome_config or {}
        self._last_develop_time_ms: float = 0.0

        # Merge user config with defaults
        config = dict(_DEFAULT_CONFIGS.get(variant, {}))
        config.update(self._genome_config)
        config["device"] = self.device

        # Create genome
        genome_cls = CA_VARIANTS[variant]
        self.genome: CAGenome = genome_cls(**config)

    @classmethod
    def available_variants(cls) -> list[str]:
        """Return list of all registered CA variant names.

        Returns:
            Sorted list of variant name strings.
        """
        return sorted(CA_VARIANTS.keys())

    def develop_weights(
        self,
        model: nn.Module,
        n_steps: int = 64,
        seed_pattern: str = "center",
    ) -> dict[str, Tensor]:
        """Generate all weight matrices for the model using the CA genome.

        Iterates over model.get_weight_tensors() and develops each weight
        matrix using the CA genome. Each weight gets its own seed but
        shares the same genome (developmental program). Tracks total
        development time.

        Args:
            model: A GPT model with get_weight_tensors() method.
            n_steps: Number of CA development steps per weight.
            seed_pattern: Seed initialization pattern.

        Returns:
            Dictionary mapping parameter names to developed weight tensors.
        """
        target_weights = model.get_weight_tensors()
        developed: dict[str, Tensor] = {}

        start_time = time.time()

        for name, param in target_weights.items():
            shape = param.shape

            if len(shape) != 2:
                # Non-matrix weights: use small random init
                developed[name] = torch.randn(
                    shape, dtype=torch.float32, device=self.device
                ) * 0.02
                continue

            seed = self.genome.create_seed(
                shape, pattern=seed_pattern, noise_scale=0.001
            )
            weight = self.genome.develop(seed, shape, n_steps=n_steps)

            # Ensure output is on the correct device
            developed[name] = weight.to(
                device=param.device, dtype=torch.float32
            )

        elapsed_ms = (time.time() - start_time) * 1000.0
        self._last_develop_time_ms = elapsed_ms

        return developed

    def genome_size(self) -> int:
        """Total number of learnable parameters in the genome.

        Returns:
            Integer count of genome parameters.
        """
        return self.genome.genome_size()

    def compression_ratio(self, model: nn.Module) -> float:
        """Compute compression ratio: model_params / genome_params.

        This measures how much the genome compresses the weight
        specification. A ratio of 1000 means the genome is 1000x
        smaller than the weights it generates.

        Args:
            model: A model with get_weight_tensors() method.

        Returns:
            Compression ratio as a float. Higher means more compression.
            Returns float('inf') if genome has 0 parameters.
        """
        genome_params = self.genome_size()
        if genome_params == 0:
            return float("inf")

        # Count only the weight tensor params (not biases, LN, etc.)
        model_weight_params = sum(
            p.numel() for p in model.get_weight_tensors().values()
        )
        return model_weight_params / genome_params

    @property
    def last_develop_time_ms(self) -> float:
        """Wall-clock time of the last develop_weights() call in ms."""
        return self._last_develop_time_ms

    def initialize(self, model: nn.Module) -> dict[str, Tensor]:
        """Conforming initializer interface: initialize(model) -> weights.

        This allows the CAWeightEngine to be used interchangeably with
        baseline initializers that follow the same interface.

        Args:
            model: The model to initialize.

        Returns:
            Dictionary of developed weight tensors.
        """
        return self.develop_weights(model)

    def get_genome(self) -> CAGenome:
        """Get the underlying CA genome module.

        Returns:
            The CAGenome instance used by this engine.
        """
        return self.genome

    def __repr__(self) -> str:
        """String representation of the engine."""
        return (
            f"CAWeightEngine(variant='{self.variant}', "
            f"genome_size={self.genome_size()}, "
            f"device='{self.device}')"
        )
