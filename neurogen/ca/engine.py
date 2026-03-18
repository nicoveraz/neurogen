"""CAWeightEngine: dispatches to CA variants and handles full model initialization."""

import time
from dataclasses import dataclass

import torch

from neurogen.ca.genome import CAGenome
from neurogen.ca.grid_ca import GridCAGenome
from neurogen.ca.neural_ca import NeuralCAGenome
from neurogen.ca.reaction_diffusion import ReactionDiffusionGenome
from neurogen.ca.spectral_ca import SpectralCAGenome
from neurogen.ca.topo_ca import TopologicalCAGenome
from neurogen.config import CAConfig
from neurogen.model.gpt import GPT


# Registry of CA variants
CA_VARIANTS: dict[str, type[CAGenome]] = {
    "grid_ca": GridCAGenome,
    "neural_ca": NeuralCAGenome,
    "spectral_ca": SpectralCAGenome,
    "topo_ca": TopologicalCAGenome,
    "reaction_diffusion": ReactionDiffusionGenome,
}


def _genome_kwargs(variant: str, config: CAConfig) -> dict:
    """Extract variant-specific kwargs from a CAConfig."""
    if variant == "grid_ca":
        return {
            "hidden_dim": config.hidden_dim,
            "neighborhood": config.neighborhood,
            "boundary": config.boundary,
            "seed_pattern": config.seed_pattern,
        }
    elif variant == "neural_ca":
        return {
            "n_channels": config.n_channels,
            "hidden_dim": config.hidden_dim,
            "stochastic_rate": config.stochastic_rate,
        }
    elif variant == "spectral_ca":
        return {"hidden_dim": config.hidden_dim}
    elif variant == "topo_ca":
        return {"hidden_dim": config.hidden_dim}
    elif variant == "reaction_diffusion":
        return {}
    return {}


class CAWeightEngine:
    """Engine for developing weight matrices using cellular automata.

    Args:
        variant: Name of the CA variant to use.
        config: CA configuration. Can be a CAConfig or a dict.
        device: Device to run development on.
    """

    def __init__(
        self,
        variant: str = "grid_ca",
        config: CAConfig | dict | None = None,
        device: str = "cpu",
    ) -> None:
        if variant not in CA_VARIANTS:
            available = ", ".join(sorted(CA_VARIANTS.keys()))
            raise ValueError(
                f"Unknown CA variant '{variant}'. Available: {available}"
            )

        self.variant = variant

        if config is None:
            config = CAConfig(variant=variant)
        if isinstance(config, dict):
            ca_config = CAConfig(variant=variant)
            for k, v in config.items():
                if hasattr(ca_config, k):
                    setattr(ca_config, k, v)
            config = ca_config

        self.config = config
        self.device = device

        # Create genome
        kwargs = _genome_kwargs(variant, config)
        self.genome: CAGenome = CA_VARIANTS[variant](**kwargs)
        self.genome = self.genome.to(device)

    @staticmethod
    def available_variants() -> list[str]:
        """Return list of available CA variant names."""
        return sorted(CA_VARIANTS.keys())

    def develop_weights(
        self,
        model: GPT,
        seed: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generate all weight matrices for the model.

        Args:
            model: The GPT model whose weight shapes to target.
            seed: Optional random seed for reproducibility.

        Returns:
            Dict mapping weight names to developed tensors.
        """
        if seed is not None:
            torch.manual_seed(seed)

        target_weights = model.get_weight_tensors()
        developed = {}

        for name, param in target_weights.items():
            shape = param.shape
            assert len(shape) == 2, f"Expected 2D weight, got {len(shape)}D for {name}"
            w = self.genome.develop(
                seed=None,
                target_shape=(shape[0], shape[1]),
                n_steps=self.config.n_steps,
            )
            developed[name] = w.to(param.device)

        return developed

    def genome_size(self) -> int:
        """Total parameters in the developmental program."""
        return self.genome.genome_size()

    def compression_ratio(self, model: GPT) -> float:
        """Ratio: model_weight_params / genome_params."""
        model_params = sum(
            p.numel() for p in model.get_weight_tensors().values()
        )
        genome_params = self.genome_size()
        if genome_params == 0:
            return float("inf")
        return model_params / genome_params

    def get_genome_params(self) -> torch.Tensor:
        """Get genome parameters as a flat numpy-compatible vector."""
        return self.genome.get_params_flat().detach().cpu()

    def set_genome_params(self, params: torch.Tensor) -> None:
        """Set genome parameters from a flat vector."""
        self.genome.set_params_flat(params.to(self.device))


def initialize_with_ca(
    model: GPT,
    variant: str = "grid_ca",
    config: CAConfig | dict | None = None,
    device: str = "cpu",
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Convenience function matching the baseline initializer interface.

    Args:
        model: The GPT model.
        variant: CA variant name.
        config: CA configuration.
        device: Device for CA development.
        seed: Random seed.

    Returns:
        Dict of developed weight tensors.
    """
    engine = CAWeightEngine(variant=variant, config=config, device=device)
    return engine.develop_weights(model, seed=seed)
