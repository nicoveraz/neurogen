"""Base class for CA genomes — the developmental programs that grow weights."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class CAGenome(nn.Module, ABC):
    """Base class for CA developmental programs.

    A genome is a small neural network that defines the rules for how
    a cellular automaton develops weight matrices from seeds.
    """

    @abstractmethod
    def develop(
        self,
        seed: torch.Tensor,
        target_shape: tuple[int, int],
        n_steps: int = 64,
    ) -> torch.Tensor:
        """Run the CA for n_steps to develop a weight matrix.

        Args:
            seed: Initial seed tensor.
            target_shape: Desired output shape (rows, cols).
            n_steps: Number of CA development steps.

        Returns:
            Developed weight matrix of target_shape.
        """
        ...

    def genome_size(self) -> int:
        """Total number of parameters in this genome."""
        return sum(p.numel() for p in self.parameters())

    def get_params_flat(self) -> torch.Tensor:
        """Get all genome parameters as a flat vector."""
        return torch.cat([p.data.flatten() for p in self.parameters()])

    def set_params_flat(self, flat: torch.Tensor) -> None:
        """Set all genome parameters from a flat vector."""
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[offset : offset + n].reshape(p.shape))
            offset += n
