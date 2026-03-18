"""Base class for CA developmental programs (genomes)."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


class CAGenome(nn.Module):
    """Base class for CA developmental programs.

    A CA genome is a small neural network whose parameters encode a
    developmental rule. When applied iteratively to a seed tensor, it grows
    a full-sized weight matrix. The genome parameters are the "DNA" that
    can be optimized via evolution (CMA-ES) or gradient-based meta-learning.

    Subclasses must implement `develop()`.

    Attributes:
        device: The device on which tensors are created.
    """

    def __init__(self, device: str = "cpu") -> None:
        """Initialize the CAGenome.

        Args:
            device: Device string ("cpu", "cuda", or "mps").
        """
        super().__init__()
        self._device_str = device

    def develop(
        self, seed: Tensor, target_shape: tuple[int, ...], n_steps: int = 64
    ) -> Tensor:
        """Run CA for n_steps and return developed weight matrix.

        Args:
            seed: Initial seed tensor to start development from.
            target_shape: Desired shape of the output weight matrix.
            n_steps: Number of CA iteration steps.

        Returns:
            Developed weight matrix of shape target_shape.

        Raises:
            NotImplementedError: Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement develop()")

    def genome_size(self) -> int:
        """Total number of learnable parameters in this genome.

        Returns:
            Integer count of all learnable scalar parameters.
        """
        return sum(p.numel() for p in self.parameters())

    def get_params_flat(self) -> np.ndarray:
        """Flatten all parameters to a 1D numpy array.

        This is used by CMA-ES and other evolution strategies that operate
        on flat parameter vectors.

        Returns:
            1D numpy array containing all parameter values.
        """
        params: list[np.ndarray] = []
        for p in self.parameters():
            params.append(p.data.cpu().numpy().flatten())
        if not params:
            return np.array([], dtype=np.float32)
        return np.concatenate(params).astype(np.float32)

    def set_params_flat(self, params: np.ndarray) -> None:
        """Set parameters from a 1D numpy array.

        The array must have exactly genome_size() elements. Parameters are
        filled in the same order as self.parameters().

        Args:
            params: 1D numpy array of parameter values.

        Raises:
            ValueError: If params length does not match genome_size().
        """
        expected = self.genome_size()
        if len(params) != expected:
            raise ValueError(
                f"Expected {expected} params, got {len(params)}"
            )
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(
                torch.from_numpy(
                    params[offset : offset + n].reshape(p.shape)
                ).to(p.device, dtype=torch.float32)
            )
            offset += n

    def create_seed(
        self,
        target_shape: tuple[int, ...],
        pattern: str = "center",
        noise_scale: float = 0.001,
    ) -> Tensor:
        """Create a seed tensor for CA development.

        Args:
            target_shape: Shape of the target weight matrix (H, W) or
                (C, H, W) for multi-channel variants.
            pattern: Seed initialization pattern. One of "center" or "random".
            noise_scale: Scale of random noise added to ensure different
                torch seeds produce different outputs.

        Returns:
            Seed tensor of the given shape on this genome's device.

        Raises:
            ValueError: If pattern is not recognized.
        """
        seed = torch.zeros(target_shape, dtype=torch.float32,
                           device=self._device_str)

        if pattern == "center":
            # Initialize a small center region
            spatial_dims = target_shape[-2:]
            ch = spatial_dims[0] // 2
            cw = spatial_dims[1] // 2
            rh = max(1, spatial_dims[0] // 8)
            rw = max(1, spatial_dims[1] // 8)
            if len(target_shape) == 2:
                seed[ch - rh : ch + rh + 1, cw - rw : cw + rw + 1] = 1.0
            else:
                seed[:, ch - rh : ch + rh + 1, cw - rw : cw + rw + 1] = 1.0
        elif pattern == "random":
            seed = torch.randn(
                target_shape, dtype=torch.float32,
                device=self._device_str
            ) * 0.1
        else:
            raise ValueError(f"Unknown seed pattern: {pattern}")

        # Add noise so different torch seeds produce different outputs
        noise = torch.randn_like(seed) * noise_scale
        seed = seed + noise
        return seed
