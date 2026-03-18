"""Spectral Cellular Automata genome for weight development.

Operates in the frequency domain. CA rules generate and modify Fourier
coefficients, and an inverse FFT produces the final weight matrix. The
hypothesis is that useful weight structure is easier to express spectrally
(e.g., low-rank structure, periodicity, smooth gradients).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor

from neurogen.ca.genome import CAGenome


class SpectralCAGenome(CAGenome):
    """Spectral CA genome operating in the frequency domain.

    The genome consists of:
    - Initial learnable spectrum parameters (low-frequency components)
    - An MLP that iteratively refines the Fourier coefficients
    - Inverse FFT to produce the final weight matrix

    The CA steps iteratively evolve the spectrum, allowing complex
    frequency interactions to emerge. Only a subset of frequency
    components are directly parameterized (the low-frequency ones),
    providing compression.

    Args:
        n_freq: Number of frequency components per dimension to
            directly parameterize. Higher = more expressiveness but
            less compression.
        hidden_dim: Hidden dimension of the spectral update MLP.
        seed_pattern: Seed initialization pattern.
        device: Device string.
    """

    def __init__(
        self,
        n_freq: int = 16,
        hidden_dim: int = 64,
        seed_pattern: str = "center",
        device: str = "cpu",
    ) -> None:
        """Initialize the SpectralCAGenome.

        Args:
            n_freq: Number of frequency components per dimension.
            hidden_dim: Hidden dimension for the spectral update MLP.
            seed_pattern: Seed initialization pattern.
            device: Device string ("cpu", "cuda", or "mps").
        """
        super().__init__(device=device)
        self.n_freq = n_freq
        self.hidden_dim = hidden_dim
        self.seed_pattern = seed_pattern

        # Spectral update MLP: processes each frequency component
        # Input: (real, imag, freq_x, freq_y, magnitude, phase, step_frac)
        n_input = 7
        self.spectral_mlp = nn.Sequential(
            nn.Linear(n_input, hidden_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2, dtype=torch.float32),  # delta real, imag
        )

        # Initialize with small weights for stability
        for layer in self.spectral_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

        self.to(device)

    def _init_spectrum(
        self, h: int, w: int, noise_scale: float = 0.001
    ) -> Tensor:
        """Initialize the frequency spectrum with noise.

        Creates a complex spectrum with low-frequency bias and random
        noise so different torch seeds produce different outputs.

        Args:
            h: Height of the target matrix.
            w: Width of the target matrix.
            noise_scale: Scale of initial noise.

        Returns:
            Complex spectrum tensor of shape (h, w).
        """
        fh = min(self.n_freq, h)
        fw = min(self.n_freq, w)

        # Full spectrum initialized to zero
        spectrum_real = torch.zeros(
            h, w, dtype=torch.float32, device=self._device_str
        )
        spectrum_imag = torch.zeros(
            h, w, dtype=torch.float32, device=self._device_str
        )

        # Low-frequency components get larger initial values
        for i in range(fh):
            for j in range(fw):
                # Decay with frequency
                decay = 1.0 / (1.0 + math.sqrt(i * i + j * j))
                spectrum_real[i, j] = decay * 0.1
                spectrum_imag[i, j] = 0.0

        # Add noise for seed-dependent variation
        spectrum_real = spectrum_real + torch.randn_like(spectrum_real) * noise_scale
        spectrum_imag = spectrum_imag + torch.randn_like(spectrum_imag) * noise_scale

        return torch.complex(spectrum_real, spectrum_imag)

    def _build_freq_features(
        self, spectrum: Tensor, step_fraction: float
    ) -> Tensor:
        """Build per-component features from the current spectrum.

        Args:
            spectrum: Complex spectrum of shape (H, W).
            step_fraction: Current step / total steps.

        Returns:
            Feature tensor of shape (H*W, 7).
        """
        h, w = spectrum.shape

        real_part = spectrum.real.reshape(-1, 1)
        imag_part = spectrum.imag.reshape(-1, 1)

        # Frequency coordinates normalized to [0, 1]
        freq_y = torch.linspace(0, 1, h, device=spectrum.device,
                                dtype=torch.float32)
        freq_x = torch.linspace(0, 1, w, device=spectrum.device,
                                dtype=torch.float32)
        fy, fx = torch.meshgrid(freq_y, freq_x, indexing="ij")
        fy = fy.reshape(-1, 1)
        fx = fx.reshape(-1, 1)

        # Magnitude and phase
        magnitude = torch.abs(spectrum).reshape(-1, 1)
        phase = torch.angle(spectrum).reshape(-1, 1)

        # Step fraction
        step_feat = torch.full(
            (h * w, 1), step_fraction,
            dtype=torch.float32, device=spectrum.device
        )

        return torch.cat(
            [real_part, imag_part, fx, fy, magnitude, phase, step_feat],
            dim=1,
        )

    def develop(
        self, seed: Tensor, target_shape: tuple[int, ...], n_steps: int = 64
    ) -> Tensor:
        """Run spectral CA to develop a weight matrix.

        Initializes a frequency spectrum, iteratively refines it using
        the spectral MLP, then applies inverse FFT to produce the
        spatial-domain weight matrix.

        Args:
            seed: Seed tensor (used for noise seeding; the actual
                spectrum is generated internally).
            target_shape: Shape of the output weight matrix (H, W).
            n_steps: Number of spectral evolution steps.

        Returns:
            Developed weight matrix of shape (H, W).
        """
        assert len(target_shape) == 2, (
            f"SpectralCA expects 2D target shape, got {target_shape}"
        )
        h, w = target_shape

        # Initialize spectrum (noise ensures seed-dependent variation)
        spectrum = self._init_spectrum(h, w, noise_scale=0.001)

        for step in range(n_steps):
            step_fraction = step / max(n_steps - 1, 1)

            # Build features for each frequency component
            features = self._build_freq_features(spectrum, step_fraction)

            # Apply spectral MLP to get deltas
            delta = self.spectral_mlp(features)  # (H*W, 2)
            delta_real = delta[:, 0].reshape(h, w)
            delta_imag = delta[:, 1].reshape(h, w)

            # Apply delta with small step size
            spectrum = torch.complex(
                spectrum.real + delta_real * 0.1,
                spectrum.imag + delta_imag * 0.1,
            )

        # Inverse FFT to get spatial domain weight matrix
        # Move to CPU for inverse FFT if needed (MPS may not support all ops)
        compute_device = spectrum.device
        if str(compute_device).startswith("mps"):
            spectrum_cpu = spectrum.cpu()
            weights = torch.fft.ifft2(spectrum_cpu).real.to(compute_device)
        else:
            weights = torch.fft.ifft2(spectrum).real

        # Scale to reasonable magnitude
        w_std = weights.std().clamp(min=1e-8)
        weights = weights * (0.02 / w_std)

        return weights

    def forward(
        self, target_shape: tuple[int, ...], n_steps: int = 64
    ) -> Tensor:
        """Convenience forward pass that creates seed and develops.

        Args:
            target_shape: Shape of the output weight matrix (H, W).
            n_steps: Number of spectral evolution steps.

        Returns:
            Developed weight matrix.
        """
        seed = self.create_seed(
            target_shape, self.seed_pattern, noise_scale=0.001
        )
        return self.develop(seed, target_shape, n_steps)
