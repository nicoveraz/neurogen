"""Variant C: Spectral CA — operates in frequency domain.

CA rules generate/evolve Fourier coefficients; inverse FFT produces the weight matrix.
"""

import torch
import torch.nn as nn

from neurogen.ca.genome import CAGenome


class SpectralCAGenome(CAGenome):
    """Spectral CA genome — develops weights in frequency domain.

    Args:
        n_frequencies: Number of frequency components to use.
        hidden_dim: Width of frequency evolution MLP.
    """

    def __init__(
        self,
        n_frequencies: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_frequencies = n_frequencies

        # Evolution MLP: updates frequency coefficients each step
        # Input: (real_coeff, imag_coeff, freq_x, freq_y, step_frac) = 5
        self.evolution_net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),  # output: (delta_real, delta_imag)
        )

        # Seed frequencies — learned initial spectrum
        self.seed_real = nn.Parameter(torch.randn(n_frequencies, n_frequencies) * 0.01)
        self.seed_imag = nn.Parameter(torch.randn(n_frequencies, n_frequencies) * 0.01)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def develop(
        self,
        seed: torch.Tensor | None = None,
        target_shape: tuple[int, int] = (64, 64),
        n_steps: int = 32,
    ) -> torch.Tensor:
        """Develop weights by evolving frequency coefficients.

        Args:
            seed: Ignored (uses learned seed frequencies).
            target_shape: Desired output shape.
            n_steps: Number of evolution steps.

        Returns:
            Developed weight matrix.
        """
        device = next(self.parameters()).device
        H, W = target_shape
        nf = self.n_frequencies

        # Start from learned seed with seed-dependent noise
        real = self.seed_real.clone() + torch.randn(nf, nf, device=device) * 0.001
        imag = self.seed_imag.clone() + torch.randn(nf, nf, device=device) * 0.001

        # Create frequency coordinate grids
        fx = torch.linspace(0, 1, nf, device=device).unsqueeze(1).expand(nf, nf)
        fy = torch.linspace(0, 1, nf, device=device).unsqueeze(0).expand(nf, nf)

        for step in range(n_steps):
            step_frac = torch.tensor(
                step / max(1, n_steps - 1), device=device, dtype=torch.float32
            )

            # Stack features: (nf, nf, 5)
            features = torch.stack(
                [real, imag, fx, fy, step_frac.expand(nf, nf)], dim=-1
            )

            # Evolve frequencies
            delta = self.evolution_net(features)  # (nf, nf, 2)
            real = real + delta[..., 0] * 0.1
            imag = imag + delta[..., 1] * 0.1

        # Construct complex spectrum and pad/crop to target size
        spectrum = torch.zeros(H, W, dtype=torch.cfloat, device=device)
        h_freq = min(nf, H)
        w_freq = min(nf, W)
        spectrum[:h_freq, :w_freq] = (real[:h_freq, :w_freq] + 1j * imag[:h_freq, :w_freq])

        # Inverse FFT to spatial domain
        weight = torch.fft.ifft2(spectrum).real

        # Scale to initialization range
        weight = weight - weight.mean()
        if weight.std() > 1e-8:
            weight = weight / weight.std() * 0.02

        return weight
