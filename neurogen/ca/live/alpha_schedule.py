"""Alpha schedule controlling CA influence over training."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class AlphaSchedule:
    """Controls how the CA blending factor alpha changes over training.

    Supports multiple schedule modes that control the strength of the CA
    signal relative to gradient descent. The schedule reflects a biological
    developmental curve where the CA (developmental program) is strong
    early in training and fades as learning takes over.

    Modes:
        exponential_decay: alpha(t) = alpha_0 * exp(-decay * t)
        cosine: alpha(t) = alpha_0 * 0.5 * (1 + cos(pi * t / total_steps))
        phased: step function with configurable phase boundaries and alphas
        adaptive: adjusts alpha based on loss improvement rate
        cyclic: periodic developmental bursts (sleep/wake analog)

    Attributes:
        mode: Schedule mode name.
        alpha_0: Initial/base alpha value.
        decay: Decay rate for exponential mode.
        total_steps: Total training steps (for cosine and other schedules).
        phase_boundaries: Step boundaries for phased mode.
        phase_alphas: Alpha values for each phase.
        adaptive_sensitivity: Sensitivity for adaptive mode (unused, reserved).
        cycle_period: Period of cycles for cyclic mode.
        cycle_amplitude: Amplitude of cyclic oscillations.
        alpha_base: Base alpha for cyclic mode.
    """

    mode: str = "exponential_decay"
    alpha_0: float = 0.01
    decay: float = 0.001
    total_steps: int = 5000
    phase_boundaries: list[int] = field(
        default_factory=lambda: [0, 1000, 3000]
    )
    phase_alphas: list[float] = field(
        default_factory=lambda: [0.01, 0.005, 0.001]
    )
    adaptive_sensitivity: float = 1.0
    cycle_period: int = 500
    cycle_amplitude: float = 0.005
    alpha_base: float = 0.002

    def get_alpha(
        self,
        step: int,
        loss_history: list[float] | None = None,
    ) -> float:
        """Compute the alpha value at a given training step.

        Args:
            step: Current training step number.
            loss_history: Optional list of historical loss values. Required
                for adaptive mode to compute improvement rate.

        Returns:
            Alpha value (float) controlling CA influence at this step.

        Raises:
            ValueError: If mode is not recognized.
        """
        if self.mode == "exponential_decay":
            return self.alpha_0 * math.exp(-self.decay * step)

        elif self.mode == "cosine":
            t_ratio = step / max(self.total_steps, 1)
            return self.alpha_0 * 0.5 * (1.0 + math.cos(math.pi * t_ratio))

        elif self.mode == "phased":
            # Find which phase we're in based on boundaries
            for i, boundary in enumerate(self.phase_boundaries):
                if step < boundary:
                    return self.phase_alphas[max(0, i - 1)]
            return self.phase_alphas[-1]

        elif self.mode == "adaptive":
            if loss_history is not None and len(loss_history) > 10:
                recent_improvement = loss_history[-10] - loss_history[-1]
                if recent_improvement < 0.01:
                    # Loss stagnating: increase CA influence
                    return self.alpha_0 * 2.0
                else:
                    # Loss improving: reduce CA influence
                    return self.alpha_0 * 0.5
            return self.alpha_0

        elif self.mode == "cyclic":
            return self.alpha_base + self.cycle_amplitude * math.sin(
                2.0 * math.pi * step / self.cycle_period
            )

        else:
            raise ValueError(f"Unknown alpha schedule mode: {self.mode}")
