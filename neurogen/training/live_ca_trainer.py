"""Training loop with CA operating alongside gradient descent.

Implements: w(t+1) = w(t) - lr * nabla_L(w(t)) + alpha(t) * CA(w(t))
"""

from __future__ import annotations

import time
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurogen.ca.live.base import LiveCA
from neurogen.ca.live.alpha_schedule import AlphaSchedule
from neurogen.ca.live.multi_timescale import MultiTimescaleCA
from neurogen.config import LiveCAConfig


def measure_alignment(ca_delta: torch.Tensor, grad_delta: torch.Tensor) -> float:
    """Cosine similarity between CA and gradient updates.

    Returns value in [-1, 1]: +1 cooperation, 0 orthogonal, -1 competition.
    Returns 0.0 if either vector has zero norm.
    """
    flat_ca = ca_delta.flatten().unsqueeze(0)
    flat_grad = grad_delta.flatten().unsqueeze(0)
    if flat_ca.norm() < 1e-12 or flat_grad.norm() < 1e-12:
        return 0.0
    return F.cosine_similarity(flat_ca, flat_grad).item()


def _match_param(name: str, ca_rules: dict[str, LiveCA]) -> LiveCA | None:
    """Find the CA rule matching a parameter name by substring pattern."""
    for pattern, ca in ca_rules.items():
        if pattern in name:
            return ca
    return None


class LiveCATrainer:
    """Training loop with CA operating alongside gradient descent.

    CA rules are mapped to parameters by name patterns. Different weight
    matrices can use different CA rules (e.g., attention weights use
    CompetitionCA while FFN weights use PruningCA).

    Args:
        model: The GPT model to train.
        ca_rules: Pattern-to-LiveCA mapping, e.g. {"attn": CompetitionCA()}.
        alpha_schedule: Schedule controlling CA influence over training.
        config: Live CA configuration.
        device: Device string ("cpu", "cuda", or "mps").
    """

    def __init__(
        self,
        model: nn.Module,
        ca_rules: dict[str, LiveCA],
        alpha_schedule: AlphaSchedule,
        config: LiveCAConfig,
        device: str,
    ) -> None:
        self.model = model
        self.ca_rules = ca_rules
        self.alpha_schedule = alpha_schedule
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.alpha_0, weight_decay=0.1,
        )

        # Build param name -> CA rule mapping at init time
        self._param_ca_map: dict[str, LiveCA] = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.dim() < 2:
                continue
            ca = _match_param(name, self.ca_rules)
            if ca is not None:
                self._param_ca_map[name] = ca

    def configure_optimizer(
        self, lr: float = 3e-4, weight_decay: float = 0.1
    ) -> None:
        """Reconfigure the optimizer with new hyperparameters."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay,
        )

    def train_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        step: int,
        loss_history: list[float],
    ) -> dict[str, Any]:
        """Single training step with CA integration.

        Returns dict with: loss, ca_delta_magnitude, gradient_delta_magnitude,
        ca_gradient_alignment (per-param), ca_contribution_ratio.
        """
        self.model.train()
        x, y = batch

        # 1. Forward + backward
        logits, loss = self.model(x, y)
        loss.backward()

        # 2. Collect gradients before optimizer step
        gradients: dict[str, torch.Tensor] = {}
        if self.config.ca_sees_gradients:
            params_dict = dict(self.model.named_parameters())
            for name in self._param_ca_map:
                param = params_dict[name]
                if param.grad is not None:
                    gradients[name] = param.grad.clone().detach()

        # 3. Optimizer step
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        # 4. CA step (if interval condition met)
        ca_norms: list[float] = []
        grad_norms: list[float] = []
        alignments: dict[str, float] = {}
        total_ca = 0.0
        total_grad = 0.0

        if step % self.config.ca_interval == 0:
            alpha = self.alpha_schedule.get_alpha(step, loss_history)

            with torch.no_grad():
                params_dict = dict(self.model.named_parameters())
                for name, ca in self._param_ca_map.items():
                    param = params_dict[name]
                    grad = gradients.get(name)

                    if isinstance(ca, MultiTimescaleCA):
                        ca_delta = ca.step(param.data, grad_W=grad, step_number=step)
                    else:
                        ca_delta = ca.step(param.data, grad_W=grad)

                    param.data.add_(alpha * ca_delta)

                    if self.config.clamp_weights:
                        param.data.clamp_(-self.config.max_weight, self.config.max_weight)

                    cn = (alpha * ca_delta).norm().item()
                    ca_norms.append(cn)
                    total_ca += cn

                    if grad is not None:
                        gn = grad.norm().item()
                        grad_norms.append(gn)
                        total_grad += gn
                        alignments[name] = measure_alignment(ca_delta, grad)

        mean_ca = sum(ca_norms) / len(ca_norms) if ca_norms else 0.0
        mean_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        denom = total_ca + total_grad
        ratio = total_ca / denom if denom > 0 else 0.0

        return {
            "loss": loss.item(),
            "ca_delta_magnitude": mean_ca,
            "gradient_delta_magnitude": mean_grad,
            "ca_gradient_alignment": alignments,
            "ca_contribution_ratio": ratio,
        }

    def train(
        self,
        dataset: Any,
        callback: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Full training loop with CA. Returns metrics including CA-specific ones.

        Args:
            dataset: Object with get_batch(split, batch_size, block_size, device).
            callback: Optional (step, metrics) callback after each step.
        """
        self.model.to(self.device)
        for ca in self.ca_rules.values():
            ca.to(self.device)

        losses: list[float] = []
        ca_mags: list[float] = []
        grad_mags: list[float] = []
        ca_aligns: list[dict[str, float]] = []
        ca_contribs: list[float] = []
        loss_history: list[float] = []

        model_cfg = getattr(self.model, "config", None)
        block_size = model_cfg.block_size if model_cfg else 256
        batch_size = 64

        start_time = time.time()

        for step in range(self.config.total_steps):
            batch = dataset.get_batch("train", batch_size, block_size, self.device)
            metrics = self.train_step(batch, step, loss_history)

            losses.append(metrics["loss"])
            loss_history.append(metrics["loss"])
            ca_mags.append(metrics["ca_delta_magnitude"])
            grad_mags.append(metrics["gradient_delta_magnitude"])
            ca_aligns.append(metrics["ca_gradient_alignment"])
            ca_contribs.append(metrics["ca_contribution_ratio"])

            if callback is not None:
                callback(step, metrics)

        return {
            "train_losses": losses,
            "ca_magnitudes": ca_mags,
            "grad_magnitudes": grad_mags,
            "ca_alignments": ca_aligns,
            "ca_contributions": ca_contribs,
            "total_time": time.time() - start_time,
            "final_loss": losses[-1] if losses else float("inf"),
        }
