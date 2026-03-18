"""CMA-ES meta-learning for CA genome optimization.

Uses Covariance Matrix Adaptation Evolution Strategy to optimize the
parameters of a CA genome such that the developed weights minimize
validation loss after a short training run.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from neurogen.ca.engine import CAWeightEngine
from neurogen.config import GPTConfig, TrainConfig, get_device
from neurogen.data.shakespeare import ShakespeareDataset
from neurogen.model.gpt import GPT
from neurogen.training.trainer import train


class CMAESMetaTrainer:
    """CMA-ES meta-learner for CA genome optimization.

    Optimizes a CA genome's parameters where fitness is the validation
    loss after a short training run with the genome's developed weights.
    Uses progressive evaluation: fewer steps early, more for promising genomes.

    Args:
        ca_variant: CA variant name to optimize.
        model_config: GPT model configuration.
        dataset: Dataset identifier string.
        train_config: Training config for inner-loop evaluation.
        device: Device string. Auto-detected if None.
        population_size: CMA-ES population size per generation.
        sigma: Initial CMA-ES step size.
    """

    def __init__(
        self,
        ca_variant: str = "grid_ca",
        model_config: GPTConfig | None = None,
        dataset: str = "shakespeare_char",
        train_config: TrainConfig | None = None,
        device: str | None = None,
        population_size: int = 10,
        sigma: float = 0.5,
    ) -> None:
        self.ca_variant = ca_variant
        self.device = device or get_device()
        self.population_size = population_size
        self.sigma = sigma
        self.model_config = model_config or GPTConfig()
        self.train_config = train_config or TrainConfig(
            max_steps=500, eval_interval=100, batch_size=32, lr=3e-4,
        )
        self._dataset: ShakespeareDataset | None = None
        self._ref_engine = CAWeightEngine(variant=ca_variant, device=self.device)
        self._genome_dim = self._ref_engine.genome_size()
        self._history: list[dict[str, Any]] = []

    def _get_dataset(self) -> ShakespeareDataset:
        """Lazily load and cache the dataset."""
        if self._dataset is None:
            self._dataset = ShakespeareDataset()
        return self._dataset

    def evaluate_genome(self, params: np.ndarray, max_steps: int | None = None) -> float:
        """Evaluate a genome: set params, develop weights, train, return val_loss.

        Args:
            params: Flat 1D numpy array of genome parameters.
            max_steps: Override for training steps.

        Returns:
            Final validation loss. Returns 100.0 on failure.
        """
        try:
            engine = CAWeightEngine(variant=self.ca_variant, device=self.device)
            engine.genome.set_params_flat(params)

            dataset = self._get_dataset()
            config = GPTConfig(
                block_size=self.model_config.block_size,
                vocab_size=dataset.vocab_size,
                n_layer=self.model_config.n_layer,
                n_head=self.model_config.n_head,
                n_embd=self.model_config.n_embd,
                dropout=self.model_config.dropout,
            )
            model = GPT(config)

            steps = max_steps or self.train_config.max_steps
            t_config = TrainConfig(
                max_steps=steps, eval_interval=max(steps // 5, 1),
                batch_size=self.train_config.batch_size,
                lr=self.train_config.lr, grad_clip=self.train_config.grad_clip,
                warmup_steps=min(self.train_config.warmup_steps, steps // 5),
                min_lr=self.train_config.min_lr,
            )

            def init_fn(m: GPT) -> dict[str, torch.Tensor]:
                return engine.develop_weights(m)

            metrics = train(
                model=model, dataset=dataset, config=t_config,
                init_fn=init_fn, device=self.device,
            )
            val_loss = metrics.get("final_val_loss", float("inf"))
            if val_loss != val_loss or val_loss == float("inf"):
                return 100.0
            return val_loss
        except Exception as e:
            print(f"Genome evaluation failed: {e}")
            return 100.0

    def run(self, n_generations: int = 50, progressive: bool = True) -> dict[str, Any]:
        """Run CMA-ES optimization for n_generations.

        Args:
            n_generations: Number of CMA-ES generations.
            progressive: If True, ramp up training steps over generations.

        Returns:
            Dict with best_params, best_fitness, history, total_time_s.
        """
        try:
            import cma
        except ImportError:
            raise ImportError(
                "CMA-ES requires the 'cma' package. Install with: pip install cma"
            )

        x0 = self._ref_engine.genome.get_params_flat()
        opts = cma.CMAOptions()
        opts.set("popsize", self.population_size)
        opts.set("maxiter", n_generations)
        opts.set("verb_disp", 0)
        opts.set("verb_log", 0)
        opts.set("verb_filenameprefix", "")
        opts.set("verbose", -1)

        es = cma.CMAEvolutionStrategy(x0.tolist(), self.sigma, opts)
        self._history = []
        best_params, best_fitness = x0.copy(), float("inf")
        start_time = time.time()
        base_steps = self.train_config.max_steps

        for gen in range(n_generations):
            if es.stop():
                print(f"CMA-ES converged at generation {gen}")
                break

            if progressive:
                frac = min(1.0, (gen + 1) / max(n_generations * 0.5, 1))
                current_steps = max(50, int(base_steps * (0.2 + 0.8 * frac)))
            else:
                current_steps = base_steps

            candidates = es.ask()
            fitnesses = [
                self.evaluate_genome(np.array(c, dtype=np.float32), current_steps)
                for c in candidates
            ]
            es.tell(candidates, fitnesses)

            gi = int(np.argmin(fitnesses))
            gf = fitnesses[gi]
            gp = np.array(candidates[gi], dtype=np.float32)
            if gf < best_fitness:
                best_fitness, best_params = gf, gp.copy()

            self._history.append({
                "generation": gen, "best_fitness": gf,
                "mean_fitness": float(np.mean(fitnesses)),
                "std_fitness": float(np.std(fitnesses)),
                "overall_best": best_fitness,
                "training_steps": current_steps,
                "sigma": float(es.sigma),
            })
            elapsed = time.time() - start_time
            print(
                f"Gen {gen:3d} | best: {gf:.4f} | mean: {np.mean(fitnesses):.4f} "
                f"| overall: {best_fitness:.4f} | steps: {current_steps} "
                f"| time: {elapsed:.0f}s"
            )

        return {
            "best_params": best_params, "best_fitness": best_fitness,
            "history": self._history, "total_time_s": time.time() - start_time,
            "n_generations": len(self._history),
            "genome_dim": self._genome_dim, "ca_variant": self.ca_variant,
        }

    @property
    def genome_dim(self) -> int:
        """Dimensionality of the genome parameter space."""
        return self._genome_dim

    @property
    def history(self) -> list[dict[str, Any]]:
        """Per-generation optimization history."""
        return self._history
