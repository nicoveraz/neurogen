"""Meta-learning: CMA-ES genome optimization for CA weight engines."""

from dataclasses import dataclass, field

import numpy as np
import torch

from neurogen.ca.engine import CAWeightEngine
from neurogen.config import GPTConfig, get_device
from neurogen.data.shakespeare import ShakespeareDataset
from neurogen.model.gpt import GPT
from neurogen.training.evaluator import train_and_evaluate


@dataclass
class MetaTrainConfig:
    """Configuration for CMA-ES meta-learning."""

    population_size: int = 20
    max_generations: int = 100
    inner_train_steps: int = 2000
    inner_seeds: int = 2
    sigma0: float = 0.1
    early_stop_loss: float = 10.0
    model_config: GPTConfig | None = None
    device: str = ""
    batch_size: int = 32
    lr: float = 3e-4
    progressive_schedule: dict = field(
        default_factory=lambda: {0: 500, 30: 1000, 60: 2000}
    )

    def __post_init__(self) -> None:
        if not self.device:
            self.device = get_device()


@dataclass
class MetaTrainResults:
    """Results from CMA-ES meta-learning."""

    best_genome: np.ndarray | None = None
    best_fitness: float = float("inf")
    history: list = field(default_factory=list)


class CMAESMetaTrainer:
    """Optimize CA genome parameters using CMA-ES.

    Args:
        variant: CA variant name.
        arch_config: Architecture config dict for the CA.
        meta_config: Meta-training configuration.
    """

    def __init__(
        self,
        variant: str,
        arch_config: dict | None = None,
        meta_config: MetaTrainConfig | None = None,
    ) -> None:
        self.variant = variant
        self.arch_config = arch_config or {}
        self.meta_config = meta_config or MetaTrainConfig()

        self.engine = CAWeightEngine(
            variant=variant,
            config=arch_config,
            device="cpu",  # develop on CPU for compatibility
        )
        self.genome_dim = self.engine.genome_size()
        self.dataset = ShakespeareDataset()

        if self.meta_config.model_config is None:
            self.meta_config.model_config = GPTConfig(
                block_size=32,
                vocab_size=self.dataset.vocab_size,
                n_layer=2,
                n_head=2,
                n_embd=64,
                dropout=0.0,
            )

    def evaluate_genome(
        self, genome_params: np.ndarray, generation: int
    ) -> float:
        """Inner loop: set genome, develop weights, train, return val_loss.

        Args:
            genome_params: Flat genome parameter vector.
            generation: Current generation number.

        Returns:
            Mean validation loss across seeds.
        """
        self.engine.set_genome_params(torch.tensor(genome_params, dtype=torch.float32))

        inner_steps = self._get_progressive_steps(generation)
        losses = []

        for seed in range(self.meta_config.inner_seeds):
            torch.manual_seed(seed + 1000)
            model = GPT(self.meta_config.model_config)
            weights = self.engine.develop_weights(model)
            model.set_weight_tensors(weights)

            val_loss = train_and_evaluate(
                model,
                self.dataset,
                steps=inner_steps,
                batch_size=self.meta_config.batch_size,
                lr=self.meta_config.lr,
                device=self.meta_config.device,
            )

            if val_loss > self.meta_config.early_stop_loss:
                return val_loss

            losses.append(val_loss)

        return float(np.mean(losses))

    def run(self) -> MetaTrainResults:
        """Outer loop: CMA-ES optimization of genome.

        Returns:
            MetaTrainResults with best genome and history.
        """
        try:
            import cma
        except ImportError:
            raise ImportError(
                "CMA-ES requires the 'cma' package. Install with: pip install cma"
            )

        x0 = self.engine.get_genome_params().numpy()
        es = cma.CMAEvolutionStrategy(
            x0,
            self.meta_config.sigma0,
            {
                "popsize": self.meta_config.population_size,
                "maxiter": self.meta_config.max_generations,
                "seed": 42,
                "verb_disp": 1,
            },
        )

        history = []
        for generation in range(self.meta_config.max_generations):
            solutions = es.ask()
            fitnesses = [
                self.evaluate_genome(s, generation) for s in solutions
            ]
            es.tell(solutions, fitnesses)

            gen_record = {
                "generation": generation,
                "best_fitness": min(fitnesses),
                "mean_fitness": float(np.mean(fitnesses)),
                "std_fitness": float(np.std(fitnesses)),
                "sigma": float(es.sigma),
            }
            history.append(gen_record)
            print(
                f"Gen {generation:3d} | "
                f"best={min(fitnesses):.4f} | "
                f"mean={np.mean(fitnesses):.4f} | "
                f"sigma={es.sigma:.4f}"
            )

            if es.stop():
                print("CMA-ES converged.")
                break

        return MetaTrainResults(
            best_genome=es.result.xbest,
            best_fitness=es.result.fbest,
            history=history,
        )

    def _get_progressive_steps(self, generation: int) -> int:
        """More training steps as search progresses."""
        steps = 500
        for gen_threshold, step_count in sorted(
            self.meta_config.progressive_schedule.items()
        ):
            if generation >= gen_threshold:
                steps = step_count
        return steps
