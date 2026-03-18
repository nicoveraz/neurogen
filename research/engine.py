"""Experiment runner: YAML-driven experiment execution with metrics collection."""

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from neurogen.baselines.initializers import INITIALIZERS, get_initializer
from neurogen.ca.engine import CAWeightEngine, CA_VARIANTS
from neurogen.config import CAConfig, ExperimentConfig, GPTConfig, TrainConfig, get_device
from neurogen.data.shakespeare import ShakespeareDataset
from neurogen.model.gpt import GPT
from neurogen.training.trainer import train
from research.registry import ExperimentRegistry


def load_experiment_config(yaml_path: str | Path) -> dict:
    """Load and validate an experiment YAML file.

    Args:
        yaml_path: Path to the YAML file.

    Returns:
        Parsed experiment configuration dict.

    Raises:
        ValueError: If required fields are missing.
    """
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    required = ["name", "model", "training", "inits"]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    return config


def _build_model_config(model_dict: dict, vocab_size: int) -> GPTConfig:
    """Build GPTConfig from a dict, handling 'default_tiny' shorthand."""
    if isinstance(model_dict, str) and model_dict == "default_tiny":
        return GPTConfig(
            block_size=32, vocab_size=vocab_size,
            n_layer=2, n_head=2, n_embd=64, dropout=0.0,
        )
    config_dict = model_dict.get("config", model_dict)
    if isinstance(config_dict, str) and config_dict == "default_tiny":
        return GPTConfig(
            block_size=32, vocab_size=vocab_size,
            n_layer=2, n_head=2, n_embd=64, dropout=0.0,
        )
    return GPTConfig(
        block_size=config_dict.get("block_size", 256),
        vocab_size=vocab_size,
        n_layer=config_dict.get("n_layer", 6),
        n_head=config_dict.get("n_head", 6),
        n_embd=config_dict.get("n_embd", 384),
        dropout=config_dict.get("dropout", 0.2),
    )


def _build_train_config(train_dict: dict, device: str) -> TrainConfig:
    """Build TrainConfig from a dict."""
    return TrainConfig(
        max_steps=train_dict.get("max_steps", 5000),
        eval_interval=train_dict.get("eval_interval", 250),
        eval_steps=train_dict.get("eval_steps", 20),
        batch_size=train_dict.get("batch_size", 64),
        lr=train_dict.get("lr", 3e-4),
        device=device,
        log_interval=train_dict.get("log_interval", 10),
    )


def _apply_init(model: GPT, init_name: str, device: str) -> None:
    """Apply an initialization strategy to a model.

    Args:
        model: The model to initialize.
        init_name: Name of the initializer (baseline or CA variant).
        device: Device string.
    """
    if init_name in INITIALIZERS:
        weights = get_initializer(init_name)(model)
        model.set_weight_tensors(weights)
    elif init_name in CA_VARIANTS:
        engine = CAWeightEngine(variant=init_name, device=device)
        weights = engine.develop_weights(model)
        model.set_weight_tensors(weights)
    elif init_name == "default":
        pass  # Use PyTorch default init
    else:
        raise ValueError(f"Unknown init: {init_name}")


def run_single_experiment(
    init_name: str,
    model_config: GPTConfig,
    train_config: TrainConfig,
    dataset: ShakespeareDataset,
    seed: int = 42,
) -> dict:
    """Run a single training experiment.

    Args:
        init_name: Initialization strategy name.
        model_config: Model configuration.
        train_config: Training configuration.
        dataset: The dataset.
        seed: Random seed.

    Returns:
        Dict of metrics from the training run.
    """
    torch.manual_seed(seed)
    model = GPT(model_config)
    _apply_init(model, init_name, train_config.device)
    metrics = train(model, dataset, train_config)
    metrics["init_method"] = init_name
    metrics["seed"] = seed
    return metrics


def run_experiment_from_yaml(
    yaml_path: str | Path,
    output_dir: str | Path = "outputs/experiments",
    device: str = "",
    registry: ExperimentRegistry | None = None,
) -> dict:
    """Run a full experiment defined by a YAML file.

    Args:
        yaml_path: Path to experiment YAML.
        output_dir: Directory for outputs.
        device: Device override (empty = auto-detect).
        registry: Optional experiment registry.

    Returns:
        Dict with all results.
    """
    yaml_path = Path(yaml_path)
    output_dir = Path(output_dir)
    device = device or get_device()

    config = load_experiment_config(yaml_path)
    exp_name = config["name"]
    exp_id = yaml_path.stem

    # Setup output directory
    exp_dir = output_dir / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Register experiment
    if registry:
        registry.register(exp_id, exp_name, str(yaml_path), str(exp_dir))
        registry.mark_running(exp_id)

    try:
        # Load dataset
        dataset = ShakespeareDataset()

        # Build configs
        model_config = _build_model_config(config["model"], dataset.vocab_size)
        train_config = _build_train_config(config["training"], device)

        inits = config["inits"]
        seeds = config.get("seeds", [42])

        all_results = {}
        for init_name in inits:
            all_results[init_name] = []
            for seed in seeds:
                print(f"\n=== {exp_name}: {init_name}, seed={seed} ===")
                metrics = run_single_experiment(
                    init_name, model_config, train_config, dataset, seed
                )
                all_results[init_name].append(metrics)

                # Save individual result
                result_path = exp_dir / f"{init_name}_seed{seed}.json"
                with open(result_path, "w") as f:
                    json.dump(metrics, f, indent=2)

        # Save combined results
        summary_path = exp_dir / "results.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        # Mark complete
        if registry:
            summary = {
                init: {
                    "mean_final_val_loss": sum(
                        r.get("final_val_loss", 0) or 0 for r in results
                    ) / len(results),
                    "n_runs": len(results),
                }
                for init, results in all_results.items()
            }
            registry.mark_complete(exp_id, summary)

        return all_results

    except Exception as e:
        if registry:
            registry.mark_failed(exp_id, str(e))
        raise
