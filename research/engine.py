"""Research experiment runner: loads YAML configs, runs training, collects metrics.

Handles dispatching to baseline initializers and CA variants, running
training with metric collection, and saving results as JSON.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from neurogen.config import ExperimentConfig, GPTConfig, TrainConfig, get_device
from neurogen.data.shakespeare import ShakespeareDataset
from neurogen.model.gpt import GPT
from neurogen.training.trainer import train


def load_experiment_config(yaml_path: str) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file.

    Args:
        yaml_path: Path to the YAML experiment definition file.

    Returns:
        ExperimentConfig populated from the YAML.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {yaml_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Empty YAML file: {yaml_path}")

    model_raw = raw.get("model", {})
    model_config = GPTConfig(
        block_size=model_raw.get("block_size", 256), vocab_size=0,
        n_layer=model_raw.get("n_layer", 6),
        n_head=model_raw.get("n_head", 6),
        n_embd=model_raw.get("n_embd", 384),
        dropout=model_raw.get("dropout", 0.2),
    )
    train_raw = raw.get("training", {})
    train_config = TrainConfig(
        max_steps=train_raw.get("max_steps", 5000),
        eval_interval=train_raw.get("eval_interval", 250),
        batch_size=train_raw.get("batch_size", 64),
        lr=train_raw.get("lr", 3e-4),
    )
    inits = raw.get("inits", ["xavier_normal"])
    first_init = inits[0] if isinstance(inits, list) else inits

    return ExperimentConfig(
        name=raw.get("name", "unnamed"),
        hypothesis=raw.get("hypothesis", ""),
        model_config=model_config, train_config=train_config,
        init_method=first_init,
        dataset=raw.get("dataset", "shakespeare_char"),
        metrics=raw.get("metrics", ["train_loss", "val_loss"]),
        seeds=raw.get("seeds", [42, 137, 256]),
    )


def load_raw_yaml(yaml_path: str) -> dict[str, Any]:
    """Load raw YAML config without converting to dataclass."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_init_fn(init_name: str, device: str) -> Any:
    """Get an initialization function by name (baseline or CA variant)."""
    from neurogen.baselines.initializers import INITIALIZERS
    from neurogen.ca.engine import CAWeightEngine

    if init_name in INITIALIZERS:
        fn = INITIALIZERS[init_name]
        return lambda model: fn(model)

    if init_name in set(CAWeightEngine.available_variants()):
        engine = CAWeightEngine(variant=init_name, device=device)
        return lambda model: engine.develop_weights(model)

    raise ValueError(f"Unknown init method '{init_name}'.")


def _collect_gradient_norm(model: GPT) -> float:
    """Compute total gradient L2 norm across all parameters."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def _make_serializable(obj: Any) -> Any:
    """Convert an object to JSON-serializable form."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj:
            return None
        if obj == float("inf") or obj == float("-inf"):
            return str(obj)
    return obj


def run_single_experiment(
    config: ExperimentConfig, seed: int, output_dir: str,
    device: str | None = None,
) -> dict[str, Any]:
    """Run one training run and collect metrics.

    Args:
        config: Experiment configuration.
        seed: Random seed for this run.
        output_dir: Directory to save results.
        device: Device override. Auto-detected if None.

    Returns:
        Dictionary of collected metrics.
    """
    device = device or get_device()
    torch.manual_seed(seed)

    dataset = ShakespeareDataset()
    model_config = GPTConfig(
        block_size=config.model_config.block_size,
        vocab_size=dataset.vocab_size,
        n_layer=config.model_config.n_layer,
        n_head=config.model_config.n_head,
        n_embd=config.model_config.n_embd,
        dropout=config.model_config.dropout,
    )
    model = GPT(model_config)
    init_fn = _get_init_fn(config.init_method, device)

    gradient_norms: list[float] = []

    def step_callback(step: int, tl: float, vl: float | None) -> None:
        gradient_norms.append(_collect_gradient_norm(model))

    start_time = time.time()
    train_metrics = train(
        model=model, dataset=dataset, config=config.train_config,
        init_fn=init_fn, device=device, callback=step_callback,
    )
    total_time = time.time() - start_time

    weight_stats: dict[str, Any] = {}
    if any(m in config.metrics for m in
           ("weight_spectral_norm", "weight_effective_rank")):
        from neurogen.analysis.weight_analysis import (
            analyze_weight_dict, aggregate_analysis,
        )
        per_layer = analyze_weight_dict(model.get_weight_tensors())
        weight_stats = {
            "per_layer": {k: dict(v) for k, v in per_layer.items()},
            "aggregate": aggregate_analysis(per_layer),
        }

    results: dict[str, Any] = {
        "experiment_name": config.name, "init_method": config.init_method,
        "seed": seed,
        "train_losses": train_metrics["train_losses"],
        "val_losses": train_metrics["val_losses"],
        "best_val_loss": train_metrics["best_val_loss"],
        "final_train_loss": train_metrics["final_train_loss"],
        "final_val_loss": train_metrics["final_val_loss"],
        "gradient_norms": gradient_norms, "weight_stats": weight_stats,
        "total_time_s": total_time,
        "steps_per_sec": train_metrics["steps_per_sec"], "device": device,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    safe_name = config.init_method.replace("/", "_")
    result_file = out_path / f"{safe_name}_seed{seed}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(_make_serializable(results), f, indent=2)
    print(f"Results saved to {result_file}")
    return results


def _aggregate_results(
    all_results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Aggregate per-seed results for each init method."""
    aggregated: dict[str, Any] = {}
    for init_name, seed_results in all_results.items():
        if not seed_results:
            continue
        fvl = [r["final_val_loss"] for r in seed_results]
        bvl = [r["best_val_loss"] for r in seed_results]
        ftl = [r["final_train_loss"] for r in seed_results]
        times = [r["total_time_s"] for r in seed_results]
        n = len(fvl)
        mean_fvl = sum(fvl) / n
        mean_bvl = sum(bvl) / n
        std_fvl = (sum((x - mean_fvl) ** 2 for x in fvl) / max(n - 1, 1)) ** 0.5
        std_bvl = (sum((x - mean_bvl) ** 2 for x in bvl) / max(n - 1, 1)) ** 0.5
        aggregated[init_name] = {
            "n_seeds": n,
            "final_val_loss_mean": mean_fvl, "final_val_loss_std": std_fvl,
            "best_val_loss_mean": mean_bvl, "best_val_loss_std": std_bvl,
            "final_train_loss_mean": sum(ftl) / n,
            "total_time_mean_s": sum(times) / n,
            "per_seed": seed_results,
        }
    return aggregated


def run_experiment_from_yaml(
    yaml_path: str, output_dir: str, device: str | None = None,
) -> dict[str, Any]:
    """Load a YAML experiment, run all seeds and inits, aggregate results.

    Args:
        yaml_path: Path to the experiment YAML file.
        output_dir: Base directory for results output.
        device: Device override.

    Returns:
        Dictionary with aggregated results across all inits and seeds.
    """
    raw = load_raw_yaml(yaml_path)
    base_config = load_experiment_config(yaml_path)
    inits = raw.get("inits", [base_config.init_method])
    seeds = raw.get("seeds", base_config.seeds)
    device = device or get_device()

    all_results: dict[str, list[dict[str, Any]]] = {}
    for init_name in inits:
        init_results: list[dict[str, Any]] = []
        for seed in seeds:
            config = ExperimentConfig(
                name=base_config.name, hypothesis=base_config.hypothesis,
                model_config=base_config.model_config,
                train_config=base_config.train_config,
                init_method=init_name, dataset=base_config.dataset,
                metrics=base_config.metrics, seeds=seeds,
            )
            init_dir = Path(output_dir) / init_name
            result = run_single_experiment(
                config=config, seed=seed,
                output_dir=str(init_dir), device=device,
            )
            init_results.append(result)
        all_results[init_name] = init_results

    aggregated = _aggregate_results(all_results)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    agg_file = out_path / "aggregated_results.json"
    with open(agg_file, "w", encoding="utf-8") as f:
        json.dump(_make_serializable(aggregated), f, indent=2)
    print(f"Aggregated results saved to {agg_file}")
    return aggregated


def run_experiment(exp_config: dict, device: str | None = None) -> dict[str, Any]:
    """Run a single experiment from a config dict (programmatic API).

    This is the interface used by the auto-research engine. Unlike
    run_single_experiment() which requires an ExperimentConfig,
    this accepts the dict format produced by experiment_generator.

    Args:
        exp_config: Dict with keys: init_method, seed, steps,
            model_config, train_config, ca_config, question_id.
        device: Device override.

    Returns:
        Dict with metrics: val_loss, train_loss, train_losses, val_losses,
        best_val_loss, total_time_s, steps_per_sec, etc.
    """
    device = device or get_device()
    seed = exp_config.get("seed", 42)
    torch.manual_seed(seed)

    dataset = ShakespeareDataset()

    model_config = exp_config.get("model_config")
    if isinstance(model_config, GPTConfig):
        mc = GPTConfig(
            block_size=model_config.block_size,
            vocab_size=dataset.vocab_size,
            n_layer=model_config.n_layer,
            n_head=model_config.n_head,
            n_embd=model_config.n_embd,
            dropout=model_config.dropout,
        )
    else:
        mc = GPTConfig(block_size=256, vocab_size=dataset.vocab_size,
                       n_layer=6, n_head=6, n_embd=384, dropout=0.2)

    model = GPT(mc)

    init_method = exp_config.get("init_method", "xavier_normal")
    ca_config = exp_config.get("ca_config", {})
    is_live = ca_config.get("live", False)

    # Get init function
    init_fn = _get_init_fn(init_method, device)

    train_config = exp_config.get("train_config")
    if not isinstance(train_config, TrainConfig):
        steps = exp_config.get("steps", 1000)
        train_config = TrainConfig(max_steps=steps, eval_interval=max(steps // 10, 1))

    start = time.time()

    if is_live:
        # Run with LiveCATrainer
        from neurogen.ca.live import LIVE_CA_RULES
        from neurogen.ca.live.alpha_schedule import AlphaSchedule
        from neurogen.training.live_ca_trainer import LiveCATrainer
        from neurogen.config import LiveCAConfig

        # Initialize weights first
        weights = init_fn(model)
        model.set_weight_tensors(weights)
        model = model.to(device)

        rule_name = ca_config.get("live_ca_rule", "local_norm")
        rule_cls = LIVE_CA_RULES.get(rule_name)
        if rule_cls is None:
            raise ValueError(f"Unknown live CA rule: {rule_name}")

        ca_rules = {"": rule_cls()}  # empty string matches all params
        schedule = AlphaSchedule(
            mode=ca_config.get("alpha_schedule", "exponential_decay"),
            alpha_0=ca_config.get("alpha_0", 0.01),
            total_steps=train_config.max_steps,
        )
        live_config = LiveCAConfig(
            ca_type=rule_name,
            total_steps=train_config.max_steps,
        )

        trainer = LiveCATrainer(model, ca_rules, schedule, live_config, device)
        trainer.configure_optimizer(lr=train_config.lr)
        metrics = trainer.train(dataset)

        duration = time.time() - start
        return {
            "val_loss": metrics.get("final_loss", float("inf")),
            "final_val_loss": metrics.get("final_loss", float("inf")),
            "final_train_loss": metrics.get("final_loss", float("inf")),
            "best_val_loss": min(metrics.get("train_losses", [float("inf")])),
            "train_losses": metrics.get("train_losses", []),
            "ca_magnitudes": metrics.get("ca_magnitudes", []),
            "ca_alignments": metrics.get("ca_alignments", []),
            "total_time_s": duration,
            "steps_per_sec": train_config.max_steps / duration if duration > 0 else 0,
        }
    else:
        # Standard training
        train_metrics = train(
            model=model, dataset=dataset, config=train_config,
            init_fn=init_fn, device=device,
        )
        duration = time.time() - start
        return {
            "val_loss": train_metrics.get("final_val_loss", float("inf")),
            "final_val_loss": train_metrics.get("final_val_loss", float("inf")),
            "final_train_loss": train_metrics.get("final_train_loss", float("inf")),
            "best_val_loss": train_metrics.get("best_val_loss", float("inf")),
            "train_losses": train_metrics.get("train_losses", []),
            "val_losses": train_metrics.get("val_losses", []),
            "total_time_s": duration,
            "steps_per_sec": train_metrics.get("steps_per_sec", 0),
        }
