"""CLI script to run all experiments for a specific research phase.

Usage:
    python scripts/run_phase.py --phase 1
    python scripts/run_phase.py --phase 2 --output-dir outputs/phase2 --device mps
    python scripts/run_phase.py --phase 3 --device cpu
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurogen.config import get_device
from research.engine import load_raw_yaml, run_experiment_from_yaml
from research.registry import ExperimentRegistry
from research.report import generate_phase_report

# Mapping of phase numbers to YAML config files
PHASE_CONFIGS: dict[int, str] = {
    1: "research/experiments/phase1_baselines.yaml",
    2: "research/experiments/phase2_ca_validation.yaml",
    3: "research/experiments/phase3_random_ca.yaml",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run all experiments for a NeuroGen research phase."
    )
    parser.add_argument(
        "--phase",
        type=int,
        required=True,
        choices=sorted(PHASE_CONFIGS.keys()),
        help="Phase number to run (1, 2, or 3).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/phases",
        help="Base output directory (default: outputs/phases).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cuda/mps/cpu). Auto-detects if not set.",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="outputs/registry.json",
        help="Path to experiment registry file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run all experiments for the specified phase."""
    args = parse_args()
    device = args.device or get_device()

    phase = args.phase
    config_path = Path(PHASE_CONFIGS[phase])

    if not config_path.exists():
        print(f"Error: Phase {phase} config not found: {config_path}")
        sys.exit(1)

    print(f"NeuroGen Phase {phase}")
    print(f"Config: {config_path}")
    print(f"Device: {device}")
    print("=" * 60)

    # Set up output directory
    output_dir = Path(args.output_dir) / f"phase{phase}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up registry
    registry = ExperimentRegistry(args.registry)
    experiment_id = f"phase{phase}"

    # Load raw config
    raw_config = load_raw_yaml(str(config_path))

    # Register experiment
    if experiment_id not in registry:
        registry.register(
            experiment_id=experiment_id,
            config=raw_config,
            status="pending",
        )

    # Update status to running
    registry.update_status(experiment_id, "running")

    try:
        # Run all experiments in this phase
        results = run_experiment_from_yaml(
            yaml_path=str(config_path),
            output_dir=str(output_dir),
            device=device,
        )

        # Generate phase report
        report_path = output_dir / "report.md"
        generate_phase_report(
            phase_results=results,
            output_path=str(report_path),
            config=raw_config,
        )

        # Update registry with results
        serializable_results = {
            k: {
                mk: mv
                for mk, mv in v.items()
                if mk != "per_seed"
            }
            if isinstance(v, dict)
            else v
            for k, v in results.items()
        }
        registry.update_status(
            experiment_id, "complete", results=serializable_results
        )

        print("=" * 60)
        print(f"Phase {phase} complete.")
        print(f"Report: {report_path}")
        print(f"Results: {output_dir / 'aggregated_results.json'}")

        # Print summary table
        print("\nResults Summary:")
        print("-" * 60)
        print(f"{'Init Method':<25} {'Val Loss (mean)':<20} {'Val Loss (std)':<15}")
        print("-" * 60)
        for init_name, init_data in sorted(results.items()):
            if not isinstance(init_data, dict):
                continue
            mean_vl = init_data.get("final_val_loss_mean", float("nan"))
            std_vl = init_data.get("final_val_loss_std", float("nan"))
            print(f"{init_name:<25} {mean_vl:<20.4f} {std_vl:<15.4f}")
        print("-" * 60)

    except Exception as e:
        registry.update_status(experiment_id, "failed")
        print(f"Phase {phase} failed: {e}")
        raise


if __name__ == "__main__":
    main()
