"""CLI script to run a single experiment from a YAML config file.

Usage:
    python scripts/run_experiment.py --config research/experiments/phase1_baselines.yaml
    python scripts/run_experiment.py --config path/to/experiment.yaml --output-dir outputs/exp1
    python scripts/run_experiment.py --config experiment.yaml --device cpu
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurogen.config import get_device
from research.engine import load_raw_yaml, run_experiment_from_yaml
from research.report import generate_phase_report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run a NeuroGen experiment from a YAML config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/experiments",
        help="Base output directory for results (default: outputs/experiments).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cuda/mps/cpu). Auto-detects if not set.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation after experiment.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the experiment pipeline."""
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    device = args.device or get_device()
    print(f"Using device: {device}")
    print(f"Config: {config_path}")

    # Derive output directory from config name
    config_name = config_path.stem
    output_dir = Path(args.output_dir) / config_name

    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Load raw YAML for report
    raw_config = load_raw_yaml(str(config_path))

    # Run all inits and seeds
    results = run_experiment_from_yaml(
        yaml_path=str(config_path),
        output_dir=str(output_dir),
        device=device,
    )

    print("=" * 60)
    print("Experiment complete.")

    # Generate report
    if not args.no_report:
        report_path = output_dir / "report.md"
        generate_phase_report(
            phase_results=results,
            output_path=str(report_path),
            config=raw_config,
        )
        print(f"Report: {report_path}")

    # Print summary
    print("\nSummary:")
    for init_name, init_data in results.items():
        if not isinstance(init_data, dict):
            continue
        mean_vl = init_data.get("final_val_loss_mean", "N/A")
        std_vl = init_data.get("final_val_loss_std", "N/A")
        if isinstance(mean_vl, float):
            print(f"  {init_name}: val_loss = {mean_vl:.4f} +/- {std_vl:.4f}")
        else:
            print(f"  {init_name}: val_loss = {mean_vl}")


if __name__ == "__main__":
    main()
