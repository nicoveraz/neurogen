"""CLI for running a single experiment from a YAML file."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurogen.config import get_device
from research.engine import run_experiment_from_yaml
from research.registry import ExperimentRegistry
from research.report import generate_phase_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a NeuroGen experiment")
    parser.add_argument("yaml", type=str, help="Path to experiment YAML file")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--report", action="store_true", help="Generate report")
    args = parser.parse_args()

    device = args.device or get_device()
    print(f"Using device: {device}")

    registry = ExperimentRegistry()
    results = run_experiment_from_yaml(
        args.yaml,
        output_dir=args.output_dir,
        device=device,
        registry=registry,
    )

    if args.report:
        import yaml
        with open(args.yaml) as f:
            config = yaml.safe_load(f)
        report_path = Path(args.output_dir) / Path(args.yaml).stem / "report.md"
        generate_phase_report(results, config["name"], report_path, config)
        print(f"Report saved to {report_path}")

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
