"""CLI for running all experiments in a phase."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurogen.config import get_device
from research.engine import load_experiment_config, run_experiment_from_yaml
from research.registry import ExperimentRegistry
from research.report import generate_phase_report

PHASE_FILES = {
    "1": "research/experiments/phase1_baselines.yaml",
    "2": "research/experiments/phase2_ca_validation.yaml",
    "3": "research/experiments/phase3_random_ca.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a NeuroGen research phase")
    parser.add_argument("--phase", type=str, required=True, help="Phase number (1-8)")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    if args.phase not in PHASE_FILES:
        print(f"Phase {args.phase} not yet defined. Available: {list(PHASE_FILES.keys())}")
        return

    yaml_path = PHASE_FILES[args.phase]
    device = args.device or get_device()
    print(f"Running Phase {args.phase} on device: {device}")

    import yaml
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    registry = ExperimentRegistry()
    results = run_experiment_from_yaml(
        yaml_path,
        output_dir=args.output_dir,
        device=device,
        registry=registry,
    )

    # Generate report
    report_path = Path(args.output_dir) / Path(yaml_path).stem / "report.md"
    generate_phase_report(results, config["name"], report_path, config)
    print(f"\nPhase {args.phase} complete. Report: {report_path}")


if __name__ == "__main__":
    main()
