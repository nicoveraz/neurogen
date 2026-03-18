"""CLI for auto-research.

Usage:
    python scripts/run_auto_research.py                          # default agenda
    python scripts/run_auto_research.py --agenda my_agenda.yaml  # custom
    python scripts/run_auto_research.py --status                 # current status
    python scripts/run_auto_research.py --max-cycles 10          # limit cycles
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurogen.config import get_device


def main() -> None:
    """Entry point for the auto-research CLI."""
    parser = argparse.ArgumentParser(description="NeuroGen Auto-Research")
    parser.add_argument(
        "--agenda",
        default="research/agenda.yaml",
        help="Path to research agenda YAML",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/auto_research",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (cuda, mps, cpu)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current status and exit",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Maximum number of cycles",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Experiments per cycle",
    )
    args = parser.parse_args()

    from research.auto_research import AutoResearch

    device = args.device or get_device()
    auto = AutoResearch(
        agenda_path=args.agenda,
        output_dir=args.output_dir,
        device=device,
    )

    if args.status:
        status = auto.status()
        for qid, info in status.items():
            print(
                f"{qid}: {info['status']} "
                f"({info['experiments']} experiments, "
                f"{info['hours_used']}h)"
            )
        return

    auto.run(max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()
