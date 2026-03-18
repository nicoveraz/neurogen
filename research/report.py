"""Auto-generate markdown reports from experiment results."""

import json
from pathlib import Path

import numpy as np


def generate_phase_report(
    results: dict,
    phase_name: str,
    output_path: str | Path,
    config: dict | None = None,
) -> Path:
    """Generate a markdown report from experiment results.

    Args:
        results: Dict mapping init_name -> list of metrics dicts.
        phase_name: Name of the phase for the report title.
        output_path: Path to write the markdown report.
        config: Optional experiment configuration to include.

    Returns:
        Path to the generated report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# {phase_name}\n")
    lines.append(f"*Auto-generated report*\n")

    # Configuration section
    if config:
        lines.append("## Configuration\n")
        lines.append(f"- **Initializers tested:** {', '.join(config.get('inits', []))}")
        training = config.get("training", {})
        lines.append(f"- **Max steps:** {training.get('max_steps', 'N/A')}")
        lines.append(f"- **Batch size:** {training.get('batch_size', 'N/A')}")
        lines.append(f"- **Seeds:** {config.get('seeds', 'N/A')}")
        lines.append("")

    # Results table
    lines.append("## Results Summary\n")
    lines.append("| Init Method | Final Val Loss | Best Val Loss | Train Time (s) |")
    lines.append("|-------------|---------------|---------------|----------------|")

    summary_data = {}
    for init_name, runs in results.items():
        final_losses = [
            r.get("final_val_loss", None) for r in runs
            if r.get("final_val_loss") is not None
        ]
        best_losses = [
            r.get("best_val_loss", None) for r in runs
            if r.get("best_val_loss") is not None
        ]
        train_times = [
            r.get("total_train_time_s", 0) for r in runs
        ]

        if final_losses:
            mean_final = np.mean(final_losses)
            std_final = np.std(final_losses) if len(final_losses) > 1 else 0
            mean_best = np.mean(best_losses) if best_losses else mean_final
            mean_time = np.mean(train_times)

            lines.append(
                f"| {init_name} | {mean_final:.4f} +/- {std_final:.4f} | "
                f"{mean_best:.4f} | {mean_time:.1f} |"
            )
            summary_data[init_name] = {
                "mean_final_val_loss": float(mean_final),
                "std_final_val_loss": float(std_final),
                "mean_best_val_loss": float(mean_best),
                "mean_train_time_s": float(mean_time),
            }

    lines.append("")

    # Best performer
    if summary_data:
        best_init = min(summary_data, key=lambda k: summary_data[k]["mean_best_val_loss"])
        lines.append(f"**Best performer:** {best_init} "
                     f"(best val_loss: {summary_data[best_init]['mean_best_val_loss']:.4f})\n")

    # Per-initializer details
    lines.append("## Detailed Results\n")
    for init_name, runs in results.items():
        lines.append(f"### {init_name}\n")
        for i, run in enumerate(runs):
            seed = run.get("seed", i)
            lines.append(f"**Seed {seed}:**")
            if run.get("final_val_loss") is not None:
                lines.append(f"- Final val_loss: {run['final_val_loss']:.4f}")
            if run.get("best_val_loss") is not None:
                lines.append(f"- Best val_loss: {run['best_val_loss']:.4f}")
            if run.get("total_train_time_s") is not None:
                lines.append(f"- Train time: {run['total_train_time_s']:.1f}s")
            lines.append("")

    # Write report
    report_text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report_text)

    return output_path
