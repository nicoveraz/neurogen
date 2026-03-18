"""Report generation for research experiments.

Generates markdown reports with comparison tables, figure references,
and analysis sections from experiment results.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def _fmt(value: float | None, decimals: int = 4) -> str:
    """Format a float for display, handling None and special values."""
    if value is None or isinstance(value, str):
        return str(value) if value is not None else "N/A"
    if value != value:
        return "NaN"
    if value == float("inf") or value == float("-inf"):
        return str(value)
    return f"{value:.{decimals}f}"


def _make_comparison_table(results: dict[str, Any]) -> str:
    """Create a markdown comparison table from aggregated results."""
    metrics = [
        ("final_val_loss_mean", "Val Loss (mean)"),
        ("final_val_loss_std", "Val Loss (std)"),
        ("best_val_loss_mean", "Best Val (mean)"),
        ("best_val_loss_std", "Best Val (std)"),
        ("final_train_loss_mean", "Train Loss"),
        ("total_time_mean_s", "Time (s)"),
    ]
    cols = ["Init Method"] + [label for _, label in metrics]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [header, sep]

    for init_name, data in sorted(results.items()):
        if not isinstance(data, dict):
            continue
        vals = [init_name]
        for key, _ in metrics:
            v = data.get(key)
            d = 1 if key.endswith("_s") else 4
            vals.append(_fmt(v, d))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def _config_section(config: dict[str, Any]) -> str:
    """Create a markdown section summarizing experiment configuration."""
    lines = ["## Configuration", ""]
    if "name" in config:
        lines.append(f"**Experiment:** {config['name']}")
    if "hypothesis" in config:
        lines.append(f"**Hypothesis:** {config['hypothesis']}")
    lines.append("")

    model = config.get("model", {})
    if model:
        lines.extend([
            "### Model", "",
            f"- Layers: {model.get('n_layer', 'N/A')}, "
            f"Heads: {model.get('n_head', 'N/A')}, "
            f"Embd: {model.get('n_embd', 'N/A')}, "
            f"Block: {model.get('block_size', 'N/A')}", "",
        ])

    tr = config.get("training", {})
    if tr:
        lines.extend([
            "### Training", "",
            f"- Steps: {tr.get('max_steps', 'N/A')}, "
            f"LR: {tr.get('lr', 'N/A')}, "
            f"Batch: {tr.get('batch_size', 'N/A')}", "",
        ])

    inits = config.get("inits", [])
    if inits:
        lines.append("### Initialization Methods\n")
        for name in inits:
            lines.append(f"- `{name}`")
        lines.append("")

    seeds = config.get("seeds", [])
    if seeds:
        lines.append(f"**Seeds:** {seeds}\n")
    return "\n".join(lines)


def _results_section(results: dict[str, Any]) -> str:
    """Create results section with comparison table."""
    return "\n".join([
        "## Results", "", "### Summary Table", "",
        _make_comparison_table(results), "",
    ])


def _analysis_section(results: dict[str, Any]) -> str:
    """Create analysis section identifying best init and relative performance."""
    lines = ["## Analysis", ""]
    best_init, best_loss = None, float("inf")
    for name, data in results.items():
        if not isinstance(data, dict):
            continue
        loss = data.get("best_val_loss_mean", float("inf"))
        if isinstance(loss, (int, float)) and loss < best_loss:
            best_loss, best_init = loss, name

    if best_init:
        lines.append(
            f"**Best:** `{best_init}` (best val loss: {_fmt(best_loss)})\n"
        )
        lines.append("### Relative Performance\n")
        for name, data in sorted(results.items()):
            if not isinstance(data, dict):
                continue
            loss = data.get("best_val_loss_mean", float("inf"))
            if isinstance(loss, (int, float)) and best_loss > 0:
                rel = ((loss - best_loss) / best_loss) * 100
                lines.append(f"- `{name}`: {_fmt(loss)} ({_fmt(rel, 1)}% vs best)")
        lines.append("")
    return "\n".join(lines)


def _figures_section(output_dir: str) -> str:
    """Create section referencing any generated PNG figures."""
    lines = ["## Figures", ""]
    fig_dir = Path(output_dir) / "figures"
    if fig_dir.exists():
        pngs = sorted(fig_dir.glob("*.png"))
        for png in pngs:
            cap = png.stem.replace("_", " ").title()
            lines.extend([f"### {cap}", "", f"![{cap}](figures/{png.name})", ""])
        if not pngs:
            lines.append("No figures generated.\n")
    else:
        lines.append("No figures directory found.\n")
    return "\n".join(lines)


def _conclusions_section(results: dict[str, Any]) -> str:
    """Create conclusions section."""
    n = sum(1 for v in results.values() if isinstance(v, dict))
    ca = {"grid_ca", "neural_ca", "spectral_ca", "topo_ca", "reaction_diffusion"}
    has_ca = any(k in ca for k in results)
    has_bl = any(k not in ca for k in results if isinstance(results.get(k), dict))

    lines = ["## Conclusions", "", f"Compared {n} initialization methods."]
    if has_ca and has_bl:
        lines.append("Both CA and baseline methods were evaluated.")
    lines.append("")
    return "\n".join(lines)


def generate_phase_report(
    phase_results: dict[str, Any],
    output_path: str,
    config: dict[str, Any] | None = None,
) -> None:
    """Generate a complete markdown report for a research phase.

    Args:
        phase_results: Aggregated results from run_experiment_from_yaml().
        output_path: Path to write the markdown report file.
        config: Optional raw YAML config dict for the config section.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    title = config.get("name", "Experiment Report") if config else "Experiment Report"

    parts = [
        f"# {title}\n",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
        "---\n",
    ]
    if config:
        parts.extend([_config_section(config), "---\n"])
    parts.extend([
        _results_section(phase_results), "---\n",
        _analysis_section(phase_results), "---\n",
        _figures_section(str(output_file.parent)), "---\n",
        _conclusions_section(phase_results),
    ])

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"Report written to {output_file}")
