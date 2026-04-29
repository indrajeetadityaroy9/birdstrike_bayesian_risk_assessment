"""Results formatting: LaTeX tables and comparison plots from experiment results."""

from pathlib import Path

import numpy as np


def format_metric(values: list[float], bold: bool = False, sig: bool = False) -> str:
    """Format mean+-std (LaTeX)."""
    mean = np.mean(values)
    std = np.std(values)
    s = f"{mean:.4f}$\\pm${std:.4f}"
    if bold:
        s = f"\\textbf{{{s}}}"
    if sig:
        s += "*"
    return s


def generate_latex_table(
    results: dict,
    metrics: list[str] = None,
) -> str:
    """Generate LaTeX comparison table."""
    if metrics is None:
        metrics = [
            "rmse", "mae", "nll", "crps", "scaled_crps",
            "coverage_95", "calibration_error_95",
            "ause", "interval_score_95",
            "tail_crps_upper", "tail_crps_lower",
        ]

    methods = list(results.keys())

    # Determine best method per metric (lower is better, except coverage)
    higher_is_better = {"coverage_95", "coverage_90", "coverage_68"}

    best_method = {}
    for metric in metrics:
        means = {}
        for method in methods:
            runs = results[method]
            values = [r[metric] for r in runs if metric in r]
            if values:
                means[method] = np.mean(values)
        if means:
            if metric in higher_is_better:
                nominal = float(metric.split("_")[-1]) / 100 if "coverage" in metric else 1.0
                best_method[metric] = min(means, key=lambda m: abs(means[m] - nominal))
            else:
                best_method[metric] = min(means, key=lambda m: means[m])

    # Build table
    n_metrics = len(metrics)
    header = " & ".join(["Method"] + [m.replace("_", "\\_") for m in metrics])

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\begin{{tabular}}{{l{'c' * n_metrics}}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]

    for method in methods:
        runs = results[method]
        row = [method.replace("_", "\\_")]
        for metric in metrics:
            values = [r[metric] for r in runs if metric in r]
            if values:
                is_best = best_method.get(metric) == method
                row.append(format_metric(values, bold=is_best))
            else:
                row.append("--")
        lines.append(" & ".join(row) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Comparison of methods across evaluation metrics. "
        "Bold indicates best performance. * denotes $p < 0.05$.}",
        "\\label{tab:results}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def generate_comparison_plot(results: dict, output_path: str = "results/comparison.pdf"):
    """Generate bar plots for CRPS, Coverage, AUSE, RMSE."""
    import matplotlib.pyplot as plt

    plot_metrics = ["crps", "coverage_95", "ause", "rmse"]
    methods = list(results.keys())
    n_methods = len(methods)
    n_metrics = len(plot_metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))

    for ax, metric in zip(axes, plot_metrics, strict=False):
        means = []
        stds = []
        for method in methods:
            runs = results[method]
            values = [r[metric] for r in runs if metric in r]
            means.append(np.mean(values) if values else 0)
            stds.append(np.std(values) if values else 0)

        x = np.arange(n_methods)
        ax.bar(x, means, yerr=stds, color=colors, capsize=3, edgecolor="black", linewidth=0.5)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace("_", "\n") for m in methods], fontsize=8, rotation=45, ha="right"
        )

        if "coverage" in metric:
            nominal = float(metric.split("_")[-1]) / 100
            ax.axhline(
                y=nominal, color="red", linestyle="--", alpha=0.7, label=f"Nominal ({nominal})"
            )
            ax.legend(fontsize=7)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"status=plot_saved path={output_path}")
    plt.close()
