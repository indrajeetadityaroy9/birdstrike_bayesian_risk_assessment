"""
Results Generator: LaTeX tables and comparison plots from JSON.

Usage:
    python scripts/results_table.py --results results/ablation_final.json
    python scripts/results_table.py --results results/ablation_final.json --plot --output results/comparison.pdf
"""

import argparse
import json

from dkrl.evaluation.formatting import generate_comparison_plot, generate_latex_table


def main():
    parser = argparse.ArgumentParser(description="Generate results tables and plots")
    parser.add_argument("--results", required=True, help="Path to results JSON")
    parser.add_argument("--plot", action="store_true", help="Also generate comparison plot")
    parser.add_argument("--output", default="results/comparison.pdf", help="Plot output path")
    parser.add_argument("--metrics", nargs="+", default=None, help="Metrics to include")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    latex = generate_latex_table(results, metrics=args.metrics)
    print(latex)

    if args.plot:
        generate_comparison_plot(results, args.output)


if __name__ == "__main__":
    main()
