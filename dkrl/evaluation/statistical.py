"""Statistical testing utilities for experiment comparison.

Provides Welch's t-tests for pairwise method comparison and JSON serialization.
"""


import numpy as np
from scipy import stats


def run_all_pairwise_tests(
    detailed_results: list[dict],
    baseline: str = "NIGP-DKL",
    *,
    metrics: list[str],
) -> dict:
    """Run pairwise Welch t-tests comparing all models to baseline."""
    experiments = {}
    for result in detailed_results:
        exp_name = result["experiment"]
        if exp_name not in experiments:
            experiments[exp_name] = []
        experiments[exp_name].append(result)

    if baseline not in experiments:
        return {}

    baseline_results = experiments[baseline]

    stat_results = {}
    for metric in metrics:
        stat_results[metric] = {}
        baseline_vals = [r[metric] for r in baseline_results if not np.isnan(r[metric])]

        for exp_name, exp_results in experiments.items():
            if exp_name == baseline:
                continue

            exp_vals = [r[metric] for r in exp_results if not np.isnan(r[metric])]

            if len(baseline_vals) >= 2 and len(exp_vals) >= 2:
                t_stat, p_value = stats.ttest_ind(baseline_vals, exp_vals, equal_var=False)
                pooled_std = np.sqrt(
                    (np.std(baseline_vals, ddof=1) ** 2 + np.std(exp_vals, ddof=1) ** 2) / 2
                )
                effect_size = (
                    (np.mean(baseline_vals) - np.mean(exp_vals)) / pooled_std
                    if pooled_std > 0
                    else 0.0
                )

                stat_results[metric][exp_name] = {
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                    "effect_size": float(effect_size),
                    "baseline_mean": float(np.mean(baseline_vals)),
                    "exp_mean": float(np.mean(exp_vals)),
                    "significant": bool(p_value < 0.05),
                }

    return stat_results


def format_statistical_report(stat_results: dict) -> str:
    """Format statistical test results for printing."""
    lines = []
    for metric, comparisons in stat_results.items():
        for exp_name, result in comparisons.items():
            sig = "*" if result["significant"] else ""
            lines.append(
                f"stat metric={metric} exp={exp_name} p={result['p_value']:.4f}{sig} "
                f"d={result['effect_size']:.3f} baseline={result['baseline_mean']:.4f} "
                f"exp_mean={result['exp_mean']:.4f}"
            )
    return "\n".join(lines)


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer | np.int64 | np.int32):
        return int(obj)
    elif isinstance(obj, np.floating | np.float64 | np.float32):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj
