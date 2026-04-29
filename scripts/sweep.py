"""
Ablation Sweep - Runs experiments from YAML configs.

Iterates over experiment YAML files, trains models, evaluates, and produces
publication-grade JSON results with statistical analysis.

Usage:
    python scripts/sweep.py --experiments experiments/ --output_dir results --models_dir checkpoints
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dkrl import (
    create_dataset,
    evaluate_nigp_model,
    save_model,
    train_nigp_model,
)
from dkrl.config.loader import config_to_model_config, load_config
from dkrl.evaluation.statistical import (
    convert_to_serializable,
    format_statistical_report,
    run_all_pairwise_tests,
)
from dkrl.inference.sigma_point import SigmaPointNIGP


def main():
    parser = argparse.ArgumentParser(
        description="NIGP-DKL Ablation Sweep (reads experiment YAMLs)"
    )
    parser.add_argument(
        "--experiments", type=str, required=True,
        help="Directory containing experiment YAML files (searched recursively)",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for JSON results")
    parser.add_argument("--models_dir", type=str, required=True, help="Directory to save models")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    models_dir = Path(args.models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    exp_dir = Path(args.experiments)
    yaml_files = sorted(exp_dir.rglob("*.yaml"))
    experiments = []
    for yf in yaml_files:
        cfg = load_config(yf)
        experiments.append((yf, cfg))

    print(f"status=found_experiments n={len(experiments)} dir={exp_dir}")

    bundle, audit = create_dataset(seed=42)
    print(f"status=dataset_created n_total={audit.n_samples} n_train={audit.n_train} n_val={audit.n_val} n_test={audit.n_test}")

    all_results = []
    detailed_results = []

    for yaml_path, cfg in experiments:
        exp_name = cfg.get("name", yaml_path.stem)
        model_config = config_to_model_config(cfg)
        eval_cfg = cfg.get("evaluation", {})
        use_sigma_point = cfg.get("sigma_point", False)
        num_seeds = eval_cfg.get("num_seeds", 10)

        exp_metrics = []

        for seed in range(num_seeds):
            seed_bundle = bundle.resplit(seed)

            model, history = train_nigp_model(
                seed_bundle,
                model_config=model_config,
            )

            predict_fn = None
            if use_sigma_point:
                predict_fn = SigmaPointNIGP(model).predict

            metrics = evaluate_nigp_model(
                model, seed_bundle, split="test", predict_fn=predict_fn,
            )

            metrics["final_train_loss"] = history["train_loss"][-1]
            metrics["best_val_rmse"] = min(history["val_rmse"])
            metrics["epochs_trained"] = len(history["train_loss"])
            metrics["seed"] = seed
            metrics["experiment"] = exp_name

            exp_metrics.append(metrics)
            detailed_results.append(metrics)

            nll_str = f"{metrics['nll']:.4f}" if not np.isnan(metrics["nll"]) else "N/A"
            cov_str = f"{metrics['coverage_95']:.3f}" if not np.isnan(metrics["coverage_95"]) else "N/A"
            print(f"status=train_complete exp={exp_name} seed={seed} rmse={metrics['rmse']:.4f} nll={nll_str} cov95={cov_str}")

            if seed == 0:
                model_path = models_dir / f"{exp_name.lower().replace('-', '_')}_model.pt"
                save_model(model, str(model_path))

        agg = {"experiment": exp_name}
        metric_keys = [
            "rmse", "mae", "nll", "coverage_95", "coverage_90", "coverage_68",
            "mean_std", "crps", "scaled_crps", "dss", "interval_score_90",
            "interval_score_95", "ause",
        ]
        for key in metric_keys:
            vals = [m[key] for m in exp_metrics]
            agg[f"{key}_mean"] = np.mean(vals)
            agg[f"{key}_std"] = np.std(vals)
        for key in ["calibration_error_95", "calibration_error_90", "calibration_error_68"]:
            agg[f"{key}_mean"] = np.mean([m[key] for m in exp_metrics])
        all_results.append(agg)

    stat_results = run_all_pairwise_tests(
        detailed_results,
        baseline="NIGP-DKL",
        metrics=["rmse", "nll", "coverage_95", "crps"],
    )
    if stat_results:
        print(format_statistical_report(stat_results))

    with open(output_dir / "ablation_final.json", "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)

    with open(output_dir / "ablation_detailed.json", "w") as f:
        json.dump(convert_to_serializable(detailed_results), f, indent=2)

    if stat_results:
        with open(output_dir / "statistical_tests.json", "w") as f:
            json.dump(convert_to_serializable(stat_results), f, indent=2)

    for agg in all_results:
        print(
            f"status=experiment_result exp={agg['experiment']} rmse={agg['rmse_mean']:.4f}+/-{agg['rmse_std']:.4f} "
            f"nll={agg['nll_mean']:.4f} cov95={agg['coverage_95_mean']:.3f}"
        )

    print(f"status=complete n_experiments={len(experiments)} output={output_dir}")


if __name__ == "__main__":
    main()
