"""
Model Evaluation Script

Evaluates trained NIGP-DKL models with:
- Calibration analysis (NIGP uncertainty metrics)
- Conformal prediction validity
- Data audit

Usage:
    python scripts/eval.py --mode all --model_path checkpoints/model.pt --output results/eval.json
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from dkrl import (
    KernelConformalizedNIGP,
    LeverageSplitCP,
    create_dataset,
    evaluate_nigp_model,
    load_model,
)
from dkrl.config.hardware import DEVICE


def main():
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument("--mode", choices=["calibration", "conformal", "audit", "all"], required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path to save JSON results")
    args = parser.parse_args()

    print(f"status=startup mode={args.mode} model={args.model_path}")
    model = load_model(args.model_path)

    bundle, audit = create_dataset(seed=42)
    print(f"dataset n={audit.n_samples} split=test")

    results = {"model_path": args.model_path, "mode": args.mode, "split": "test"}

    if args.mode in ["calibration", "all"]:
        metrics = evaluate_nigp_model(model, bundle, split="test")
        results["calibration"] = metrics
        print(f"calibration rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} nll={metrics['nll']:.4f} "
              f"cov68={metrics['coverage_68']:.4f} cov90={metrics['coverage_90']:.4f} cov95={metrics['coverage_95']:.4f} "
              f"crps={metrics['crps']:.4f} ause={metrics['ause']:.4f}")

    if args.mode in ["conformal", "all"]:
        features, labels, covariances, _ = bundle.get_split("test", normalize_quality=False)
        X_t = torch.as_tensor(features, dtype=torch.float32, device=DEVICE)
        y_t = torch.as_tensor(labels, dtype=torch.float32, device=DEVICE)
        P_t = torch.as_tensor(covariances, dtype=torch.float32, device=DEVICE)

        n = len(X_t)
        cal_size = int(0.3 * n)
        perm_idx = np.random.RandomState(42).permutation(n)
        cal_idx, eval_idx = perm_idx[:cal_size], perm_idx[cal_size:]
        X_cal_t, y_cal_t, P_cal_t = X_t[cal_idx], y_t[cal_idx], P_t[cal_idx]
        X_eval_t, y_eval_t, P_eval_t = X_t[eval_idx], y_t[eval_idx], P_t[eval_idx]

        print(f"conformal method=lcmqr n_cal={len(X_cal_t)} n_eval={len(X_eval_t)}")
        lcmqr = KernelConformalizedNIGP(model)  # Auto-bandwidth via Silverman's rule
        lcmqr.calibrate(X_cal_t, y_cal_t, P_cal_t)

        conformal_results = {"method": "lcmqr"}
        for confidence in [0.90, 0.95]:
            metrics = lcmqr.evaluate(X_eval_t, y_eval_t, P_eval_t, confidence=confidence)
            conformal_results[f"{int(confidence*100)}"] = asdict(metrics)
            print(f"conformal method=lcmqr confidence={confidence:.2f} coverage={metrics.coverage:.4f} width={metrics.efficiency:.4f}")

        n_test = len(X_t)
        perm = torch.randperm(n_test)
        cp_cal_idx, cp_eval_idx = perm[: n_test // 2], perm[n_test // 2 :]
        print(f"conformal method=leverage-split-cp n_cal={len(cp_cal_idx)} n_eval={len(cp_eval_idx)}")
        gn_cp = LeverageSplitCP(model)
        gn_cp.calibrate(X_t[cp_cal_idx], y_t[cp_cal_idx], P_t[cp_cal_idx])

        gn_results = {"method": "leverage-split-cp"}
        for confidence in [0.90, 0.95]:
            metrics = gn_cp.evaluate(X_t[cp_eval_idx], y_t[cp_eval_idx], P_t[cp_eval_idx], confidence=confidence)
            gn_results[f"{int(confidence*100)}"] = asdict(metrics)
            print(f"conformal method=leverage-split-cp confidence={confidence:.2f} coverage={metrics.coverage:.4f} width={metrics.efficiency:.4f}")

        results["conformal"] = {"lcmqr": conformal_results, "gn_full_cp": gn_results}

    if args.mode in ["audit", "all"]:
        features, labels, covariances, _ = bundle.get_split("test", normalize_quality=False)
        nan_features = np.isnan(features).sum()
        nan_labels = np.isnan(labels).sum()
        nan_covs = np.isnan(covariances).sum()

        print(f"audit nan_features={nan_features} nan_labels={nan_labels} nan_covs={nan_covs} "
              f"labels_range=[{labels.min():.4f},{labels.max():.4f}] features_mean={features.mean():.4f} features_std={features.std():.4f}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"audit params_total={total_params} params_trainable={trainable_params}")
        results["audit"] = {
            "nan_features": int(nan_features),
            "nan_labels": int(nan_labels),
            "nan_covs": int(nan_covs),
            "labels_min": float(labels.min()),
            "labels_max": float(labels.max()),
            "params_total": total_params,
            "params_trainable": trainable_params,
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"status=complete mode={args.mode} output={output_path}")


if __name__ == "__main__":
    main()
