"""
Train baseline models (MC Dropout, Deep Ensemble, Standard GP).

These baselines use their own training loops and YAML schema distinct from
the NIGP-DKL sweep pipeline. Results are saved in the same JSON format
as sweep.py for direct comparison.

Usage:
    python scripts/train_baselines.py \
        --config experiments/baselines/deep_ensemble.yaml \
        --output_dir results/ --models_dir checkpoints/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dkrl import create_dataset, evaluate_nigp_model
from dkrl.config.hardware import DEVICE
from dkrl.config.loader import load_config
from dkrl.models.baselines import DeepEnsembleModel, MCDropoutModel, StandardGPModel
from dkrl.utils.seeding import set_all_seeds

MODEL_REGISTRY = {
    "mc_dropout": MCDropoutModel,
    "deep_ensemble": DeepEnsembleModel,
    "standard_gp": StandardGPModel,
}


def _build_model(cfg):
    """Instantiate a baseline model from YAML config."""
    model_type = cfg["model_type"]
    model_cfg = cfg.get("model", {})
    cls = MODEL_REGISTRY[model_type]

    if model_type == "mc_dropout":
        return cls(
            input_dim=model_cfg.get("input_dim", 19),
            hidden_dim=model_cfg.get("hidden_dim", 76),
            dropout_rate=model_cfg.get("dropout_rate", 0.1),
            n_forward=model_cfg.get("n_forward", 30),
        )
    elif model_type == "deep_ensemble":
        return cls(
            input_dim=model_cfg.get("input_dim", 19),
            hidden_dim=model_cfg.get("hidden_dim", 76),
            n_members=model_cfg.get("n_members", 5),
        )
    elif model_type == "standard_gp":
        return cls(
            input_dim=model_cfg.get("input_dim", 19),
            max_train=model_cfg.get("max_train", 2000),
        )


def _train_mc_dropout(model, X, y, train_cfg):
    """Train MC Dropout with Adam."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 0.001))
    epochs = train_cfg.get("epochs", 500)
    batch_size = train_cfg.get("batch_size", 256)
    patience = train_cfg.get("early_stopping", {}).get("patience", 20)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X), device=X.device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            loss = model.train_step(X[idx], y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"status=early_stop epoch={epoch + 1}")
                break

    return {"epochs_trained": epoch + 1, "best_loss": best_loss}


def _train_deep_ensemble(model, X, y, train_cfg):
    """Train Deep Ensemble members independently."""
    model.to(DEVICE)
    epochs = train_cfg.get("epochs", 500)
    batch_size = train_cfg.get("batch_size", 256)
    lr = train_cfg.get("lr", 0.001)
    patience = train_cfg.get("early_stopping", {}).get("patience", 20)

    for k in range(model.n_members):
        optimizer = torch.optim.Adam(model.members[k].parameters(), lr=lr)
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(len(X), device=X.device)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(X), batch_size):
                idx = perm[i:i + batch_size]
                loss = model.train_step(X[idx], y[idx], member_idx=k)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        print(f"status=member_trained member={k} epochs={epoch + 1}")

    return {"epochs_trained": epochs, "n_members": model.n_members}


def _train_standard_gp(model, X, y, train_cfg):
    """Train Standard GP via MLL optimization."""
    model.to(DEVICE)
    n_iter = train_cfg.get("n_iter", 100)
    model.fit(X, y, n_iter=n_iter)
    return {"n_iter": n_iter}


TRAINERS = {
    "mc_dropout": _train_mc_dropout,
    "deep_ensemble": _train_deep_ensemble,
    "standard_gp": _train_standard_gp,
}


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--config", type=str, required=True, help="Baseline YAML config")
    parser.add_argument("--output_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--models_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--num_seeds", type=int, default=None, help="Override num_seeds from config")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    models_dir = Path(args.models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    model_type = cfg["model_type"]
    exp_name = cfg.get("name", Path(args.config).stem)
    train_cfg = cfg.get("training", {})
    num_seeds = args.num_seeds or cfg.get("evaluation", {}).get("num_seeds", 10)

    bundle, audit = create_dataset(seed=42)
    print(f"status=dataset n={audit.n_samples} model_type={model_type}")

    detailed_results = []

    for seed in range(num_seeds):
        set_all_seeds(seed)
        seed_bundle = bundle.resplit(seed)

        X_train, y_train, P_train, _ = seed_bundle.get_split("train")
        X = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
        y = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)

        model = _build_model(cfg)
        model.set_normalization(X.mean(dim=0), X.std(dim=0))
        model.set_target_normalization(y.mean(), y.std())

        trainer_fn = TRAINERS[model_type]
        train_info = trainer_fn(model, X, y, train_cfg)

        metrics = evaluate_nigp_model(
            model, seed_bundle, split="test", predict_fn=model.predict,
        )
        metrics["seed"] = seed
        metrics["experiment"] = exp_name
        metrics.update(train_info)
        detailed_results.append(metrics)

        print(
            f"status=seed_complete seed={seed} rmse={metrics['rmse']:.4f} "
            f"nll={metrics['nll']:.4f}"
        )

        if seed == 0:
            model_path = models_dir / f"{exp_name.lower().replace('-', '_')}_model.pt"
            torch.save(model.state_dict(), str(model_path))

    # Aggregate
    metric_keys = [
        "rmse", "mae", "nll", "coverage_95", "coverage_90", "coverage_68",
        "crps", "dss", "ause",
    ]
    agg = {"experiment": exp_name, "model_type": model_type}
    for key in metric_keys:
        vals = [m[key] for m in detailed_results]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))

    results_path = output_dir / f"{exp_name.lower().replace('-', '_')}_results.json"
    with open(results_path, "w") as f:
        json.dump({"aggregate": agg, "detailed": detailed_results}, f, indent=2, default=float)

    print(f"status=complete exp={exp_name} rmse={agg['rmse_mean']:.4f} output={results_path}")


if __name__ == "__main__":
    main()
