"""
Training entry point for NIGP-DKL.

Loads configuration from YAML and runs the training pipeline.

Usage:
    python scripts/train.py --config configs/default.yaml --output checkpoints/model.pt
"""

import argparse
from pathlib import Path

from dkrl import create_dataset, save_model, train_nigp_model
from dkrl.config.loader import config_to_model_config, load_config


def main():
    parser = argparse.ArgumentParser(description="Train NIGP-DKL model")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML (default: configs/default.yaml)",
    )
    parser.add_argument("--output", type=str, required=True, help="Path to save trained model")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    model_config = config_to_model_config(cfg)

    seed = cfg.get("dataset", {}).get("seed", 42)

    print(f"status=startup config={args.config} output={args.output}")

    bundle, audit = create_dataset(seed=seed)
    print(f"status=dataset n_total={audit.n_samples} n_train={audit.n_train} n_val={audit.n_val} n_test={audit.n_test}")

    model, history = train_nigp_model(
        bundle,
        model_config=model_config,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(output_path))
    print(f"status=saved model={output_path}")


if __name__ == "__main__":
    main()
