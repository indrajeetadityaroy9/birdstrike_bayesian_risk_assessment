#!/usr/bin/env bash
# Reproduce all DKRL paper results.
#
# Prerequisites:
#   - pip install -e .
#   - NVIDIA GPU with CUDA 12.x
#   - Data in data/processed/ (or set DKRL_DATA_DIR)
#
# Usage:
#   bash scripts/reproduce_results.sh

set -euo pipefail

echo "=== DKRL Results Reproduction ==="
echo "Date: $(date)"
echo ""

# 1. Run full ablation sweep (12 experiments x 10 seeds)
echo "--- Step 1: Ablation Sweep ---"
python scripts/sweep.py \
    --experiments experiments/ \
    --output_dir results/ \
    --models_dir checkpoints/
echo ""

# 2. Evaluate SOTA model
echo "--- Step 2: Evaluate SOTA Model ---"
python scripts/eval.py \
    --mode all \
    --model_path checkpoints/nigp_dkl_model.pt \
    --output results/evaluation.json
echo ""

echo "=== Reproduction Complete ==="
echo "Results saved to results/"
echo "Models saved to checkpoints/"
