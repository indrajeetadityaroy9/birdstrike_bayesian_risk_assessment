# Deep Kernel Risk Learning (DKRL)

**Second-order NIGP Deep Kernel Learning for risk prediction under input uncertainty**

This repository implements a single end-to-end system for uncertainty-aware regression when inputs are noisy and each sample carries a covariance estimate. The current codebase centers on a CUDA-only PyTorch implementation of:

- a bi-Lipschitz deep kernel feature extractor,
- a low-rank Gaussian Process (GP) head,
- first- and second-order Noisy-Input GP (NIGP) corrections,
- conformal prediction for distribution-free interval guarantees,
- adaptive (data-derived) training controls.

The target application in this repo is bird-strike risk prediction from GPS telemetry with uncertainty propagation from compressed input covariances.

## 1. Research Objective

The project objective is to build a **calibrated risk predictor** that remains reliable when model inputs are uncertain (for example, GPS position/velocity measurement error). Standard deep kernel pipelines usually treat inputs as noise-free. DKRL explicitly propagates input covariance through prediction and uncertainty estimation.

In short, the goal is:

1. high predictive accuracy,
2. calibrated uncertainty,
3. robust uncertainty decomposition (aleatoric vs epistemic),
4. reproducible evaluation with proper scoring and conformal coverage.

## 2. Core Contributions in This Codebase

The implemented system combines the following components:

1. **Second-order NIGP corrections** with Hessian-trace estimation via Hutchinson/Hutch++ (`dkrl/models/nigp.py`).
2. **Fused first-order NIGP trace kernel** `Tr(J P J^T)` from compressed covariance (`dkrl/models/kernels.py`).
3. **Bi-Lipschitz deep kernel layers** using dissipative parameterization + BatchEnsemble residual blocks (`dkrl/models/layers.py`).
4. **Low-rank GP head** with closed-form fitting and basis trace regularization (`dkrl/models/gp.py`).
5. **Composite training objective** (DB-MTL style balancing over NLL/NIGP/CRPS/tail/Lipschitz/TMCB/trace terms) (`dkrl/training/losses.py`).
6. **Conformal inference modules**: localized kernel conformal and leverage-based split conformal (`dkrl/inference/conformal.py`).

## 3. Integrated Method (Single Cohesive System)

Given features `x in R^19` and compressed covariance `P10 in R^10`:

1. Normalize `x` and map through a dissipative residual deep kernel.
2. Predict GP mean/variance in latent space (low-rank GP).
3. Compute first-order correction from Jacobian:
   - `v1 = Tr(J P J^T)`
4. Compute second-order correction from Hessian trace:
   - mean correction: `m2 = 0.5 * Tr(H P)`
   - variance correction: `v2 = 0.5 * (Tr(H P))^2`
5. Return corrected prediction:
   - `mu = mu_gp + m2`
   - `var = var_gp + v1 + v2`
6. Optionally split uncertainty into aleatoric/epistemic components.
7. Optionally wrap predictions with conformal calibration.

### System Flow

```text
Processed data -> create_dataset()
  -> features [N,19], cov [N,10], labels [N], quality [N]
  -> train/val/test indices

Training:
  features + cov -> augmentation -> NIGPDeepKernelGP forward
  -> GP dist + Jacobians + NIGP corrections
  -> composite loss + adaptive controls (GNS, ZClip, convergence)
  -> checkpoint

Inference/Eval:
  checkpoint + test split -> predict_with_nigp()
  -> scoring metrics + coverage + decomposition
  -> optional conformal intervals
```

## 4. Repository Structure

```text
dkrl/
  config/        # defaults, constants, hardware checks, YAML loader
  data/          # dataset creation, augmentation, validation
  models/        # dissipative layers, covariance ops, GP, NIGP, baselines
  training/      # trainer, losses, GP fitting utility
  inference/     # conformal and sigma-point wrappers
  evaluation/    # scoring metrics, evaluator, statistical tests
  utils/         # adaptive controls, I/O, seeding

scripts/
  train.py               # train from config
  eval.py                # calibration/conformal/audit evaluation
  sweep.py               # recursive experiment sweep + statistics
  train_baselines.py     # train baseline models (MC Dropout, Deep Ensemble, Standard GP)
  results_table.py       # LaTeX/plot utility (format-dependent)
```

## 5. Installation

```bash
git clone <your-repo-url>
cd deep_kernel_risk_learning
pip install -e .
```

### Runtime Requirements

- Python `>=3.10,<3.13`
- CUDA GPU required at runtime (the package raises if CUDA is unavailable)
- Core libs: PyTorch, GPyTorch, Triton, JAX (see `pyproject.toml`)

## 6. Data

The pipeline expects processed parquet data with this layout:

```text
<data_dir>/
  telemetry_partitioned/*.parquet
  traits/european_species_traits.parquet
  birdstrike/uk_caa_species_strikes_2023_2024.parquet
```

Set data directory via environment variable:

```bash
export DKRL_DATA_DIR=/absolute/path/to/data/processed
```

If `data_dir` is not passed explicitly, `create_dataset()` reads `DKRL_DATA_DIR`.

## 7. Reproducible Execution Flow

### Step 1: Train model

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --output checkpoints/model.pt
```

### Step 2: Evaluate calibration + conformal + audit

```bash
python scripts/eval.py \
  --mode all \
  --model_path checkpoints/model.pt \
  --output results/eval.json
```

### Step 3: Run experiment sweep

```bash
python scripts/sweep.py \
  --experiments experiments/ \
  --output_dir results/ \
  --models_dir checkpoints/
```

Outputs include aggregated metrics and pairwise statistical tests.

## 8. Configuration Surface (Current Code)

`NIGPModelConfig` currently exposes three active capacity knobs:

- `low_rank_dim`
- `hutchinson_samples`
- `num_ensemble`

These are the fields consumed by `config_to_model_config()` and used by training.

## 9. Evaluation Protocol

Implemented scoring/calibration utilities include:

- RMSE, MAE, NLL
- Coverage at 68/90/95%
- CRPS and scaled CRPS
- DSS
- Interval scores
- Tail-CRPS (upper/lower)
- AUSE
Main entry points:

- `dkrl/evaluation/evaluator.py`
- `dkrl/evaluation/metrics.py`

## 10. arXiv Papers and Implementation Mapping

This repository integrates ideas from multiple papers. The table below shows how each is reflected in code.

| Paper | Role in this repository | Primary modules |
|---|---|---|
| [arXiv:2509.14710](https://arxiv.org/abs/2509.14710) | Second-order NIGP corrections | `dkrl/models/nigp.py` |
| [arXiv:2505.18526](https://arxiv.org/abs/2505.18526) | Low-rank GP + trace regularization | `dkrl/models/gp.py`, `dkrl/training/losses.py` |
| [arXiv:2410.22258](https://arxiv.org/abs/2410.22258) | Dissipative network parameterization | `dkrl/models/layers.py` |
| [arXiv:2503.14297](https://arxiv.org/abs/2503.14297) | ECLipsE Lipschitz bounds | `dkrl/models/layers.py` |
| [arXiv:2404.12215](https://arxiv.org/abs/2404.12215) | Scoring-rule uncertainty decomposition | `dkrl/models/nigp.py`, `dkrl/evaluation/metrics.py` |
| [arXiv:2407.00650](https://arxiv.org/abs/2407.00650) | Tail-weighted (rectified) CRPS | `dkrl/training/losses.py`, `dkrl/evaluation/metrics.py` |
| [arXiv:2506.13687](https://arxiv.org/abs/2506.13687) | TMCB tail miscalibration term | `dkrl/training/losses.py` |
| [arXiv:1812.06162](https://arxiv.org/abs/1812.06162) | Gradient noise scale batch control | `dkrl/training/batch_sizing.py`, `dkrl/training/gradient_clip.py` |
| [arXiv:2504.02507](https://arxiv.org/abs/2504.02507) | ZClip adaptive gradient clipping | `dkrl/training/batch_sizing.py`, `dkrl/training/gradient_clip.py` |
| [arXiv:2308.12029](https://arxiv.org/abs/2308.12029) | DB-MTL-style loss balancing | `dkrl/training/losses.py` |
| [arXiv:2306.06101](https://arxiv.org/abs/2306.06101) | Prodigy optimizer usage | `dkrl/training/trainer.py` |
| [arXiv:2507.20272](https://arxiv.org/abs/2507.20272) | Leverage-based conformal calibration | `dkrl/inference/conformal.py` |
| [arXiv:2002.09112](https://arxiv.org/abs/2002.09112) | Sigma-point uncertainty propagation baseline | `dkrl/inference/sigma_point.py` |

## 11. Testing

```bash
pytest -q
```

Notes:

- Most model tests require CUDA.
- Metric tests require JAX.

## 12. Current Scope and Reproducibility Notes

- Runtime is CUDA-only (`dkrl/config/hardware.py`).
- `scripts/sweep.py` reads experiment YAMLs recursively and reports aggregate metrics + Welch t-tests.
- The YAML parser currently applies only the three `NIGPModelConfig` fields above for model construction.
- In `scripts/sweep.py`, all experiments train `NIGPDeepKernelGP`; `sigma_point: true` switches evaluation to the sigma-point wrapper, but `model_type` entries in YAML are not currently used by the sweep runner.
- `scripts/reproduce_results.sh` includes `--num-seeds`, which is not a valid CLI argument for `scripts/sweep.py`; use the explicit commands in Section 7 instead.
- `scripts/results_table.py` expects a specific JSON layout; use it only with matching input format.
