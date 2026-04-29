# Research Objectives & Canonical Mapping

## 1. Core Research Problem

Calibrated regression under input uncertainty: predict bird-strike risk from GPS telemetry where each input carries a known measurement covariance (position/velocity errors). Standard deep kernel learning treats inputs as noise-free. This system propagates input covariance through both prediction and uncertainty estimation via second-order Noisy-Input GP corrections, producing calibrated predictive intervals with decomposed aleatoric/epistemic uncertainty.

## 2. Novel Contributions & Mechanisms

### 2a. Second-Order NIGP Corrections (arXiv:2509.14710)

**What**: Propagate input covariance P through the GP predictive mean and variance using first-order Jacobian trace and second-order Hessian trace corrections.

**Math**:
- 1st-order variance: `v1 = Tr(J_gp @ P @ J_gp^T)` where `J_gp = d(mu_gp)/dx`
- 2nd-order mean: `m2 = 0.5 * Tr(H @ P)` where `H = d²(mu_gp)/dx²`
- 2nd-order variance: `v2 = 0.5 * (Tr(H @ P))²`
- Corrected output: `mu = mu_gp + m2`, `var = var_gp + v1 + v2`

**Code**:
- `dkrl/models/nigp.py:NIGPDeepKernelGP._compute_gp_mean_jacobian` — J_gp via `torch.vmap(torch.func.jacrev(...))`
- `dkrl/models/nigp.py:NIGPDeepKernelGP._hutchpp_nigp_correction` — Hutch++ Hessian trace estimator (O(1/m²) variance)
- `dkrl/models/nigp.py:NIGPDeepKernelGP._hutchinson_standard` — Standard Hutchinson (O(1/m) variance, ablation baseline)
- `dkrl/models/kernels.py:nigp_trace` — Fused Triton kernel: `Tr(J @ P @ J^T)` directly from compressed 10-element covariance (no 19×19 materialization)
- `dkrl/models/nigp.py:NIGPDeepKernelGP.predict_with_nigp` — Inference path combining all corrections

### 2b. Bi-Lipschitz Deep Kernel (arXiv:2410.22258 + arXiv:2503.14297)

**What**: Dissipative parameterization `W = sqrt(γ) * U @ diag(σ(s)) @ V^T` guarantees `||W||_2 ≤ sqrt(γ)` by construction, combined with BatchEnsemble residual blocks for uncertainty via ensemble disagreement.

**Code**:
- `dkrl/models/layers.py:DissipativeLinear` — SVD-parameterized layer with QR-cached orthogonal bases
- `dkrl/models/layers.py:DissipativeBatchEnsembleLinear` — Ensemble scaling with tanh-constrained r,s vectors
- `dkrl/models/layers.py:ResidualDissipativeBlock` — Residual block: `project(x) + activation(dissipative(x))`
- `dkrl/models/layers.py:compute_eclipse_tightened_bound` — ECLipsE tightest Lipschitz bound across GC/SN/Shift variants
- `dkrl/models/layers.py:compute_lower_lipschitz_penalty` — Bi-Lipschitz lower bound via `σ_min(J)`

### 2c. Low-Rank GP Head (arXiv:2505.18526)

**What**: O(N) exact GP inference via learned basis functions `φ(x)` with closed-form fitting `α = (Φ^T Φ + σ²I)^{-1} Φ^T y`, plus anti-collapse trace regularization.

**Code**:
- `dkrl/models/gp.py:LowRankGPLayer` — Basis network `φ: R^d → R^r`, closed-form `fit()`, predictive `forward()`
- `dkrl/models/gp.py:LowRankGPLayer.compute_trace_regularization` — Penalizes variance in basis norms to prevent rank-1 collapse

### 2d. Composite DB-MTL Loss (arXiv:2308.12029)

**What**: 10-term objective with log-transform scale invariance and gradient magnitude balancing. Each loss is transformed via `log(1 + |L_i|)` and scaled by `max_norm / grad_norm_i`.

**Terms**: NLL, NIGP-1st, NIGP-2nd, CRPS, tail-CRPS-upper, tail-CRPS-lower, lower-Lipschitz, ECLipsE-bound, TMCB, trace-regularization.

**Code**:
- `dkrl/training/losses.py:LowRankNIGPLoss` — DB-MTL loss with exponential gradient-balance schedule
- `dkrl/training/losses.py:_crps_rectified_gaussian_torch` — Exact rectified Gaussian CRPS (arXiv:2407.00650)
- `dkrl/training/losses.py:differentiable_tmcb` — Tail MisCaliBration via Wasserstein-1 (arXiv:2506.13687)

### 2e. Conformal Prediction (arXiv:2507.20272)

**What**: Distribution-free prediction intervals via two methods: LCMQR (RBF-kernel-weighted quantiles) and Leverage-Split CP (LOO residuals via GP hat matrix diagonals).

**Code**:
- `dkrl/inference/conformal.py:KernelConformalizedNIGP` — LCMQR with auto-bandwidth Silverman's rule
- `dkrl/inference/conformal.py:LeverageSplitCP` — Jackknife+ quantile from `r_i^LOO = r_i / (1 - h_ii)`

### 2f. Compressed Covariance Format & Triton Kernels

**What**: 10-element compressed representation of rank-6 block-diagonal 19×19 covariance (only position/velocity have nonzero covariance). Fused GPU kernels avoid materializing the full matrix for 1st-order trace.

**Code**:
- `dkrl/models/covariance.py` — `build_cov_19x19` (Triton), `build_cov_6x6`, `normalize_cov_10`
- `dkrl/models/kernels.py:_cov_to_trace_kernel` — Triton kernel for `Tr(J @ P @ J^T)` directly from 10 elements

### 2g. Scoring-Rule Uncertainty Decomposition (arXiv:2404.12215)

**What**: CRPS-based aleatoric/epistemic decomposition via ensemble cross-scores: `AU = (1/K) Σ_k σ_k/√π`, `TU = (1/K²) Σ_{i,j} cross-CRPS(i,j)`, `EU = TU - AU`.

**Code**:
- `dkrl/models/nigp.py:NIGPDeepKernelGP.compute_scoring_rule_decomposition` — PyTorch (training/inference)
- `dkrl/evaluation/metrics.py:compute_epistemic_crps` — JAX (evaluation, JIT-compiled)

## 3. Primary Execution Path

### 3a. Data

**Source**: GPS telemetry parquets + species traits parquet + UK CAA birdstrike damage rates parquet.

**Pipeline**: `dkrl/data/pipeline.py:create_dataset()`
1. Load telemetry → filter (NULL GPS, large dt, single-point segments)
2. Compute 19D features via `dkrl/data/features.py:compute_features_19d` — position (ENU), velocity (finite diff), speed, heading, vertical velocity, time encoding, species traits, spatial context
3. Compute 10-element compressed covariance via `dkrl/data/covariance_pipeline.py:compute_covariance_10`
4. Compute quality scores via `dkrl/data/quality.py:compute_quality_scores`
5. Calibrate risk labels via `dkrl/data/labels.py:calibrate_risk_labels` — Bayesian combination of kinematic risk (altitude/speed Gaussian) and UK CAA damage rate prior
6. Stratified train/val/test split via `dkrl/data/splitting.py:stratified_split`

**Output**: `DatasetBundle(features=[N,19], labels=[N], covariances=[N,10], quality_weights=[N], indices)`

### 3b. Training

**Entry**: `scripts/train.py` → `dkrl/training/trainer.py:train_nigp_model()`

1. Initialize `NIGPDeepKernelGP` with dissipative residual blocks + low-rank GP head
2. Set input/target normalization from training statistics
3. Initial GP fit via closed-form `fit_low_rank_gp()`
4. Train loop:
   - Optional covariance-based data augmentation (`augment_features_19d`)
   - Forward: `model.forward_with_jacobian_and_hutchinson_trace()` → GP distribution + Jacobians + NIGP corrections
   - Loss: `LowRankNIGPLoss.forward()` → DB-MTL log-transform + gradient magnitude balancing over 10 terms
   - Adaptive batch sizing via GNS (`GNSBatchSizeController`, arXiv:1812.06162)
   - Adaptive gradient clipping via ZClip (`ZClipGradientClipper`, arXiv:2504.02507)
   - Convergence detection via OLS slope (`check_convergence`)
   - Residual-norm-triggered GP refit (>5% drift)
5. Restore best-val-RMSE checkpoint, final GP refit

### 3c. Inference

**Entry**: `scripts/eval.py` → `dkrl/evaluation/evaluator.py:evaluate_nigp_model()`

1. `model.predict_with_nigp(X, P)` → corrected `(mean, var)` in original scale
2. Coverage at 68/90/95% via normal quantiles
3. Scoring rules via `dkrl/evaluation/metrics.py:compute_all_scores()` (JAX JIT):
   - CRPS, Scaled CRPS, DSS, Interval Scores (90/95), AUSE, Tail-CRPS (upper/lower)
4. Uncertainty decomposition via `model.predict_with_decomposed_uncertainty()`
5. Epistemic CRPS decomposition via ensemble cross-scores
6. Optional conformal prediction: LCMQR + LeverageSplitCP

### 3d. Ablation Sweep

**Entry**: `scripts/sweep.py`

Reads experiment YAMLs from `experiments/`, trains with N seeds per config, evaluates, runs pairwise Welch t-tests against NIGP-DKL baseline. Statistical significance at p<0.05.

## 4. SOTA Baselines & Ablation Structure

### Ablation Experiments (single-delta from SOTA)

| Experiment | Delta from NIGP-DKL | Tests |
|------------|---------------------|-------|
| `SOTA-minus-HutchPP` | `hessian_mode: hutchinson` | Value of Hutch++ over plain Hutchinson |
| `SOTA-minus-Augmentation` | `augmentation: false` | Value of covariance-based augmentation |
| `Baseline-DKL` | `hutchinson_samples: 10` | Sensitivity to Hessian sample count |

### Comparison Baselines (different architectures)

| Baseline | Architecture | Script |
|----------|-------------|--------|
| `SigmaPoint-NIGP` | Same model, UT evaluation instead of Hutchinson | `sweep.py` with `sigma_point: true` |
| `SigmaPoint-SOTA` | Same model, UT evaluation with SOTA config | `sweep.py` |
| `MC-Dropout` | 2-layer MLP + dropout, 30 stochastic passes | `train_baselines.py` |
| `Deep-Ensemble` | 5-member MLP ensemble | `train_baselines.py` |
| `Standard-GP` | GPyTorch ExactGP on raw features | `train_baselines.py` |
| `Standard-GP-DKL` | Standard DKL without NIGP corrections | `sweep.py` |
| `Static-KF-DKL` | DKL with Kalman filter covariance | `sweep.py` |

## 5. Evaluation Metrics — Mathematical Definitions

### Proper Scoring Rules

| Metric | Formula | Library Status |
|--------|---------|---------------|
| **CRPS** | `σ[z(2Φ(z)-1) + 2φ(z) - 1/√π]`, `z = (y-μ)/σ` | `scoringrules.crps_normal` (v0.9.0, JAX + PyTorch backends). Replaces custom implementation. |
| **Scaled CRPS** | `CRPS / std(y)` | Derived from CRPS. |
| **DSS** | `log(σ²) + (y-μ)²/σ²` | Custom. Trivial formula, no library needed. |
| **Interval Score** | `(u-l) + (2/α)(l-y)𝟙{y<l} + (2/α)(y-u)𝟙{y>u}` | `scoringrules.interval_score` (v0.9.0, JAX backend). Replaces custom implementation. |
| **Tail-CRPS** | Exact rectified Gaussian CRPS (arXiv:2407.00650) | Custom. No standard implementation exists. |
| **AUSE** | Area under sparsification error curve, normalized by oracle/random | Custom. No standard library. |
| **Epistemic CRPS** | Cross-CRPS ensemble decomposition (arXiv:2404.12215) | Custom. Novel metric. |

### Standard Metrics

| Metric | Formula | Library Status |
|--------|---------|---------------|
| **RMSE** | `√(mean((ŷ-y)²))` | Inline (1 line). `torchmetrics` would add a dependency for no benefit. |
| **MAE** | `mean(|ŷ-y|)` | Inline (1 line). |
| **NLL** | `-mean(log p(y|μ,σ))` | `torch.distributions.Normal.log_prob` — already using stdlib. |
| **Coverage** | `mean(𝟙{l ≤ y ≤ u})` at z-quantiles | Inline (1 line per level). |
| **Calibration Error** | `|coverage - target|` | Derived from coverage. |

### Statistical Testing

| Test | Implementation | Library |
|------|---------------|---------|
| **Welch's t-test** | `dkrl/evaluation/statistical.py:run_all_pairwise_tests` | `scipy.stats.ttest_ind` — already using stdlib. |
| **Cohen's d** | Effect size: `(μ_baseline - μ_exp) / σ_pooled` | Inline (1 line). |

## 6. Constraint Checklist

- [ ] No defensive guards — The environment is deterministic (CUDA-only, fixed data schema, pre-filtered data with EDA-verified guarantees). Guards against missing columns, NaN fallbacks, and type validation were removed in prior cleanup.
- [ ] No custom standard metrics — RMSE, MAE, NLL, Coverage are 1-line formulas. Importing `torchmetrics` for these would add dependency weight with zero benefit.
- [ ] Custom scoring rules are justified — CRPS, Tail-CRPS, AUSE, Epistemic-CRPS have no maintained standard library implementations at the required precision/scale. The JAX JIT versions are the SOTA path.
- [ ] SOTA-by-default execution — `configs/default.yaml` + `experiments/main/nigp_dkl.yaml` produce the full NIGP-DKL system. No flags needed to enable novel mechanisms.
- [ ] Single-delta ablation structure — Each ablation experiment changes exactly one config field from the NIGP-DKL default. Statistical significance tested via Welch's t-test (scipy).

## 7. File Dependency Graph (Novel Mechanisms Only)

```
Layer 0 (Constants):
  config/_constants.py

Layer 1 (Primitives):
  models/layers.py          ← Dissipative parameterization
  models/covariance.py      ← Compressed covariance ops (Triton)
  models/kernels.py         ← Fused NIGP trace kernel (Triton)

Layer 2 (Core Model):
  models/gp.py              ← Low-rank GP head
  models/nigp.py            ← NIGPDeepKernelGP (integrates Layers 0-1)

Layer 3 (Training):
  training/losses.py        ← DB-MTL composite loss
  training/trainer.py       ← Training loop with adaptive controls

Layer 4 (Inference):
  inference/conformal.py    ← LCMQR + Leverage-Split CP
  inference/sigma_point.py  ← Unscented Transform baseline

Layer 5 (Evaluation):
  evaluation/metrics.py     ← JAX scoring rules
  evaluation/evaluator.py   ← Full evaluation pipeline

Entry Points:
  scripts/train.py          ← Single-model training
  scripts/eval.py           ← Calibration + conformal evaluation
  scripts/sweep.py          ← Ablation sweep + statistical tests
  scripts/train_baselines.py ← Baseline model training
```
