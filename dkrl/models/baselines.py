"""
Baseline Models for Ablation Studies.

Provides comparison models that share the same prediction interface
predict(x, cov_10) -> (mean, var) as NIGPDeepKernelGP.

Models:
- MCDropoutModel: MC Dropout uncertainty via stochastic forward passes
- DeepEnsembleModel: Ensemble of independent MLPs
- StandardGPModel: Standard GP on raw features (no deep kernel)

Reference: RESEARCH_OBJECTIVES.md Objective 4 — Robust Evaluation Protocol
"""

import gpytorch
import torch
import torch.nn as nn
from gpytorch.distributions import MultivariateNormal

from dkrl.models.normalization import NormalizationMixin


class MCDropoutModel(NormalizationMixin, nn.Module):
    """
    MC Dropout baseline for uncertainty estimation.

    Same MLP architecture as NIGPDeepKernelGP feature extractor but with
    dropout layers. At inference, runs multiple stochastic forward passes
    and computes mean/variance from the ensemble of predictions.

    Reference: Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
    """

    def __init__(
        self,
        input_dim: int = 19,
        hidden_dim: int = 76,
        output_dim: int = 1,
        dropout_rate: float = 0.1,
        n_forward: int = 30,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_forward = n_forward

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

        self.register_normalization_buffers(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.normalize(x)
        return self.network(x_norm)

    def predict(
        self, x: torch.Tensor, cov_10: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout prediction with uncertainty."""
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(self.n_forward):
                pred_norm = self.forward(x).squeeze(-1)
                pred = pred_norm * self.target_std + self.target_mean
                preds.append(pred)
        preds = torch.stack(preds)  # [n_forward, N]
        mean = preds.mean(dim=0)
        var = preds.var(dim=0)
        self.eval()
        return mean, var

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pred = self.forward(x).squeeze(-1)
        y_norm = (y - self.target_mean) / self.target_std
        return nn.functional.mse_loss(pred, y_norm)


class DeepEnsembleModel(NormalizationMixin, nn.Module):
    """
    Deep Ensemble baseline for uncertainty estimation.

    Trains K independent MLPs and combines their predictions.
    Mean = average of individual means.
    Variance = average of individual variances + variance of individual means.

    Reference: Lakshminarayanan et al. (2017) "Simple and Scalable
    Predictive Uncertainty Estimation using Deep Ensembles"
    """

    def __init__(
        self,
        input_dim: int = 19,
        hidden_dim: int = 76,
        n_members: int = 5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_members = n_members

        # Each member outputs (mean, log_var)
        self.members = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 2),  # mean + log_var
            )
            for _ in range(n_members)
        ])

        self.register_normalization_buffers(input_dim)

    def forward_member(self, x: torch.Tensor, member_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single ensemble member."""
        x_norm = self.normalize(x)
        out = self.members[member_idx](x_norm)
        mean_norm = out[:, 0]
        log_var = out[:, 1]
        return mean_norm, log_var

    def predict(
        self, x: torch.Tensor, cov_10: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ensemble prediction with uncertainty decomposition."""
        means = []
        vars_ = []
        with torch.no_grad():
            for k in range(self.n_members):
                mean_norm, log_var = self.forward_member(x, k)
                mean = mean_norm * self.target_std + self.target_mean
                var = torch.exp(log_var) * (self.target_std ** 2)
                means.append(mean)
                vars_.append(var)

        means = torch.stack(means)  # [K, N]
        vars_ = torch.stack(vars_)  # [K, N]

        # Ensemble mean and variance
        ensemble_mean = means.mean(dim=0)
        # Law of total variance: Var = E[Var] + Var[E]
        ensemble_var = vars_.mean(dim=0) + means.var(dim=0)

        return ensemble_mean, ensemble_var

    def train_step(self, x: torch.Tensor, y: torch.Tensor, member_idx: int) -> torch.Tensor:
        mean_norm, log_var = self.forward_member(x, member_idx)
        y_norm = (y - self.target_mean) / self.target_std
        var = torch.exp(log_var)
        return (log_var + (y_norm - mean_norm) ** 2 / var).mean()


class _ExactGPModel(gpytorch.models.ExactGP):
    """GPyTorch ExactGP with ARD RBF kernel."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1))
        )

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class StandardGPModel(NormalizationMixin, nn.Module):
    """
    Standard GP on raw features (no deep kernel) using GPyTorch ExactGP.

    For large datasets, uses a random subset for training due to O(N^3) scaling.
    Provides a baseline to measure the value of the deep kernel.

    Reference: Rasmussen & Williams (2006) "Gaussian Processes for Machine Learning"
    """

    def __init__(self, input_dim: int = 19, max_train: int = 2000):
        super().__init__()
        self.input_dim = input_dim
        self.max_train = max_train

        self.register_normalization_buffers(input_dim)

        self.gp_model: _ExactGPModel | None = None
        self.likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None

    def fit(self, X: torch.Tensor, y: torch.Tensor, n_iter: int = 100):
        # Subsample if too large
        N = X.shape[0]
        if N > self.max_train:
            idx = torch.randperm(N, device=X.device)[:self.max_train]
            X = X[idx]
            y = y[idx]

        # Normalize
        X_norm = self.normalize(X)
        y_norm = (y - self.target_mean) / self.target_std

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(X.device)
        self.gp_model = _ExactGPModel(X_norm, y_norm.squeeze(), self.likelihood).to(X.device)

        self.gp_model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        for _ in range(n_iter):
            optimizer.zero_grad()
            output = self.gp_model(X_norm)
            loss = -mll(output, y_norm.squeeze())
            loss.backward()
            optimizer.step()

    def predict(
        self, x: torch.Tensor, cov_10: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GP prediction (ignores input covariance — no NIGP)."""
        self.gp_model.eval()
        self.likelihood.eval()

        x_norm = self.normalize(x)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.gp_model(x_norm))
            mean = pred.mean * self.target_std + self.target_mean
            var = pred.variance * (self.target_std ** 2)

        return mean, var
