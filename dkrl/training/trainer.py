"""
Training loop for NIGP-DKL.

Features: Prodigy optimizer, FP32+TF32 precision, convergence early stopping,
GNS dynamic batch sizing, ZClip gradient clipping, DB-MTL loss weighting.
"""

import math
import time

import torch
from prodigyopt import Prodigy
from torch.utils.data import DataLoader, TensorDataset

from dkrl.config.defaults import NIGPModelConfig
from dkrl.config.hardware import DEVICE, configure_cuda_backends
from dkrl.data import DatasetBundle
from dkrl.data.augmentation import augment_features_19d
from dkrl.models.covariance import build_cov_19x19
from dkrl.models.layers import DissipativeLinear
from dkrl.models.nigp import NIGPDeepKernelGP
from dkrl.training.batch_sizing import GNSBatchSizeController, compute_batch_size
from dkrl.training.gradient_clip import ZClipGradientClipper
from dkrl.training.losses import LowRankNIGPLoss, differentiable_tmcb
from dkrl.training.train_utils import (
    check_convergence,
    compute_lipschitz_bounds,
    fit_low_rank_gp,
)
from dkrl.utils.seeding import set_all_seeds


def train_nigp_model(
    bundle: DatasetBundle,
    *,
    model_config: NIGPModelConfig,
) -> tuple[NIGPDeepKernelGP, dict]:
    """
    Train NIGP-DKL model with fully adaptive hyperparameters.
    """
    configure_cuda_backends()
    set_all_seeds(model_config.seed)

    X_train_np, y_train_np, P_train_np, Q_train_np = bundle.get_split("train", normalize_quality=True)
    X_val_np, y_val_np, P_val_np, _ = bundle.get_split("val", normalize_quality=False)

    X = torch.tensor(X_train_np, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y_train_np, dtype=torch.float32, device=DEVICE)
    P = torch.tensor(P_train_np, dtype=torch.float32, device=DEVICE)
    Q = torch.tensor(Q_train_np, dtype=torch.float32, device=DEVICE)

    X_val = torch.tensor(X_val_np, dtype=torch.float32, device=DEVICE)
    y_val = torch.tensor(y_val_np, dtype=torch.float32, device=DEVICE)
    P_val = torch.tensor(P_val_np, dtype=torch.float32, device=DEVICE)

    n_samples = len(X)
    input_dim = X.shape[1]
    max_batch_size = compute_batch_size(n_samples, input_dim)
    min_lipschitz, _ = compute_lipschitz_bounds(X)

    # sqrt(N) initial batch size (Bottou 2010), GNS adapts from there
    initial_batch_size = min(max(32, int(math.sqrt(n_samples))), max_batch_size)
    batch_controller = GNSBatchSizeController.from_data(
        n_samples=n_samples,
        initial_batch_size=initial_batch_size,
        max_batch_size=max_batch_size,
    )

    y_mean = y.mean()
    y_std = y.std()
    y_normalized = (y - y_mean) / y_std
    y_val_normalized = (y_val - y_mean) / y_std

    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0)

    val_data = (X_val, y_val_normalized, P_val, None, y_val)

    train_tensors = [X, y_normalized, P, Q, y]
    train_dataset = TensorDataset(*train_tensors)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_controller.current_batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(model_config.seed),
    )

    model = NIGPDeepKernelGP(
        input_dim=input_dim,
        hidden_dims=model_config.hidden_dims,
        num_ensemble=model_config.num_ensemble,
        low_rank_dim=model_config.low_rank_dim,
        hutchinson_samples=model_config.hutchinson_samples,
        hessian_mode=model_config.hessian_mode,
        lipschitz_gamma=model_config.lipschitz_gamma,
    ).to(DEVICE)
    model.set_normalization(X_mean, X_std)
    model.set_target_normalization(y_mean, y_std)

    target_lipschitz = model_config.lipschitz_gamma

    print(f"status=init batch_size={initial_batch_size}->{max_batch_size} lipschitz=[{min_lipschitz:.4f}, {target_lipschitz:.4f}]")

    optimizer = Prodigy(model.parameters(), lr=1.0, betas=(0.9, 0.999), safeguard_warmup=True)

    loss_fn = LowRankNIGPLoss(
        feature_extractor=model.feature_extractor,
        min_lipschitz=min_lipschitz,
        target_lipschitz=target_lipschitz,
        tail_quantile=model_config.tail_quantile,
    )

    grad_clipper = ZClipGradientClipper.from_data(n_samples=n_samples, batch_size=initial_batch_size)

    history = {
        "train_loss": [],
        "val_rmse": [],
        "val_mae": [],
        "batch_size": [],
    }

    lipschitz_interval = model_config.lipschitz_interval
    # GNS micro-batch interval: ~10% of batches per epoch
    n_batches_per_epoch = max(1, n_samples // initial_batch_size)
    gns_micro_batch_interval = max(5, n_batches_per_epoch // 10)
    train_X_all = train_loader.dataset.tensors[0]
    train_y_all = train_loader.dataset.tensors[1]
    fit_low_rank_gp(model, train_X_all, train_y_all)

    best_state = None
    best_val_rmse = float("inf")
    epoch = 0

    # Relative TMCB scaling: amplification relative to initial miscalibration
    epoch_tmcb_scale = 1.0
    baseline_tmcb = None

    # Residual-norm GP refit trigger (replaces fixed interval)
    _last_refit_residual = None

    max_wall_seconds = model_config.max_wall_seconds
    start_time = time.monotonic()

    while True:
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_preds = []  # PIT accumulation for epoch-level TMCB

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            batch_features, batch_labels, batch_cov, batch_weights = batch[:4]

            if model_config.augmentation:
                features_input = augment_features_19d(batch_features, batch_cov, build_cov_19x19)
            else:
                features_input = batch_features

            compute_regularization = (batch_idx % lipschitz_interval == 0)

            dist, jacobian_feat, nigp_1st, nigp_2nd, _ = (
                model.forward_with_jacobian_and_hutchinson_trace(
                    features_input, batch_cov,
                    skip_feature_jacobian=not compute_regularization,
                )
            )

            # GP basis trace regularization (anti-collapse)
            h_for_trace = model.feature_forward(model.normalize(features_input))
            trace_reg = model.gp_layer.compute_trace_regularization(h_for_trace)

            # Compute per-batch TMCB with epoch-level scaling for gradient signal
            pred_mean_batch = dist.mean
            pred_var_batch = dist.variance
            batch_tmcb = differentiable_tmcb(pred_mean_batch, pred_var_batch, batch_labels)
            epoch_scaled_tmcb = epoch_tmcb_scale * batch_tmcb

            loss = loss_fn(
                dist, batch_labels,
                jacobian_feat=jacobian_feat,
                cov_10=batch_cov,
                sample_weights=batch_weights,
                nigp_1st_precomputed=nigp_1st,
                nigp_2nd_precomputed=nigp_2nd,
                trace_reg=trace_reg,
            )

            # Add epoch-scaled TMCB as gradient-bearing auxiliary loss
            loss = loss + epoch_scaled_tmcb

            # Accumulate predictions for epoch-level TMCB
            epoch_preds.append((
                pred_mean_batch.detach(),
                pred_var_batch.detach(),
                batch_labels.detach(),
            ))

            loss.backward()

            clip_norm = grad_clipper.compute_clip_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            # GNS: use micro-batch gradient splitting periodically for accurate
            # noise/signal estimation; fall back to single-batch approximation otherwise
            use_micro_batch = (batch_idx % gns_micro_batch_interval == 0 and len(batch_features) >= 4)
            if use_micro_batch:
                # Save full-batch gradients
                saved_grads = {
                    p: p.grad.clone() for p in model.parameters() if p.grad is not None
                }

                half = len(batch_features) // 2
                micro_grads = []
                for start, end in [(0, half), (half, 2 * half)]:
                    optimizer.zero_grad(set_to_none=True)
                    mf = features_input[start:end]
                    ml = batch_labels[start:end]
                    md = model(mf)
                    micro_loss = -md.log_prob(ml).mean()
                    micro_loss.backward()
                    grad_vec = torch.cat([
                        p.grad.detach().flatten() for p in model.parameters()
                        if p.grad is not None
                    ])
                    micro_grads.append(grad_vec)

                batch_changed = batch_controller.update_from_micro_batches(
                    micro_grads[0], micro_grads[1], len(batch_features),
                )

                # Restore full-batch gradients for optimizer step
                optimizer.zero_grad(set_to_none=True)
                for p in model.parameters():
                    if p in saved_grads:
                        p.grad = saved_grads[p]
            else:
                batch_changed = batch_controller.update(model)
            if batch_changed:
                new_batch_size = batch_controller.current_batch_size
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=new_batch_size,
                    shuffle=True,
                    pin_memory=False,
                    num_workers=0,
                    generator=torch.Generator().manual_seed(model_config.seed),
                )
                # Update GNS interval for new batch size
                n_batches_per_epoch = max(1, n_samples // new_batch_size)
                gns_micro_batch_interval = max(5, n_batches_per_epoch // 10)
                print(f"status=batch_size_update batch_size={new_batch_size}")

            optimizer.step()

            for module in model.modules():
                if isinstance(module, DissipativeLinear):
                    module.invalidate_cache()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_loss)
        history["batch_size"].append(batch_controller.current_batch_size)

        # Epoch-level TMCB: compute on full dataset to scale next epoch's per-batch TMCB
        # Relative scaling: amplification proportional to initial miscalibration
        if epoch_preds:
            all_mean = torch.cat([p[0] for p in epoch_preds])
            all_var = torch.cat([p[1] for p in epoch_preds])
            all_target = torch.cat([p[2] for p in epoch_preds])
            epoch_tmcb_val = differentiable_tmcb(all_mean, all_var, all_target).item()
            if baseline_tmcb is None:
                baseline_tmcb = max(epoch_tmcb_val, 1e-6)
            epoch_tmcb_scale = 1.0 + epoch_tmcb_val / baseline_tmcb
            del epoch_preds

        # Residual-norm GP refit: refit when features have drifted >5%
        with torch.no_grad():
            h = model.feature_forward(model.normalize(train_X_all))
            pred = model.gp_layer(h).mean
            residual_norm = ((train_y_all - pred) ** 2).mean().sqrt().item()

        if _last_refit_residual is None or abs(residual_norm - _last_refit_residual) / max(_last_refit_residual, 1e-8) > 0.05:
            with torch.no_grad():
                fit_low_rank_gp(model, train_X_all, train_y_all)
            _last_refit_residual = residual_norm

        val_features, val_labels_norm, _, _, val_labels_orig = val_data[:5]
        model.eval()
        with torch.no_grad():
            f_val = model(val_features)
            pred_mean_norm = f_val.mean
            pred_mean = pred_mean_norm * model.target_std + model.target_mean
            rmse = torch.sqrt(((pred_mean - val_labels_orig) ** 2).mean()).item()
            mae = (pred_mean - val_labels_orig).abs().mean().item()
        history["val_rmse"].append(rmse)
        history["val_mae"].append(mae)

        if rmse < best_val_rmse:
            best_val_rmse = rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"status=epoch epoch={epoch+1} loss={avg_loss:.4f} val_rmse={rmse:.4f} val_mae={mae:.4f} batch={batch_controller.current_batch_size}")

        if check_convergence(history["val_rmse"], n_train=n_samples, epoch=epoch):
            print(f"status=converged epoch={epoch+1}")
            break

        if time.monotonic() - start_time > max_wall_seconds:
            print(f"status=timeout wall_seconds={max_wall_seconds}")
            break

        epoch += 1

    model.load_state_dict(best_state)

    fit_low_rank_gp(model, train_X_all, train_y_all)

    print(f"status=complete best_val_rmse={best_val_rmse:.4f}")
    return model, history
