import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern, ConstantKernel
import datetime
from utils.logger import get_logger
import warnings
from scipy.stats import norm
import pathlib
from utils.geometry import calculate_distance_to_path

logger = get_logger(__name__)

class SequentialGPRiskMapper:
    GP_TARGET_SCALE = 2.5

    def __init__(self, x_range=(-5, 5), y_range=(-5, 5), z_range=(0, 3), resolution=0.2):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.resolution = resolution

        self.grid_x, self.grid_y, self.grid_z = np.meshgrid(
            np.arange(x_range[0], x_range[1] + resolution, resolution),
            np.arange(y_range[0], y_range[1] + resolution, resolution),
            np.arange(z_range[0], z_range[1] + resolution, resolution),
            indexing='ij'
        )

        self.grid_shape = self.grid_x.shape
        logger.info(f"GP Grid shape: {self.grid_shape}")

        length_scale = [0.8, 0.8, 0.4]
        noise_level_init = 0.1
        self.kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=length_scale, nu=1.5) + WhiteKernel(noise_level=noise_level_init, noise_level_bounds=(1e-5, 1.0))

        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-5,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=42
        )

        self.max_history_size = 1000
        self.time_decay_constant_hours = 24
        self.max_alpha_multiplier = 25
        logger.info(f"GP Temporal Params: HistorySize={self.max_history_size}, DecayHours={self.time_decay_constant_hours}, MaxAlphaMult={self.max_alpha_multiplier}")

        self.X_train_history = None
        self.y_train_history = None
        self.timestamps = []
        self.temporal_weights = np.array([])
        self.initialized = False
        self.cached_map = None
        self.cached_std_dev = None
        self.cached_prob_high = None
        self.cached_prob_med_or_high = None
        self.cache_timestamp = None
        self.cache_valid_duration = datetime.timedelta(seconds=15)
        self.update_count = 0
        self.log_marginal_likelihood = -np.inf
        self.overall_confidence = 0.0

        medium_prob_equiv = 0.25
        high_prob_equiv = 0.60
        self.risk_threshold_medium = self.GP_TARGET_SCALE * medium_prob_equiv
        self.risk_threshold_high = self.GP_TARGET_SCALE * high_prob_equiv
        logger.info(f"Initialized GP Mapper. Target Scale={self.GP_TARGET_SCALE}. Risk Thresholds (Scaled): Med={self.risk_threshold_medium:.3f}, High={self.risk_threshold_high:.3f}")

    def _calculate_temporal_weights(self, current_time):
        if not self.timestamps:
            return np.array([])
        time_diffs_hours = np.maximum(0, np.array(
            [(current_time - ts).total_seconds() / 3600.0 for ts in self.timestamps]))
        weights = np.exp(-time_diffs_hours / self.time_decay_constant_hours)
        return weights

    def generate_training_data(self, bird_positions, species_list, family_list, flight_paths, faa_risk_system=None, bird_trackers=None, month=None, season=None):
        X_train = []
        y_train = []

        if not bird_positions or not faa_risk_system:
            logger.warning("[GP Train] No bird positions or FAA risk system provided. Returning empty training data.")
            return np.array([]).reshape(0, 3), np.array([])

        num_birds = len(bird_positions)
        logger.debug(f"[GP Train] Generating training data for {num_birds} bird positions.")

        for i, pos in enumerate(bird_positions):
            if pos is None:
                continue

            bird_id = bird_trackers[i].bird_id if bird_trackers and i < len(bird_trackers) else f"Idx_{i}"
            log_prefix = f"[GP Train Bird {bird_id}]"

            pos_tuple = tuple(pos)
            species = species_list[i % len(species_list)] if species_list else "Unknown"
            family = family_list[i % len(family_list)] if family_list else "Unknown"
            pos_unc_std = bird_trackers[i].get_position_uncertainty() if bird_trackers and i < len(
                bird_trackers) else None

            gp_target_value = 0.0

            min_dist = float('inf')
            nearest_path_info = {"id": "None", "name": "None"}

            if not flight_paths:

                nearest_path = None
            else:
                nearest_path = flight_paths[0]
                for path in flight_paths:
                    try:
                        dist = calculate_distance_to_path(pos_tuple, path)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_path = path
                            nearest_path_info = {"id": path.get('id', 'N/A'), "name": path.get('name', 'N/A')}
                    except Exception as e:
                        logger.error(f"{log_prefix} Error calculating distance to path '{path.get('id', 'N/A')}': {e}")
                        continue

            if nearest_path:

                try:
                    risk_metrics = faa_risk_system.calculate_probabilistic_risk_metrics(
                        pos_tuple, nearest_path, family, species, month, season
                    )
                    mean_prob = risk_metrics['mean_prob']
                    gp_target_value = mean_prob * self.GP_TARGET_SCALE

                except Exception as e:
                    logger.warning(
                        f"{log_prefix} FAA risk metrics calculation failed: {e}. Using default GP target (0.0).",
                        exc_info=False)
            else:

                pass

            X_train.append(pos_tuple)
            y_train.append(gp_target_value)

            if pos_unc_std is not None:
                for dim in range(2):
                    for sign in [-1, 1]:
                        if pos_unc_std[dim] > 1e-3:
                            v_pos = list(pos_tuple)
                            v_pos[dim] += sign * pos_unc_std[dim]
                            v_pos_tuple = tuple(v_pos)
                            uncertainty_target = gp_target_value * 0.8
                            X_train.append(v_pos_tuple)
                            y_train.append(uncertainty_target)

        if flight_paths:

            for path in flight_paths:
                path_type = path.get("type", "").lower()
                base_prob = 0.4 if path_type == "approach" else 0.3 if path_type == "departure" else 0.05
                start, end = np.array(path["start"]), np.array(path["end"])

                for t in np.linspace(0.1, 0.9, 3):
                    path_point = tuple(start + t * (end - start))
                    path_target = base_prob * self.GP_TARGET_SCALE
                    X_train.append(path_point)
                    y_train.append(path_target)

        return (np.array(X_train), np.array(y_train)) if X_train else (np.array([]).reshape(0, 3), np.array([]))

    def update_sequential(self, X_new, y_new, timestamp=None):
        if X_new is None or len(X_new) == 0:
            logger.debug("[GP Update] No new training data provided. Skipping update.")
            return self

        if timestamp is None:
            timestamp = datetime.datetime.now()

        new_timestamps = [timestamp] * len(X_new)

        if self.X_train_history is None or len(self.X_train_history) == 0:
            self.X_train_history = X_new
            self.y_train_history = y_new
            self.timestamps = new_timestamps
        else:
            self.X_train_history = np.vstack((self.X_train_history, X_new))
            self.y_train_history = np.append(self.y_train_history, y_new)
            self.timestamps.extend(new_timestamps)

        current_size = len(self.y_train_history)

        if current_size > self.max_history_size:
            self.temporal_weights = self._calculate_temporal_weights(timestamp)
            n_remove = current_size - self.max_history_size
            if len(self.temporal_weights) >= current_size:
                idx_remove = np.argsort(self.temporal_weights)[:n_remove]
                self.X_train_history = np.delete(self.X_train_history, idx_remove, axis=0)
                self.y_train_history = np.delete(self.y_train_history, idx_remove)
                self.timestamps = [ts for i, ts in enumerate(self.timestamps) if i not in idx_remove]
                logger.info(f"[GP Update] History pruned by {n_remove}. New size: {len(self.y_train_history)} (Max: {self.max_history_size}).")
                self.temporal_weights = self._calculate_temporal_weights(timestamp)
            else:
                logger.warning(f"[GP Update] History pruning skipped: Weight array size ({len(self.temporal_weights)}) != History size ({current_size}).")
                self.temporal_weights = np.array([])

        try:
            base_noise = self.gp.kernel_.k2.noise_level
        except AttributeError:
            base_noise = self.gp.kernel.k2.noise_level
        except Exception as e:
            logger.warning(f"[GP Update] Could not access kernel noise level: {e}. Using default 0.1")
            base_noise = 0.1

        min_alpha = base_noise
        max_alpha = base_noise * self.max_alpha_multiplier

        num_history = len(self.y_train_history)
        if 0 < num_history == len(self.temporal_weights):
            max_weight = np.max(self.temporal_weights)
            if max_weight > 1e-9:
                w_norm = self.temporal_weights / max_weight
            else:
                w_norm = np.zeros_like(self.temporal_weights)

            alpha_per_sample = np.clip(
                min_alpha + (max_alpha - min_alpha) * (1. - w_norm),
                min_alpha, max_alpha
            )
            self.gp.alpha = alpha_per_sample

        else:
            self.gp.alpha = base_noise

        try:
            fit_start_time = datetime.datetime.now()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=FutureWarning)
                self.gp.fit(self.X_train_history, self.y_train_history)

            fit_duration = (datetime.datetime.now() - fit_start_time).total_seconds()
            try:
                self.log_marginal_likelihood = self.gp.log_marginal_likelihood_value_
                lml_str = f"{self.log_marginal_likelihood:.2f}"
            except AttributeError:
                self.log_marginal_likelihood = -np.inf
                lml_str = "N/A"

            logger.info(f"[GP Update] GP fit completed in {fit_duration:.3f}s. LML={lml_str}")

            self.initialized = True
            self.update_count += 1
            self._update_overall_confidence()
            self.cached_map = None
            self.cached_std_dev = None
            self.cached_prob_high = None
            self.cached_prob_med_or_high = None
            self.cache_timestamp = None

        except Exception as e:
            logger.error(f"[GP Update] GP fitting failed: {e}.", exc_info=True)

        return self

    def _update_overall_confidence(self):
        if not self.initialized or not hasattr(self.gp, 'log_marginal_likelihood_value_'):
            self.overall_confidence = 0.0
            return

        lml = self.gp.log_marginal_likelihood_value_
        lml_min = -500
        lml_max = 1500
        if lml_max > lml_min:
            scaled_lml = (lml - lml_min) / (lml_max - lml_min)
        else:
            scaled_lml = 0.5
        conf_lml = np.clip(scaled_lml, 0., 1.)

        data_factor = min(1.,
                          len(self.y_train_history) / self.max_history_size) if self.y_train_history is not None and self.max_history_size > 0 else 0.
        self.overall_confidence = np.clip(0.7 * conf_lml + 0.3 * data_factor, 0.05, 0.95)
        logger.debug(f"[GP Confidence] Updated overall confidence: {self.overall_confidence:.3f} (LML Factor={conf_lml:.3f}, Data Factor={data_factor:.3f})")

    def predict_probabilistic(self, X):
        if not self.initialized:

            n = X.shape[0]
            default_std = self.GP_TARGET_SCALE / 2.0
            return np.zeros(n), np.ones(n) * default_std, np.zeros(n), np.zeros(n)
        try:
            mean_risk, std_dev = self.gp.predict(X, return_std=True)
            std_dev = np.maximum(std_dev, 1e-6)

        except Exception as e:
            logger.error(f"[GP Predict] GP prediction failed: {e}. Returning default prediction.", exc_info=False)
            n = X.shape[0]
            default_std = self.GP_TARGET_SCALE / 2.0
            return np.zeros(n), np.ones(n) * default_std, np.zeros(n), np.zeros(n)
        try:
            prob_high = 1. - norm.cdf(self.risk_threshold_high, loc=mean_risk, scale=std_dev)
            prob_med_or_high = 1. - norm.cdf(self.risk_threshold_medium, loc=mean_risk, scale=std_dev)

        except Exception as e:
            logger.error(f"[GP Predict] Probability calculation failed: {e}. Returning zeros.", exc_info=False)
            n = X.shape[0]
            return mean_risk, std_dev, np.zeros(n), np.zeros(n)
        return mean_risk, std_dev, np.clip(prob_high, 0., 1.), np.clip(prob_med_or_high, 0., 1.)

    def compute_risk_map(self, z_level=None, force_recompute=False):
        current_time = datetime.datetime.now()
        is_cache_valid = False
        if not force_recompute and self.cached_map is not None and self.cache_timestamp and (current_time - self.cache_timestamp) < self.cache_valid_duration:
            if z_level is None:
                is_cache_valid = True
        if is_cache_valid:

            return self.cached_map, self.cached_std_dev, self.cached_prob_high, self.cached_prob_med_or_high

        map_type = f"slice at Z={z_level:.2f}" if z_level is not None else "full 3D grid"

        if not self.initialized:
            logger.warning("[GP Map Compute] GP not initialized. Returning default zero map.")
            shape = self.grid_shape[:2] if z_level is not None else self.grid_shape
            default_std = self.GP_TARGET_SCALE / 2.0
            return np.zeros(shape), np.ones(shape) * default_std, np.zeros(shape), np.zeros(shape)

        if z_level is not None:
            grid_z_coords = self.z_range[0] + np.arange(self.grid_shape[2]) * self.resolution
            z_idx = np.argmin(np.abs(grid_z_coords - z_level))
            actual_z_level = grid_z_coords[z_idx]
            target_shape = self.grid_shape[:2]

            X_pred = np.array([(self.grid_x[i, j, z_idx], self.grid_y[i, j, z_idx], actual_z_level)
                               for i in range(target_shape[0]) for j in range(target_shape[1])])
        else:
            target_shape = self.grid_shape
            X_pred = np.vstack([self.grid_x.ravel(), self.grid_y.ravel(), self.grid_z.ravel()]).T

        try:
            mean_pred, std_dev_pred, prob_high_pred, prob_med_or_high_pred = self.predict_probabilistic(X_pred)
            risk_map = mean_pred.reshape(target_shape)
            std_dev_map = std_dev_pred.reshape(target_shape)
            prob_high_map = prob_high_pred.reshape(target_shape)
            prob_med_or_high_map = prob_med_or_high_pred.reshape(target_shape)

        except Exception as e:
            logger.error(f"[GP Map Compute] Error during map computation: {e}. Returning default map.", exc_info=True)
            shape = target_shape
            default_std = self.GP_TARGET_SCALE / 2.0
            return np.zeros(shape), np.ones(shape) * default_std, np.zeros(shape), np.zeros(shape)
        if z_level is None:
            self.cached_map, self.cached_std_dev = risk_map, std_dev_map
            self.cached_prob_high, self.cached_prob_med_or_high = prob_high_map, prob_med_or_high_map
            self.cache_timestamp = current_time

        return risk_map, std_dev_map, prob_high_map, prob_med_or_high_map

    def plot_temporal_difference_map(self, previous_risk_map, z_level=0.5, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        else:
            fig = ax.figure

        current_risk_map, _, _, _ = self.compute_risk_map(z_level=z_level)

        if previous_risk_map is None or previous_risk_map.shape != current_risk_map.shape:
            ax.text(0.5, 0.5, "No previous risk map for comparison",
                    transform=ax.transAxes, ha='center', va='center')
            return ax

        diff_map = current_risk_map - previous_risk_map
        cmap = plt.cm.RdBu_r
        abs_max = max(abs(np.min(diff_map)), abs(np.max(diff_map)))
        vmin, vmax = -abs_max, abs_max

        extent = [self.x_range[0], self.x_range[1], self.y_range[0], self.y_range[1]]
        contour = ax.contourf(
            self.grid_x[:, :, 0], self.grid_y[:, :, 0], diff_map,
            levels=15, cmap=cmap, alpha=0.8, extent=extent, vmin=vmin, vmax=vmax
        )

        thresholds = [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]
        line_contour = ax.contour(
            self.grid_x[:, :, 0], self.grid_y[:, :, 0], diff_map,
            levels=thresholds, colors='black', linewidths=0.5, alpha=0.7, extent=extent
        )
        ax.clabel(line_contour, inline=True, fontsize=8, fmt='%.1f')

        plt.colorbar(contour, ax=ax, label='Risk Probability Change', shrink=0.8)

        self._add_airport_features(ax, kwargs)

        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_title(f"Risk Change Map (z={z_level:.2f} km)", fontsize=12)
        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_aspect('equal', adjustable='box')

        return ax

    def plot_2d_risk_map(self, z_level=0.5, ax=None, plot_type='probability', adaptive_contours=True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        else:
            fig = ax.figure

        try:
            mean_map, std_map, prob_high_map, prob_med_or_high_map = self.compute_risk_map(z_level=z_level)
        except Exception as e:
            logger.error(f"[Plotting] Failed to compute risk map for plotting Z={z_level}: {e}", exc_info=True)
            ax.text(0.5, 0.5, "Error generating map data", transform=ax.transAxes, ha='center', va='center')
            return ax

        plot_map, cmap, cbar_label, title_suffix, vmin, vmax = None, 'viridis', 'Value', "", None, None
        if plot_type == 'mean':
            plot_map = mean_map
            cmap = 'YlOrRd'
            cbar_label = f'Mean Risk Score (Scale 0-{self.GP_TARGET_SCALE})'
            vmin, vmax = 0, self.GP_TARGET_SCALE
            title_suffix = "(Mean Score)"
        elif plot_type == 'std_dev':
            plot_map = std_map
            cmap = 'Blues'
            cbar_label = 'Predictive StdDev'
            vmin = 0
            vmax_sd = np.max(plot_map) if plot_map.size > 0 else 1.0
            vmax = min(vmax_sd, self.GP_TARGET_SCALE / 1.5)
            title_suffix = "(Uncertainty)"
        elif plot_type == 'probability':
            plot_map = prob_med_or_high_map
            cmap = 'YlOrRd'
            cbar_label = 'P(Risk>=Med)'
            vmin, vmax = 0, 1
            title_suffix = "(Prob Risk >= Med)"
        else:
            logger.error(f"Invalid plot_type '{plot_type}'. Defaulting to 'probability'.")
            plot_map = prob_med_or_high_map
            cmap = 'YlOrRd'
            cbar_label = 'P(Risk>=Med)'
            vmin, vmax = 0, 1
            title_suffix = "(Prob Risk >= Med)"

        if plot_map is None or plot_map.size == 0:
            logger.error(f"[Plotting] Map data for plot_type '{plot_type}' is empty.")
            ax.text(0.5, 0.5, f"No data for '{plot_type}'", transform=ax.transAxes, ha='center', va='center')
            return ax

        extent = [self.x_range[0], self.x_range[1], self.y_range[0], self.y_range[1]]

        if adaptive_contours:
            grad_y, grad_x = np.gradient(plot_map)
            grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            high_grad_threshold = np.percentile(grad_magnitude, 75)
            high_grad_areas = grad_magnitude > high_grad_threshold

            base_levels = np.linspace(vmin, vmax, 15)

            if plot_type == 'probability':
                critical_levels = np.linspace(0.2, 0.8, 12)
                levels = np.sort(np.unique(np.concatenate([base_levels, critical_levels])))
            else:
                levels = base_levels
        else:
            levels = np.linspace(vmin, vmax, 15)

        contour = ax.contourf(
            self.grid_x[:, :, 0], self.grid_y[:, :, 0], plot_map,
            levels=levels, cmap=cmap, alpha=0.8, extent=extent,
            vmin=vmin, vmax=vmax, extend='max'
        )

        line_contour = ax.contour(
            self.grid_x[:, :, 0], self.grid_y[:, :, 0], plot_map,
            levels=[0.25, 0.5, 0.75] if plot_type == 'probability' else levels[::3],
            colors='black', linewidths=0.5, alpha=0.6, extent=extent
        )
        ax.clabel(line_contour, inline=True, fontsize=8, fmt='%.2f')
        plt.colorbar(contour, ax=ax, label=cbar_label, shrink=0.8)

        if plot_type == 'probability' and prob_high_map is not None and prob_high_map.size > 0:
            high_prob_contour_levels = [0.5, 0.75]
            try:
                ax.contour(
                    self.grid_x[:, :, 0], self.grid_y[:, :, 0], prob_high_map,
                    levels=high_prob_contour_levels, colors=['orange', 'red'],
                    linestyles=['--', '-'], linewidths=1.5, extent=extent
                )
            except Exception as contour_err:
                logger.warning(f"[Plotting] Could not add P(High) contour lines: {contour_err}")

        self._add_airport_features(ax, kwargs)
        self._add_bird_positions_with_history(ax, kwargs)

        mean_std_dev_map = np.mean(std_map) if std_map is not None and std_map.size > 0 else np.nan
        ax.text(
            0.02, 0.95, f"Mean Map Uncertainty (StdDev): {mean_std_dev_map:.3f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
        )

        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_title(f"Risk Map Slice z={z_level:.2f} km {title_suffix}", fontsize=12)
        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()

        return ax

    def _add_airport_features(self, ax, kwargs):
        plotted_items_legend = {}

        runways = kwargs.get('runways')
        if runways:
            for i, r in enumerate(runways):
                label = 'Runway' if 'Runway' not in plotted_items_legend else ""
                ax.plot(
                    [r["start"][0], r["end"][0]], [r["start"][1], r["end"][1]],
                    'k-', lw=4, label=label, zorder=3
                )
                plotted_items_legend['Runway'] = True if label else plotted_items_legend.get('Runway', False)

        flight_paths = kwargs.get('flight_paths')
        if flight_paths:
            for i, p in enumerate(flight_paths):
                path_type = p.get('type', 'Path').capitalize()
                is_approach = 'app' in p.get('type', 'Path').lower()
                color = 'b' if 'app' in p.get('type', 'Path').lower() else 'g'
                linestyle = '--' if 'app' in p.get('type', 'Path').lower() else '-'
                label_key = f'{path_type} Path'
                label = label_key if label_key not in plotted_items_legend else ""
                ax.plot(
                    [p["start"][0], p["end"][0]], [p["start"][1], p["end"][1]],
                    c=color, ls=linestyle, lw=1.5, label=label, zorder=3
                )
                plotted_items_legend[label_key] = True if label else plotted_items_legend.get(label_key, False)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8, loc='upper right')

    def _add_bird_positions_with_history(self, ax, kwargs):
        bird_positions = kwargs.get('bird_positions')
        bird_trackers = kwargs.get('bird_trackers')
        actual_positions = kwargs.get('actual_positions')
        plotted_items_legend = {}

        if not bird_positions:
            return

        for i, est_pos in enumerate(bird_positions):
            if est_pos is None:
                continue

            label_est = 'Est. Bird' if 'Est. Bird' not in plotted_items_legend else ""
            ax.scatter(
                est_pos[0], est_pos[1], s=60, c='black', marker='o',
                ec='white', lw=0.5, label=label_est, zorder=5
            )
            plotted_items_legend['Est. Bird'] = True if label_est else plotted_items_legend.get('Est. Bird', False)

            if actual_positions and i < len(actual_positions) and actual_positions[i] is not None:
                act_pos = actual_positions[i]
                label_act = 'Actual Bird' if 'Actual Bird' not in plotted_items_legend else ""
                ax.scatter(
                    act_pos[0], act_pos[1], s=70, c='lime', marker='x',
                    lw=1.5, label=label_act, zorder=4
                )
                plotted_items_legend['Actual Bird'] = True if label_act else plotted_items_legend.get('Actual Bird',False)

                ax.plot([est_pos[0], act_pos[0]], [est_pos[1], act_pos[1]], 'k:', alpha=0.5, zorder=4)

            if bird_trackers and i < len(bird_trackers):
                try:
                    pos_std = bird_trackers[i].get_position_uncertainty()[:2]
                    if np.all(pos_std > 1e-6):
                        from matplotlib.patches import Ellipse
                        ellipse = Ellipse(
                            xy=est_pos[:2], width=2 * pos_std[0] * 2, height=2 * pos_std[1] * 2,
                            edgecolor='darkorange', facecolor='none', linestyle='--',
                            alpha=0.6, lw=1.0, zorder=4
                        )
                        ax.add_patch(ellipse)
                        if 'Uncertainty (95%)' not in plotted_items_legend:
                            ellipse.set_label('Uncertainty (95%)')
                            plotted_items_legend['Uncertainty (95%)'] = True
                except Exception as ellipse_err:
                    logger.debug(f"Could not plot uncertainty ellipse for bird {i}: {ellipse_err}")

            if bird_trackers and i < len(bird_trackers) and hasattr(bird_trackers[i], 'position_history'):
                history = bird_trackers[i].position_history
                if len(history) > 1:
                    hist_x = [pos[0] for pos in history]
                    hist_y = [pos[1] for pos in history]
                    alphas = np.linspace(0.1, 0.8, len(history))
                    label_trail = 'Position History' if 'Position History' not in plotted_items_legend else ""
                    colors = plt.cm.plasma(np.linspace(0, 1, len(history) - 1))

                    for j in range(len(history) - 1):
                        line = ax.plot(
                            [hist_x[j], hist_x[j + 1]], [hist_y[j], hist_y[j + 1]],
                            '-', lw=1.5, alpha=alphas[j], color=colors[j],
                            zorder=3
                        )
                        if j == 0 and label_trail:
                            line[0].set_label(label_trail)
                            plotted_items_legend['Position History'] = True

                    for j in range(len(history) - 1):
                        ax.scatter(
                            hist_x[j], hist_y[j], s=15,
                            color=colors[j], alpha=alphas[j],
                            marker='o', zorder=3
                        )

    def plot_3d_risk_map(self, plot_prob_threshold=0.5, **kwargs):
        logger.info(f"[Plotting] Generating 3D risk map plot (P(Risk>=Med) >= {plot_prob_threshold})...")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        try:
            _, _, prob_high_map, prob_med_or_high_map = self.compute_risk_map(force_recompute=False)
            assert prob_med_or_high_map is not None
            assert prob_high_map is not None
        except Exception as e:
            logger.error(f"[Plotting] Failed to compute risk maps for 3D plot: {e}", exc_info=True)
            ax.text2D(0.5, 0.5, "Error computing 3D risk map data", transform=ax.transAxes, ha='center')
            return fig

        title = f"3D Risk Volume (P(Risk>=Med) >= {plot_prob_threshold})"
        prob_points_indices = np.argwhere(prob_med_or_high_map >= plot_prob_threshold)

        if prob_points_indices.size > 0:

            ix, iy, iz = prob_points_indices.T
            x_coords = self.grid_x[ix, iy, iz]
            y_coords = self.grid_y[ix, iy, iz]
            z_coords = self.grid_z[ix, iy, iz]

            try:
                colors = prob_high_map[ix, iy, iz]
                scatter = ax.scatter(x_coords, y_coords, z_coords, c=colors, cmap='Reds',
                                     marker='.', s=8, alpha=0.6, vmin=0, vmax=1)
                cbar = plt.colorbar(scatter, ax=ax, label="P(Risk>High)", shrink=0.6, pad=0.1)
                cbar.ax.tick_params(labelsize=8)

            except Exception as scatter_err:
                logger.warning(f"[Plotting] Could not apply P(High) coloring to 3D scatter: {scatter_err}. Using default color.")
                ax.scatter(x_coords, y_coords, z_coords, c='red', marker='.', s=8, alpha=0.3)
        else:
            logger.info("[Plotting] No points found exceeding probability threshold for 3D plot.")
            ax.text2D(0.5, 0.5, "No points above threshold", transform=ax.transAxes, ha='center')

        runways = kwargs.get('runways')
        flight_paths = kwargs.get('flight_paths')
        bird_positions = kwargs.get('bird_positions')
        plotted_3d_legend = {}

        if runways:
            for r in runways:
                label = 'Runway' if 'Runway' not in plotted_3d_legend else ""
                ax.plot([r["start"][0], r["end"][0]], [r["start"][1], r["end"][1]],
                        [r["start"][2], r["end"][2]], 'k-', lw=3, label=label, zorder=1)
                plotted_3d_legend['Runway'] = True if label else plotted_3d_legend.get('Runway', False)

        if flight_paths:
            for p in flight_paths:
                path_type = p.get('type', 'Path').capitalize()
                is_approach = 'app' in p.get('type', '').lower()
                color = 'b' if 'app' in p.get('type', '').lower() else 'g'
                label_key = f'{path_type} Path'
                label = label_key if label_key not in plotted_3d_legend else ""
                ax.plot([p["start"][0], p["end"][0]], [p["start"][1], p["end"][1]],
                        [p["start"][2], p["end"][2]], c=color, lw=1.5, label=label, zorder=1)
                plotted_3d_legend[label_key] = True if label else plotted_3d_legend.get(label_key, False)

        if bird_positions:

            plotted_bird_label = False
            for i, p in enumerate(bird_positions):
                if p is not None:
                    label = 'Est. Bird' if not plotted_bird_label else ""
                    ax.scatter(p[0], p[1], p[2], s=50, c='black', marker='o',
                               ec='lime', lw=1, label=label, zorder=10, depthshade=False)
                    plotted_bird_label = True

        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_zlabel("Z (km)")
        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)
        ax.set_zlim(self.z_range)
        ax.set_title(title, fontsize=12)
        ax.view_init(elev=25, azim=-65)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.3)
        logger.info(f"[Plotting] Finished generating 3D risk map plot.")
        return fig

def evaluate_flight_path_risk(risk_mapper, flight_path, num_samples=50):
    path_id = flight_path.get('id', 'N/A')
    path_name = flight_path.get('name', path_id)
    log_prefix = f"[Path Risk Eval Path='{path_id}']"

    start, end = np.array(flight_path["start"]), np.array(flight_path["end"])
    t = np.linspace(0, 1, num_samples)
    path_points = np.array([start + v * (end - start) for v in t])

    if not risk_mapper.initialized:
        logger.warning(f"{log_prefix} Risk mapper not initialized. Returning default low risk.")
        default_std = SequentialGPRiskMapper.GP_TARGET_SCALE / 2.0
        return {
            "path_name": path_name,
            "path_id": path_id,
            "mean_risk_score": 0.,
            "max_risk_score": 0.,
            "risk_categories_by_max_prob": {"LOW": num_samples, "MEDIUM": 0, "HIGH": 0},
            "path_proportions_by_max_prob": {"LOW": 1., "MEDIUM": 0., "HIGH": 0.},
            "mean_std_dev": default_std,
            "avg_prob_high": 0.,
            "avg_prob_medium_or_high": 0.,
            "highest_risk_point": {}
        }

    try:
        mean_risk, std_dev, prob_high, prob_med_or_high = risk_mapper.predict_probabilistic(path_points)
        logger.debug(
            f"{log_prefix} GP Prediction Results (Means): RiskScore={np.mean(mean_risk):.3f}, StdDev={np.mean(std_dev):.3f}, P(>=Med)={np.mean(prob_med_or_high):.3f}, P(>High)={np.mean(prob_high):.3f}")
    except Exception as e:
        logger.error(f"{log_prefix} Failed to get GP predictions for path points: {e}. Returning default low risk.",
                     exc_info=True)
        default_std = SequentialGPRiskMapper.GP_TARGET_SCALE / 2.0
        return {
            "path_name": path_name,
            "path_id": path_id,
            "mean_risk_score": 0.,
            "max_risk_score": 0.,
            "risk_categories_by_max_prob": {"LOW": num_samples, "MEDIUM": 0, "HIGH": 0},
            "path_proportions_by_max_prob": {"LOW": 1., "MEDIUM": 0., "HIGH": 0.},
            "mean_std_dev": default_std,
            "avg_prob_high": 0.,
            "avg_prob_medium_or_high": 0.,
            "highest_risk_point": {}
        }

    prob_med_only = np.clip(prob_med_or_high - prob_high, 0.0, 1.0)
    prob_low = np.clip(1.0 - prob_med_or_high, 0.0, 1.0)
    categories = []

    for ph, pm, pl in zip(prob_high, prob_med_only, prob_low):
        categories.append("HIGH" if ph >= pm and ph >= pl else ("MEDIUM" if pm > ph and pm >= pl else "LOW"))

    category_counts = {level: categories.count(level) for level in ["HIGH", "MEDIUM", "LOW"]}
    proportions = {level: count / num_samples for level, count in category_counts.items()}

    if len(mean_risk) > 0:
        max_risk_idx = np.argmax(mean_risk)
        highest_risk_details = {
            "position": tuple(path_points[max_risk_idx]),
            "mean_risk_score": mean_risk[max_risk_idx],
            "std_dev": std_dev[max_risk_idx],
            "prob_high": prob_high[max_risk_idx],
            "prob_medium_or_high": prob_med_or_high[max_risk_idx]
        }
        logger.debug(f"{log_prefix} Highest risk point (Idx={max_risk_idx}): Score={highest_risk_details['mean_risk_score']:.3f}, P(>=Med)={highest_risk_details['prob_medium_or_high']:.3f}")
    else:
        logger.warning(f"{log_prefix} No risk scores available to determine highest risk point.")
        highest_risk_details = {}

    results = {
        "path_name": path_name,
        "path_id": path_id,
        "mean_risk_score": np.mean(mean_risk) if len(mean_risk) > 0 else 0.,
        "max_risk_score": np.max(mean_risk) if len(mean_risk) > 0 else 0.,
        "risk_categories_by_max_prob": category_counts,
        "path_proportions_by_max_prob": proportions,
        "mean_std_dev": np.mean(std_dev) if len(std_dev) > 0 else SequentialGPRiskMapper.GP_TARGET_SCALE / 2.0,
        "avg_prob_high": np.mean(prob_high) if len(prob_high) > 0 else 0.,
        "avg_prob_medium_or_high": np.mean(prob_med_or_high) if len(prob_med_or_high) > 0 else 0.,
        "highest_risk_point": highest_risk_details
    }
    logger.debug(f"{log_prefix} Evaluation complete. Avg P(>=Med)={results['avg_prob_medium_or_high']:.3f}")
    return results

def plot_risk_difference_maps(risk_mapper, scenario, bird_positions, trackers, current_iter, output_dir, prev_risk_maps=None):

    if current_iter <= 1:
        logger.info(f"Skipping difference map for iteration {current_iter} (need previous data)")
        return {}

    if not prev_risk_maps:
        logger.warning(f"No previous risk maps available for iteration {current_iter}")
        return {}

    try:
        fig, ax = plt.subplots(figsize=(12, 10))

        z_level = 0.5
        risk_mapper.plot_temporal_difference_map(
            previous_risk_map=prev_risk_maps.get('prob_med_or_high'),
            z_level=z_level,
            ax=ax,
            runways=scenario['runways'],
            flight_paths=scenario['flight_paths'],
            bird_positions=bird_positions,
            bird_trackers=trackers
        )

        diff_plot_fn = pathlib.Path(output_dir) / f"risk_diff_map_iter_{current_iter:03d}.png"
        plt.savefig(str(diff_plot_fn))
        plt.close(fig)
        logger.info(f"Saved difference map: {diff_plot_fn}")

        current_maps = {}
        mean_map, std_map, prob_high_map, prob_med_or_high_map = risk_mapper.compute_risk_map(z_level=z_level)
        current_maps = {
            'mean': mean_map.copy() if mean_map is not None else None,
            'std': std_map.copy() if std_map is not None else None,
            'prob_high': prob_high_map.copy() if prob_high_map is not None else None,
            'prob_med_or_high': prob_med_or_high_map.copy() if prob_med_or_high_map is not None else None,
        }

        return current_maps

    except Exception as e:
        logger.error(f"Error generating difference map for iteration {current_iter}: {e}", exc_info=True)
        return {}
