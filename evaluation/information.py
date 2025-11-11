import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
from scipy.stats import entropy, beta
from utils.logger import get_logger

logger = get_logger(__name__)


class InformationMetrics:

    def __init__(self):
        self.log_likelihoods = []
        self.posterior_samples = []

    def waic(self, log_likelihood_samples):
        lppd = np.sum(logsumexp(log_likelihood_samples, axis=0) - np.log(log_likelihood_samples.shape[0]))

        p_waic_1 = np.sum(np.var(log_likelihood_samples, axis=0))

        mean_ll = np.mean(log_likelihood_samples, axis=0)
        p_waic_2 = 2 * np.sum(mean_ll - logsumexp(log_likelihood_samples, axis=0) + np.log(log_likelihood_samples.shape[0]))

        waic_1 = -2 * (lppd - p_waic_1)
        waic_2 = -2 * (lppd - p_waic_2)

        pointwise_waic = -2 * (logsumexp(log_likelihood_samples, axis=0) - np.log(log_likelihood_samples.shape[0]) -
                               np.var(log_likelihood_samples, axis=0))
        se_waic = np.sqrt(log_likelihood_samples.shape[1] * np.var(pointwise_waic))

        return {
            'waic': waic_1,
            'waic_alt': waic_2,
            'lppd': lppd,
            'p_waic': p_waic_1,
            'p_waic_alt': p_waic_2,
            'se_waic': se_waic,
            'pointwise_waic': pointwise_waic
        }

    def loo_cv(self, log_likelihood_samples):
        n_samples, n_obs = log_likelihood_samples.shape

        loo_log_likes = []
        pareto_k_values = []

        for i in range(n_obs):
            log_weights = log_likelihood_samples[:, i]

            k, smoothed_weights = self._psis_smooth(log_weights)
            pareto_k_values.append(k)

            loo_lpd = logsumexp(log_weights, b=np.exp(smoothed_weights - np.max(smoothed_weights)))
            loo_log_likes.append(loo_lpd)

        elpd_loo = np.sum(loo_log_likes)
        loo = -2 * elpd_loo

        p_loo = np.sum([logsumexp(log_likelihood_samples[:, i], axis=0) - np.log(n_samples) - loo_log_likes[i]
                        for i in range(n_obs)])

        k_warnings = {
            'good': np.sum(np.array(pareto_k_values) < 0.5),
            'ok': np.sum((np.array(pareto_k_values) >= 0.5) & (np.array(pareto_k_values) < 0.7)),
            'bad': np.sum((np.array(pareto_k_values) >= 0.7) & (np.array(pareto_k_values) < 1)),
            'very_bad': np.sum(np.array(pareto_k_values) >= 1)
        }

        return {
            'loo': loo,
            'elpd_loo': elpd_loo,
            'p_loo': p_loo,
            'pareto_k': pareto_k_values,
            'k_diagnostics': k_warnings,
            'mean_pareto_k': np.mean(pareto_k_values)
        }

    def _psis_smooth(self, log_weights):
        log_weights = log_weights - np.max(log_weights)
        weights = np.exp(log_weights)

        sorted_weights = np.sort(weights)[::-1]

        n_tail = max(int(0.2 * len(weights)), 5)
        tail_weights = sorted_weights[:n_tail]

        if len(tail_weights) > 1 and np.min(tail_weights) > 0:
            k = self._estimate_pareto_k(tail_weights)

            if k > 0.7:
                smoothed = self._smooth_tail(sorted_weights, k)
                return k, np.log(smoothed)
        else:
            k = 0

        return k, log_weights

    def _estimate_pareto_k(self, tail_weights):
        n = len(tail_weights)
        min_weight = np.min(tail_weights)

        if n <= 1 or min_weight <= 1e-10:
            return 0.0

        if np.any(tail_weights <= 0):
            return 0.0

        try:
            log_ratios = np.log(tail_weights / min_weight)
            mean_log_ratio = np.mean(log_ratios)

            if mean_log_ratio < 1e-10:
                return 0.0

            k = 1.0 / mean_log_ratio

            k = np.clip(k, -10.0, 10.0)
        except (ValueError, RuntimeWarning, FloatingPointError):
            k = 0.0

        return k

    def _smooth_tail(self, weights, k):
        n = len(weights)
        n_tail = max(int(0.2 * n), 5)

        smoothed = weights.copy()

        if k > 0:
            tail_cutoff = weights[n_tail]
            for i in range(n_tail):
                if weights[i] > tail_cutoff:
                    # Apply smoothing formula
                    smoothed[i] = tail_cutoff * ((i + 1) / (n_tail + 1)) ** (-1/k)

        return smoothed

    def mutual_information(self, x, y, n_bins=10):
        x_bins = np.histogram_bin_edges(x, bins=n_bins)
        y_bins = np.histogram_bin_edges(y, bins=n_bins)

        x_discrete = np.digitize(x, x_bins)
        y_discrete = np.digitize(y, y_bins)

        joint_hist = np.histogram2d(x_discrete, y_discrete, bins=[n_bins, n_bins])[0]

        joint_prob = joint_hist / np.sum(joint_hist)

        x_prob = np.sum(joint_prob, axis=1)
        y_prob = np.sum(joint_prob, axis=0)

        h_x = entropy(x_prob[x_prob > 0])
        h_y = entropy(y_prob[y_prob > 0])

        joint_prob_flat = joint_prob.flatten()
        h_xy = entropy(joint_prob_flat[joint_prob_flat > 0])

        mi = h_x + h_y - h_xy

        return mi

    def information_gain_decomposition(self, predictions,
                                       features):
        results = []

        h_pred = entropy(np.histogram(predictions, bins=20)[0] + 1e-10)

        for feature_name, feature_values in features.items():
            mi = self.mutual_information(predictions, feature_values)

            nmi = mi / h_pred if h_pred > 0 else 0

            h_feature = entropy(np.histogram(feature_values, bins=20)[0] + 1e-10)
            gain_ratio = mi / h_feature if h_feature > 0 else 0

            results.append({
                'feature': feature_name,
                'mutual_information': mi,
                'normalized_mi': nmi,
                'gain_ratio': gain_ratio,
                'feature_entropy': h_feature
            })

        return pd.DataFrame(results).sort_values('mutual_information', ascending=False)

    def expected_information_gain(self, current_posterior_alpha,
                                  current_posterior_beta,
                                  n_future_observations=10,
                                  n_simulations=1000):
        current_kl = 0

        future_kls = []
        future_variances = []

        for _ in range(n_simulations):
            p = np.random.beta(current_posterior_alpha, current_posterior_beta)

            future_successes = np.random.binomial(n_future_observations, p)
            future_failures = n_future_observations - future_successes

            future_alpha = current_posterior_alpha + future_successes
            future_beta = current_posterior_beta + future_failures

            kl = self._beta_kl_divergence(future_alpha, future_beta,
                                         current_posterior_alpha, current_posterior_beta)
            future_kls.append(kl)

            future_var = (future_alpha * future_beta) / \
                        ((future_alpha + future_beta)**2 * (future_alpha + future_beta + 1))
            future_variances.append(future_var)

        current_var = (current_posterior_alpha * current_posterior_beta) / \
                     ((current_posterior_alpha + current_posterior_beta)**2 * \
                      (current_posterior_alpha + current_posterior_beta + 1))

        return {
            'expected_kl_divergence': np.mean(future_kls),
            'std_kl_divergence': np.std(future_kls),
            'expected_variance_reduction': current_var - np.mean(future_variances),
            'probability_variance_decrease': np.mean(np.array(future_variances) < current_var),
            'n_future_observations': n_future_observations
        }

    def _beta_kl_divergence(self, a1, b1, a2, b2):
        from scipy.special import betaln, digamma

        kl = betaln(a2, b2) - betaln(a1, b1)
        kl += (a1 - a2) * digamma(a1)
        kl += (b1 - b2) * digamma(b1)
        kl += (a2 - a1 + b2 - b1) * digamma(a1 + b1)

        return kl

    def sensitivity_analysis(self, prior_params,
                             observations):
        successes, failures = observations
        results = []

        for prior_alpha, prior_beta in prior_params:
            post_alpha = prior_alpha + successes
            post_beta = prior_beta + failures

            prior_mean = prior_alpha / (prior_alpha + prior_beta)
            post_mean = post_alpha / (post_alpha + post_beta)

            prior_var = (prior_alpha * prior_beta) / \
                       ((prior_alpha + prior_beta)**2 * (prior_alpha + prior_beta + 1))
            post_var = (post_alpha * post_beta) / \
                      ((post_alpha + post_beta)**2 * (post_alpha + post_beta + 1))

            uniform_marginal = 1
            current_marginal = np.exp(self._beta_log_marginal_likelihood(
                successes, failures, prior_alpha, prior_beta))
            bayes_factor = current_marginal / uniform_marginal

            # Information gain
            kl_gain = self._beta_kl_divergence(post_alpha, post_beta, prior_alpha, prior_beta)

            results.append({
                'prior_alpha': prior_alpha,
                'prior_beta': prior_beta,
                'prior_mean': prior_mean,
                'posterior_mean': post_mean,
                'mean_shift': post_mean - prior_mean,
                'variance_reduction': 1 - (post_var / prior_var),
                'bayes_factor': bayes_factor,
                'information_gain': kl_gain,
                'effective_sample_size': prior_alpha + prior_beta
            })

        return pd.DataFrame(results)

    def _beta_log_marginal_likelihood(self, successes, failures,
                                      prior_alpha, prior_beta):
        from scipy.special import betaln

        log_ml = betaln(successes + prior_alpha, failures + prior_beta)
        log_ml -= betaln(prior_alpha, prior_beta)

        return log_ml

    def print_information_analysis(self, info_df, top_n=10):
        print("\n" + "="*80)
        print("INFORMATION-THEORETIC ANALYSIS")
        print("="*80)

        print(f"\n1. INFORMATION GAIN BY FEATURE (Top {top_n})")
        print("-"*80)
        print(f"{'Rank':<6} {'Feature':<30} {'MI (nats)':<15} {'Normalized MI':<15}")
        print("-"*80)

        for idx, row in info_df.head(top_n).iterrows():
            rank = idx + 1 if isinstance(idx, int) else idx
            print(f"{rank:<6} {row['feature']:<30} {row['mutual_information']:<15.6f} {row['normalized_mi']:<15.6f}")

        print(f"\n2. FEATURE ENTROPY AND COMPLEXITY")
        print("-"*80)
        print(f"{'Feature':<30} {'Entropy (nats)':<18} {'MI (nats)':<15} {'Gain Ratio':<12}")
        print("-"*80)

        for idx, row in info_df.head(top_n).iterrows():
            print(f"{row['feature']:<30} {row['feature_entropy']:<18.6f} {row['mutual_information']:<15.6f} {row['gain_ratio']:<12.6f}")

        print(f"\n3. SUMMARY STATISTICS")
        print("-"*80)
        print(f"Total features analyzed:        {len(info_df)}")
        print(f"Average mutual information:     {info_df['mutual_information'].mean():.6f} nats")
        print(f"Total information (sum):        {info_df['mutual_information'].sum():.6f} nats")
        print(f"Max mutual information:         {info_df['mutual_information'].max():.6f} nats")
        print(f"Min mutual information:         {info_df['mutual_information'].min():.6f} nats")
        print(f"Average normalized MI:          {info_df['normalized_mi'].mean():.6f}")
        print(f"Average gain ratio:             {info_df['gain_ratio'].mean():.6f}")

        print("\n" + "="*80)

    def print_model_comparison(self, models_results):
        print("\n" + "="*80)
        print("MODEL COMPARISON - INFORMATION CRITERIA")
        print("="*80)

        model_names = list(models_results.keys())

        waic_values = [models_results[m].get('waic', np.nan) for m in model_names]
        loo_values = [models_results[m].get('loo', np.nan) for m in model_names]
        p_waic = [models_results[m].get('p_waic', 0) for m in model_names]
        p_loo = [models_results[m].get('p_loo', 0) for m in model_names]

        print(f"\n1. WAIC COMPARISON (Lower is better)")
        print("-"*80)
        print(f"{'Model':<25} {'WAIC':<15} {'SE':<15} {'p_WAIC':<15}")
        print("-"*80)

        for i, m in enumerate(model_names):
            se_waic = models_results[m].get('se_waic', np.nan)
            waic_str = f"{waic_values[i]:.4f}" if not np.isnan(waic_values[i]) else "N/A"
            se_str = f"{se_waic:.4f}" if not np.isnan(se_waic) else "N/A"
            p_waic_str = f"{p_waic[i]:.4f}"
            print(f"{m:<25} {waic_str:<15} {se_str:<15} {p_waic_str:<15}")

        best_waic_idx = np.nanargmin(waic_values)
        print(f"\nBest model (WAIC): {model_names[best_waic_idx]} ({waic_values[best_waic_idx]:.4f})")

        print(f"\n2. LOO-CV COMPARISON (Lower is better)")
        print("-"*80)
        print(f"{'Model':<25} {'LOO':<15} {'SE':<15} {'p_LOO':<15}")
        print("-"*80)

        for i, m in enumerate(model_names):
            se_loo = models_results[m].get('se_loo', np.nan)
            loo_str = f"{loo_values[i]:.4f}" if not np.isnan(loo_values[i]) else "N/A"
            se_str = f"{se_loo:.4f}" if not np.isnan(se_loo) else "N/A"
            p_loo_str = f"{p_loo[i]:.4f}"
            print(f"{m:<25} {loo_str:<15} {se_str:<15} {p_loo_str:<15}")

        best_loo_idx = np.nanargmin(loo_values)
        print(f"\nBest model (LOO): {model_names[best_loo_idx]} ({loo_values[best_loo_idx]:.4f})")

        print(f"\n3. MODEL COMPLEXITY (Effective Parameters)")
        print("-"*80)
        print(f"{'Model':<25} {'p_WAIC':<15} {'p_LOO':<15} {'Avg p_eff':<15}")
        print("-"*80)

        for i, m in enumerate(model_names):
            avg_p_eff = (p_waic[i] + p_loo[i]) / 2
            print(f"{m:<25} {p_waic[i]:<15.4f} {p_loo[i]:<15.4f} {avg_p_eff:<15.4f}")

        print(f"\n4. SUMMARY")
        print("-"*80)
        print(f"Total models compared:          {len(model_names)}")
        print(f"WAIC range:                     [{np.nanmin(waic_values):.4f}, {np.nanmax(waic_values):.4f}]")
        print(f"LOO range:                      [{np.nanmin(loo_values):.4f}, {np.nanmax(loo_values):.4f}]")
        print(f"Average effective parameters:   {np.mean(p_waic):.4f} (WAIC), {np.mean(p_loo):.4f} (LOO)")

        waic_diff = np.nanmax(waic_values) - np.nanmin(waic_values)
        loo_diff = np.nanmax(loo_values) - np.nanmin(loo_values)
        print(f"WAIC difference (max-min):      {waic_diff:.4f}")
        print(f"LOO difference (max-min):       {loo_diff:.4f}")

        print("\n" + "="*80)
