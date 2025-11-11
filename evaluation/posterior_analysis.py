import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import beta, ks_2samp, entropy
from utils.logger import get_logger

logger = get_logger(__name__)


class PosteriorAnalyzer:

    def __init__(self):
        self.beta_history = {}
        self.posterior_samples = {}
        self.convergence_metrics = {}

    def track_beta_evolution(self, species, alpha, beta_param,
                             iteration, prior_alpha=1, prior_beta=9):
        if species not in self.beta_history:
            self.beta_history[species] = {
                'iterations': [],
                'alpha': [],
                'beta': [],
                'mean': [],
                'variance': [],
                'mode': [],
                'prior_alpha': prior_alpha,
                'prior_beta': prior_beta
            }

        self.beta_history[species]['iterations'].append(iteration)
        self.beta_history[species]['alpha'].append(alpha)
        self.beta_history[species]['beta'].append(beta_param)

        mean = alpha / (alpha + beta_param)
        variance = (alpha * beta_param) / ((alpha + beta_param)**2 * (alpha + beta_param + 1))

        if alpha > 1 and beta_param > 1:
            mode = (alpha - 1) / (alpha + beta_param - 2)
        else:
            mode = np.nan

        self.beta_history[species]['mean'].append(mean)
        self.beta_history[species]['variance'].append(variance)
        self.beta_history[species]['mode'].append(mode)

    def posterior_predictive_check(self, observed_data,
                                   alpha, beta_param,
                                   n_simulations=1000):
        n_obs = len(observed_data)

        ppc_samples = []
        for _ in range(n_simulations):
            p = np.random.beta(alpha, beta_param)
            synthetic_data = np.random.binomial(1, p, size=n_obs)
            ppc_samples.append(synthetic_data)

        ppc_samples = np.array(ppc_samples)

        observed_mean = np.mean(observed_data)
        observed_var = np.var(observed_data)
        observed_sum = np.sum(observed_data)

        ppc_means = np.mean(ppc_samples, axis=1)
        ppc_vars = np.var(ppc_samples, axis=1)
        ppc_sums = np.sum(ppc_samples, axis=1)

        p_value_mean = np.mean(np.abs(ppc_means - observed_mean) >= np.abs(observed_mean - np.mean(ppc_means)))
        p_value_var = np.mean(np.abs(ppc_vars - observed_var) >= np.abs(observed_var - np.mean(ppc_vars)))
        p_value_sum = np.mean(np.abs(ppc_sums - observed_sum) >= np.abs(observed_sum - np.mean(ppc_sums)))

        return {
            'observed_mean': observed_mean,
            'observed_var': observed_var,
            'observed_sum': observed_sum,
            'ppc_means': ppc_means,
            'ppc_vars': ppc_vars,
            'ppc_sums': ppc_sums,
            'p_value_mean': p_value_mean,
            'p_value_var': p_value_var,
            'p_value_sum': p_value_sum,
            'ppc_samples': ppc_samples
        }

    def convergence_diagnostics(self, samples, warmup=100):
        if len(samples) <= warmup:
            logger.warning("Not enough samples for convergence diagnostics")
            return {}

        samples_post_warmup = samples[warmup:]

        n_samples = len(samples_post_warmup)
        first_portion = samples_post_warmup[:int(n_samples * 0.1)]
        last_portion = samples_post_warmup[int(n_samples * 0.5):]

        geweke_z = (np.mean(first_portion) - np.mean(last_portion)) / \
                   np.sqrt(np.var(first_portion) / len(first_portion) + \
                          np.var(last_portion) / len(last_portion))

        def autocorrelation(x, lag):
            if lag >= len(x):
                return 0
            x_centered = x - np.mean(x)
            c0 = np.dot(x_centered, x_centered) / len(x)
            ct = np.dot(x_centered[:-lag], x_centered[lag:]) / (len(x) - lag)
            return ct / c0 if c0 > 0 else 0

        max_lag = min(len(samples_post_warmup) // 4, 100)
        autocorr_values = []
        for lag in range(1, max_lag):
            ac = autocorrelation(samples_post_warmup, lag)
            autocorr_values.append(ac)
            if ac < 0:
                break

        if autocorr_values:
            sum_autocorr = 1 + 2 * sum(autocorr_values)
            ess = len(samples_post_warmup) / max(sum_autocorr, 1)
        else:
            ess = len(samples_post_warmup)

        mcse = np.std(samples_post_warmup) / np.sqrt(ess)

        return {
            'geweke_z': geweke_z,
            'geweke_p_value': 2 * (1 - stats.norm.cdf(abs(geweke_z))),
            'effective_sample_size': ess,
            'mcse': mcse,
            'n_samples': len(samples_post_warmup),
            'autocorr_values': autocorr_values[:10] if autocorr_values else []
        }

    def print_beta_evolution(self, species):
        if species not in self.beta_history:
            logger.warning(f"No history for species: {species}")
            return

        history = self.beta_history[species]

        print("\n" + "="*80)
        print(f"BETA POSTERIOR EVOLUTION ANALYSIS - {species}")
        print("="*80)

        print(f"\n1. PARAMETER EVOLUTION")
        print("-"*80)
        print(f"Prior: Beta(α={history['prior_alpha']:.1f}, β={history['prior_beta']:.1f})")
        print(f"Total iterations tracked: {len(history['iterations'])}")
        print(f"\nFinal parameters: Beta(α={history['alpha'][-1]:.2f}, β={history['beta'][-1]:.2f})")
        print(f"Initial parameters: Beta(α={history['alpha'][0]:.2f}, β={history['beta'][0]:.2f})")
        print(f"Parameter change: Δα={history['alpha'][-1]-history['alpha'][0]:.2f}, Δβ={history['beta'][-1]-history['beta'][0]:.2f}")

        print(f"\n2. POSTERIOR MEAN EVOLUTION")
        print("-"*80)
        print(f"{'Iteration':<12} {'Mean':<12} {'95% CI Lower':<15} {'95% CI Upper':<15}")
        print("-"*80)

        n_iters = len(history['iterations'])
        display_iters = np.linspace(0, n_iters-1, min(10, n_iters), dtype=int)

        for idx in display_iters:
            iter_num = history['iterations'][idx]
            mean = history['mean'][idx]
            a, b = history['alpha'][idx], history['beta'][idx]
            lower = beta.ppf(0.025, a, b)
            upper = beta.ppf(0.975, a, b)
            print(f"{iter_num:<12} {mean:<12.4f} {lower:<15.4f} {upper:<15.4f}")

        print(f"\n3. VARIANCE SHRINKAGE")
        print("-"*80)
        prior_var = (history['prior_alpha'] * history['prior_beta']) / \
                    ((history['prior_alpha'] + history['prior_beta'])**2 * \
                     (history['prior_alpha'] + history['prior_beta'] + 1))
        print(f"Prior variance:      {prior_var:.6f}")
        print(f"Initial variance:    {history['variance'][0]:.6f}")
        print(f"Final variance:      {history['variance'][-1]:.6f}")
        print(f"Variance reduction:  {(1 - history['variance'][-1]/prior_var)*100:.2f}%")

        print(f"\n4. DISTRIBUTION SHAPE AT KEY ITERATIONS")
        print("-"*80)
        selected_iters = np.linspace(0, n_iters-1, min(5, n_iters), dtype=int)
        print(f"{'Iteration':<12} {'α':<10} {'β':<10} {'Mean':<10} {'Mode':<10}")
        print("-"*80)

        for idx in selected_iters:
            iter_num = history['iterations'][idx]
            a = history['alpha'][idx]
            b = history['beta'][idx]
            mean = history['mean'][idx]
            mode = history['mode'][idx] if not np.isnan(history['mode'][idx]) else None
            mode_str = f"{mode:.4f}" if mode is not None else "N/A"
            print(f"{iter_num:<12} {a:<10.2f} {b:<10.2f} {mean:<10.4f} {mode_str:<10}")

        print(f"\n5. INFORMATION GAIN (KL Divergence from Prior)")
        print("-"*80)
        kl_divergences = []
        prior_alpha = history['prior_alpha']
        prior_beta = history['prior_beta']

        for a, b in zip(history['alpha'], history['beta']):
            kl = self._beta_kl_divergence(a, b, prior_alpha, prior_beta)
            kl_divergences.append(kl)

        print(f"Initial KL divergence:  {kl_divergences[0]:.6f} nats")
        print(f"Final KL divergence:    {kl_divergences[-1]:.6f} nats")
        print(f"Average KL divergence:  {np.mean(kl_divergences):.6f} nats")
        print(f"Total information gain: {kl_divergences[-1]:.6f} nats")

        print(f"\n6. EFFECTIVE SAMPLE SIZE")
        print("-"*80)
        variance_ratio = np.array(history['variance']) / prior_var
        effective_n = 1 / variance_ratio - 1

        print(f"Initial effective N:    {effective_n[0]:.2f}")
        print(f"Final effective N:      {effective_n[-1]:.2f}")
        print(f"Effective data growth:  {effective_n[-1] - effective_n[0]:.2f}")

        print("\n" + "="*80)

    def _beta_kl_divergence(self, a1, b1, a2, b2):
        from scipy.special import betaln, digamma

        kl = betaln(a2, b2) - betaln(a1, b1)
        kl += (a1 - a2) * digamma(a1)
        kl += (b1 - b2) * digamma(b1)
        kl += (a2 - a1 + b2 - b1) * digamma(a1 + b1)

        return kl

    def print_posterior_predictive_check(self, ppc_results):
        print("\n" + "="*80)
        print("POSTERIOR PREDICTIVE CHECK RESULTS")
        print("="*80)

        print(f"\n1. TEST STATISTICS SUMMARY")
        print("-"*80)
        print(f"Observed mean:       {ppc_results['observed_mean']:.6f}")
        print(f"Observed variance:   {ppc_results['observed_var']:.6f}")
        print(f"Observed sum:        {ppc_results['observed_sum']}")

        print(f"\n2. PPC DISTRIBUTIONS")
        print("-"*80)
        print(f"PPC mean range:      [{np.min(ppc_results['ppc_means']):.6f}, {np.max(ppc_results['ppc_means']):.6f}]")
        print(f"PPC mean avg:        {np.mean(ppc_results['ppc_means']):.6f}")
        print(f"PPC mean std:        {np.std(ppc_results['ppc_means']):.6f}")

        print(f"\nPPC variance range:  [{np.min(ppc_results['ppc_vars']):.6f}, {np.max(ppc_results['ppc_vars']):.6f}]")
        print(f"PPC variance avg:    {np.mean(ppc_results['ppc_vars']):.6f}")
        print(f"PPC variance std:    {np.std(ppc_results['ppc_vars']):.6f}")

        print(f"\nPPC sum range:       [{np.min(ppc_results['ppc_sums'])}, {np.max(ppc_results['ppc_sums'])}]")
        print(f"PPC sum avg:         {np.mean(ppc_results['ppc_sums']):.2f}")
        print(f"PPC sum std:         {np.std(ppc_results['ppc_sums']):.2f}")

        print(f"\n3. P-VALUES (Model Adequacy)")
        print("-"*80)
        print(f"Mean statistic p-value:      {ppc_results['p_value_mean']:.4f}")
        print(f"Variance statistic p-value:  {ppc_results['p_value_var']:.4f}")
        print(f"Sum statistic p-value:       {ppc_results['p_value_sum']:.4f}")

        print(f"\n4. INTERPRETATION")
        print("-"*80)
        alpha = 0.05

        mean_pass = "PASS" if ppc_results['p_value_mean'] > alpha else "FAIL"
        var_pass = "PASS" if ppc_results['p_value_var'] > alpha else "FAIL"
        sum_pass = "PASS" if ppc_results['p_value_sum'] > alpha else "FAIL"

        print(f"Mean test (α={alpha}):       {mean_pass}")
        print(f"Variance test (α={alpha}):   {var_pass}")
        print(f"Sum test (α={alpha}):        {sum_pass}")

        if all([ppc_results['p_value_mean'] > alpha,
                ppc_results['p_value_var'] > alpha,
                ppc_results['p_value_sum'] > alpha]):
            print(f"\nOverall: Model shows good fit (all p-values > {alpha})")
        else:
            print(f"\nOverall: Model shows poor fit (some p-values ≤ {alpha})")

        print("\n" + "="*80)

    def print_credible_regions(self, alpha, beta_param,
                               levels=None):
        if levels is None:
            levels = [0.5, 0.9, 0.95]
        print("\n" + "="*80)
        print(f"BETA CREDIBLE REGIONS - Beta(α={alpha:.2f}, β={beta_param:.2f})")
        print("="*80)

        mean = alpha / (alpha + beta_param)
        variance = (alpha * beta_param) / ((alpha + beta_param)**2 * (alpha + beta_param + 1))

        if alpha > 1 and beta_param > 1:
            mode = (alpha - 1) / (alpha + beta_param - 2)
            mode_str = f"{mode:.6f}"
        else:
            mode_str = "N/A (α≤1 or β≤1)"

        print(f"\n1. DISTRIBUTION SUMMARY")
        print("-"*80)
        print(f"Mean:              {mean:.6f}")
        print(f"Mode:              {mode_str}")
        print(f"Variance:          {variance:.6f}")
        print(f"Std deviation:     {np.sqrt(variance):.6f}")

        print(f"\n2. EQUAL-TAILED CREDIBLE INTERVALS")
        print("-"*80)
        print(f"{'Level':<10} {'Lower Bound':<15} {'Upper Bound':<15} {'Width':<12}")
        print("-"*80)

        for level in levels:
            lower = beta.ppf((1-level)/2, alpha, beta_param)
            upper = beta.ppf((1+level)/2, alpha, beta_param)
            width = upper - lower
            print(f"{level*100:<10.0f}%   {lower:<15.6f} {upper:<15.6f} {width:<12.6f}")

        print(f"\n3. HIGHEST POSTERIOR DENSITY (HPD) INTERVALS")
        print("-"*80)
        samples = np.random.beta(alpha, beta_param, size=10000)
        print(f"{'Level':<10} {'HPD Lower':<15} {'HPD Upper':<15} {'Width':<12}")
        print("-"*80)

        for level in levels:
            hpd_lower, hpd_upper = self._compute_hpd(samples, level)
            width = hpd_upper - hpd_lower
            print(f"{level*100:<10.0f}%   {hpd_lower:<15.6f} {hpd_upper:<15.6f} {width:<12.6f}")

        print(f"\n4. QUANTILES")
        print("-"*80)
        quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        print(f"{'Quantile':<12} {'Value':<12}")
        print("-"*80)
        for q in quantiles:
            val = beta.ppf(q, alpha, beta_param)
            print(f"{q:<12.2f} {val:<12.6f}")

        print("\n" + "="*80)

    def _compute_hpd(self, samples, level):
        sorted_samples = np.sort(samples)
        n = len(sorted_samples)
        interval_size = int(n * level)

        # Find the shortest interval containing level% of samples
        best_width = np.inf
        best_lower = 0
        best_upper = 0

        for i in range(n - interval_size):
            lower = sorted_samples[i]
            upper = sorted_samples[i + interval_size]
            width = upper - lower

            if width < best_width:
                best_width = width
                best_lower = lower
                best_upper = upper

        return best_lower, best_upper

    def analyze_information_gain(self, species_list):
        results = []

        for species in species_list:
            if species not in self.beta_history:
                continue

            history = self.beta_history[species]

            # Final parameters
            final_alpha = history['alpha'][-1] if history['alpha'] else 1
            final_beta = history['beta'][-1] if history['beta'] else 9

            # KL divergence from prior
            kl_div = self._beta_kl_divergence(
                final_alpha, final_beta,
                history['prior_alpha'], history['prior_beta']
            )

            # Variance reduction
            prior_var = (history['prior_alpha'] * history['prior_beta']) / \
                       ((history['prior_alpha'] + history['prior_beta'])**2 * \
                        (history['prior_alpha'] + history['prior_beta'] + 1))
            final_var = history['variance'][-1] if history['variance'] else prior_var
            var_reduction = 1 - (final_var / prior_var)

            # Effective sample size
            ess = (final_alpha + final_beta) - (history['prior_alpha'] + history['prior_beta'])

            results.append({
                'species': species,
                'kl_divergence': kl_div,
                'variance_reduction': var_reduction,
                'effective_sample_size': ess,
                'final_mean': history['mean'][-1] if history['mean'] else 0.1,
                'final_variance': final_var,
                'n_updates': len(history['iterations'])
            })

        return pd.DataFrame(results).sort_values('kl_divergence', ascending=False)
