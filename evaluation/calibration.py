import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_curve, auc, log_loss
from scipy import stats
from netcal.metrics import ECE as NetCalECE
from netcal.metrics import MCE as NetCalMCE
from utils.logger import get_logger

logger = get_logger(__name__)

class BayesianCalibration:

    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.calibration_results = {}

    def expected_calibration_error(self, y_prob, y_true, n_bins=None):
        if n_bins is None:
            n_bins = self.n_bins

        ece_calculator = NetCalECE(bins=n_bins)
        ece = ece_calculator.measure(y_prob.reshape(-1, 1), y_true.reshape(-1, 1))
        return float(ece)

    def maximum_calibration_error(self, y_prob, y_true, n_bins=None):
        if n_bins is None:
            n_bins = self.n_bins

        mce_calculator = NetCalMCE(bins=n_bins)
        mce = mce_calculator.measure(y_prob.reshape(-1, 1), y_true.reshape(-1, 1))
        return float(mce)

    def adaptive_calibration_error(self, y_prob, y_true, p=2):
        sorted_indices = np.argsort(y_prob)
        y_prob_sorted = y_prob[sorted_indices]
        y_true_sorted = y_true[sorted_indices]

        n = len(y_prob)
        bin_size = n // self.n_bins

        ace = 0.0
        for i in range(self.n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < self.n_bins - 1 else n

            if end_idx > start_idx:
                bin_prob = y_prob_sorted[start_idx:end_idx]
                bin_true = y_true_sorted[start_idx:end_idx]

                avg_pred = np.mean(bin_prob)
                avg_true = np.mean(bin_true)
                prop_in_bin = (end_idx - start_idx) / n

                if p == 1:
                    ace += np.abs(avg_pred - avg_true) * prop_in_bin
                else:
                    ace += (np.abs(avg_pred - avg_true) ** p * prop_in_bin) ** (1/p)

        return ace

    def brier_score_decomposition(self, y_prob, y_true):
        bs = brier_score_loss(y_true, y_prob)

        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)

        reliability = 0.0
        resolution = 0.0

        base_rate = np.mean(y_true)
        uncertainty = base_rate * (1 - base_rate)

        for i in range(self.n_bins):
            in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
            n_in_bin = np.sum(in_bin)

            if n_in_bin > 0:
                pred_in_bin = np.mean(y_prob[in_bin])
                obs_in_bin = np.mean(y_true[in_bin])
                prop_in_bin = n_in_bin / len(y_prob)

                reliability += prop_in_bin * (pred_in_bin - obs_in_bin) ** 2

                resolution += prop_in_bin * (obs_in_bin - base_rate) ** 2

        return {
            'brier_score': bs,
            'reliability': reliability,
            'resolution': resolution,
            'uncertainty': uncertainty,
            'reliability_resolution_diff': reliability - resolution
        }

    def calibration_curve_with_confidence(self, y_prob, y_true, n_bins=None, n_bootstrap=100, confidence_level=0.95):
        if n_bins is None:
            n_bins = self.n_bins

        fraction_pos, mean_pred_prob = calibration_curve(y_true, y_prob, n_bins=n_bins)

        n_samples = len(y_prob)
        bootstrap_fractions = []
        bootstrap_means = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_prob_boot = y_prob[indices]
            y_true_boot = y_true[indices]

            try:
                frac_boot, mean_boot = calibration_curve(y_true_boot, y_prob_boot, n_bins=n_bins)
                bootstrap_fractions.append(frac_boot)
                bootstrap_means.append(mean_boot)
            except:
                continue

        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        if bootstrap_fractions:
            bootstrap_fractions = np.array(bootstrap_fractions)
            lower_bound = np.percentile(bootstrap_fractions, lower_percentile, axis=0)
            upper_bound = np.percentile(bootstrap_fractions, upper_percentile, axis=0)
        else:
            lower_bound = fraction_pos
            upper_bound = fraction_pos

        return {
            'fraction_positive': fraction_pos,
            'mean_predicted_prob': mean_pred_prob,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level
        }

    def isotonic_recalibration(self, y_prob, y_true):
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(y_prob, y_true)
        y_prob_calibrated = iso_reg.transform(y_prob)

        return y_prob_calibrated, iso_reg

    def roc_analysis_with_confidence(self, y_prob, y_true, n_bootstrap=100):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        n_samples = len(y_prob)
        auc_scores = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_prob_boot = y_prob[indices]
            y_true_boot = y_true[indices]

            try:
                fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_prob_boot)
                auc_boot = auc(fpr_boot, tpr_boot)
                auc_scores.append(auc_boot)
            except:
                continue

        auc_mean = np.mean(auc_scores) if auc_scores else roc_auc
        auc_std = np.std(auc_scores) if auc_scores else 0
        auc_ci = np.percentile(auc_scores, [2.5, 97.5]) if auc_scores else [roc_auc, roc_auc]

        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]

        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc,
            'auc_mean': auc_mean,
            'auc_std': auc_std,
            'auc_ci': auc_ci,
            'optimal_threshold': optimal_threshold,
            'optimal_tpr': tpr[optimal_idx],
            'optimal_fpr': fpr[optimal_idx]
        }

    def print_calibration_suite(self, y_prob, y_true, title_prefix=""):
        print("\n" + "="*80)
        print(f"{title_prefix}CALIBRATION ANALYSIS SUITE")
        print("="*80)

        calib_data = self.calibration_curve_with_confidence(y_prob, y_true)
        ece = self.expected_calibration_error(y_prob, y_true)
        mce = self.maximum_calibration_error(y_prob, y_true)

        print(f"\n1. RELIABILITY DIAGRAM")
        print("-"*80)
        print(f"Expected Calibration Error (ECE): {ece:.6f}")
        print(f"Maximum Calibration Error (MCE): {mce:.6f}")
        print(f"\nCalibration Curve Data ({calib_data['confidence_level']:.0%} confidence intervals):")
        print(f"{'Mean Predicted':<20} {'Fraction Positive':<20} {'Lower Bound':<15} {'Upper Bound':<15}")
        print("-"*70)
        for i in range(len(calib_data['mean_predicted_prob'])):
            print(f"{calib_data['mean_predicted_prob'][i]:<20.6f} "
                  f"{calib_data['fraction_positive'][i]:<20.6f} "
                  f"{calib_data['lower_bound'][i]:<15.6f} "
                  f"{calib_data['upper_bound'][i]:<15.6f}")

        print(f"\n2. PREDICTION DISTRIBUTION BY CLASS")
        print("-"*80)
        neg_probs = y_prob[y_true == 0]
        pos_probs = y_prob[y_true == 1]
        print(f"Negative class (n={len(neg_probs)}):")
        print(f"  Mean: {np.mean(neg_probs):.6f}, Std: {np.std(neg_probs):.6f}")
        print(f"  Quantiles: 25%={np.percentile(neg_probs, 25):.6f}, "
              f"50%={np.percentile(neg_probs, 50):.6f}, "
              f"75%={np.percentile(neg_probs, 75):.6f}")
        print(f"\nPositive class (n={len(pos_probs)}):")
        print(f"  Mean: {np.mean(pos_probs):.6f}, Std: {np.std(pos_probs):.6f}")
        print(f"  Quantiles: 25%={np.percentile(pos_probs, 25):.6f}, "
              f"50%={np.percentile(pos_probs, 50):.6f}, "
              f"75%={np.percentile(pos_probs, 75):.6f}")

        print(f"\n3. ROC CURVE ANALYSIS")
        print("-"*80)
        roc_data = self.roc_analysis_with_confidence(y_prob, y_true)
        print(f"Area Under Curve (AUC): {roc_data['auc']:.6f}")
        print(f"AUC Mean (bootstrap): {roc_data['auc_mean']:.6f}")
        print(f"AUC Std (bootstrap): {roc_data['auc_std']:.6f}")
        print(f"AUC 95% CI: [{roc_data['auc_ci'][0]:.6f}, {roc_data['auc_ci'][1]:.6f}]")
        print(f"\nOptimal Operating Point:")
        print(f"  Threshold: {roc_data['optimal_threshold']:.6f}")
        print(f"  True Positive Rate (TPR): {roc_data['optimal_tpr']:.6f}")
        print(f"  False Positive Rate (FPR): {roc_data['optimal_fpr']:.6f}")

        print(f"\n4. BRIER SCORE DECOMPOSITION")
        print("-"*80)
        bs_decomp = self.brier_score_decomposition(y_prob, y_true)
        print(f"Brier Score: {bs_decomp['brier_score']:.6f}")
        print(f"Reliability (miscalibration): {bs_decomp['reliability']:.6f}")
        print(f"Resolution (discrimination): {bs_decomp['resolution']:.6f}")
        print(f"Uncertainty (irreducible): {bs_decomp['uncertainty']:.6f}")
        print(f"Reliability - Resolution: {bs_decomp['reliability_resolution_diff']:.6f}")

        print(f"\n5. CALIBRATION ERROR BY PROBABILITY RANGE")
        print("-"*80)
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_errors = []
        bin_centers = []
        bin_counts = []

        for i in range(self.n_bins):
            in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
            if np.sum(in_bin) > 0:
                avg_pred = np.mean(y_prob[in_bin])
                avg_true = np.mean(y_true[in_bin])
                bin_errors.append(avg_pred - avg_true)
                bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
                bin_counts.append(np.sum(in_bin))

        if bin_errors:
            print(f"{'Bin Center':<15} {'Error (Pred-True)':<20} {'Sample Count':<15} {'Status':<10}")
            print("-"*60)
            for center, error, count in zip(bin_centers, bin_errors, bin_counts):
                status = "Overconf" if error > 0 else "Underconf"
                print(f"{center:<15.4f} {error:<20.6f} {count:<15d} {status:<10}")

        print(f"\n6. ISOTONIC RECALIBRATION ANALYSIS")
        print("-"*80)
        y_prob_calibrated, iso_model = self.isotonic_recalibration(y_prob, y_true)
        ece_original = self.expected_calibration_error(y_prob, y_true)
        ece_calibrated = self.expected_calibration_error(y_prob_calibrated, y_true)

        print(f"ECE Before Recalibration: {ece_original:.6f}")
        print(f"ECE After Recalibration: {ece_calibrated:.6f}")
        print(f"ECE Improvement: {ece_original - ece_calibrated:.6f} ({((ece_original - ece_calibrated)/ece_original*100):.2f}%)")

        print(f"\nRecalibration Mapping (sample points):")
        print(f"{'Original Prob':<20} {'Recalibrated Prob':<20} {'Adjustment':<15}")
        print("-"*55)
        sample_indices = np.linspace(0, len(y_prob)-1, min(10, len(y_prob)), dtype=int)
        sort_idx = np.argsort(y_prob)
        for idx in sample_indices:
            orig = y_prob[sort_idx[idx]]
            recal = y_prob_calibrated[sort_idx[idx]]
            adj = recal - orig
            print(f"{orig:<20.6f} {recal:<20.6f} {adj:<15.6f}")

        print("="*80)

    def generate_calibration_report(self, y_prob, y_true, species_groups=None, temporal_bins=None):
        report = {
            'overall': {},
            'by_species': {},
            'by_temporal': {}
        }

        report['overall']['n_samples'] = len(y_prob)
        report['overall']['base_rate'] = np.mean(y_true)
        report['overall']['ece'] = self.expected_calibration_error(y_prob, y_true)
        report['overall']['mce'] = self.maximum_calibration_error(y_prob, y_true)
        report['overall']['ace'] = self.adaptive_calibration_error(y_prob, y_true)

        bs_decomp = self.brier_score_decomposition(y_prob, y_true)
        report['overall'].update(bs_decomp)

        roc_data = self.roc_analysis_with_confidence(y_prob, y_true)
        report['overall']['auc'] = roc_data['auc']
        report['overall']['auc_ci'] = roc_data['auc_ci']
        report['overall']['optimal_threshold'] = roc_data['optimal_threshold']

        try:
            report['overall']['log_loss'] = log_loss(y_true, y_prob)
        except:
            report['overall']['log_loss'] = np.nan

        if species_groups is not None:
            unique_species = np.unique(species_groups)
            for species in unique_species:
                mask = species_groups == species
                if np.sum(mask) > 10:
                    y_prob_species = y_prob[mask]
                    y_true_species = y_true[mask]

                    report['by_species'][species] = {
                        'n_samples': np.sum(mask),
                        'base_rate': np.mean(y_true_species),
                        'ece': self.expected_calibration_error(y_prob_species, y_true_species),
                        'brier_score': brier_score_loss(y_true_species, y_prob_species)
                    }

        if temporal_bins is not None:
            unique_temporal = np.unique(temporal_bins)
            for temporal in unique_temporal:
                mask = temporal_bins == temporal
                if np.sum(mask) > 10:
                    y_prob_temporal = y_prob[mask]
                    y_true_temporal = y_true[mask]

                    report['by_temporal'][temporal] = {
                        'n_samples': np.sum(mask),
                        'base_rate': np.mean(y_true_temporal),
                        'ece': self.expected_calibration_error(y_prob_temporal, y_true_temporal),
                        'brier_score': brier_score_loss(y_true_temporal, y_prob_temporal)
                    }

        return report
