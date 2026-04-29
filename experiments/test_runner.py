import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from datetime import datetime
from models.gp_mapper import SequentialGPRiskMapper, evaluate_flight_path_risk
from models.kalman_tracker import (
    MigrationDataProcessor,
    initialize_bird_trackers, simulate_bird_movement
)
from models.risk_model import BirdStrikeRiskSystem
from utils.geometry import calculate_distance_3d, calculate_distance_to_path
from utils.logger import get_logger, configure_logging
from evaluation.calibration import BayesianCalibration

from experiments.constants import (
    RISK_LEVEL_THRESHOLDS,
    SCENARIO_GROUPS,
    TEST_SCENARIOS,
)
from experiments.experiment_utils import (
    create_initial_birds,
    setup_airport_scenario,
    simulate_radar_observations,
)

logger = get_logger(__name__)

def compute_step_risk(positions, species_list, family_list, flight_paths,risk_system, trackers, month, season):
    if not positions:
        return None
    
    probabilities = []
    
    for i, (pos, species, family) in enumerate(zip(positions, species_list, family_list)):
        if pos is None:
            continue
            
        try:
            nearest_path = None
            min_distance = float('inf')
            for path in flight_paths:
                dist = calculate_distance_3d(pos[0], pos[1], pos[2], path['start'][0], path['start'][1], path['start'][2])
                if dist < min_distance:
                    min_distance = dist
                    nearest_path = path
            
            if nearest_path is not None:
                risk_metrics = risk_system.calculate_probabilistic_risk_metrics(
                    pos, nearest_path, family, species, month, season
                )
                base_prob = risk_metrics['mean_prob']
            else:
                base_prob = 0.1

            if nearest_path is not None:
                distance_factor = max(0, 1.0 - min_distance / 5.0)
                adjusted_prob = base_prob * (0.5 + 0.5 * distance_factor)
                probabilities.append(adjusted_prob)
            else:
                probabilities.append(base_prob)
                
        except Exception as e:
            logger.warning(f"Error computing risk for bird {i}: {e}")
            probabilities.append(0.1)
    
    if probabilities:
        return {
            'probabilities': probabilities,
            'mean_prob': float(np.mean(probabilities)),
            'max_prob': float(np.max(probabilities))
        }
    
    return None

def init_preprocessed_risk_system():
    system = BirdStrikeRiskSystem(
        faa_data_path="data/raw/wildlife_strikes.csv",
        use_preprocessed=True,
        spatial_profiles_path="data/processed/spatial_risk_profiles.json"
    )
    system.load_faa_data()
    system.analyze_species_risk()
    system.analyze_temporal_patterns()
    return system

def init_migration_processor():
    return MigrationDataProcessor(
        seasonal_factors_path="data/processed/seasonal_factors.json",
        species_taxonomy_path="data/processed/species_taxonomy.json",
        use_preprocessed=True
    )

class ComprehensiveTestRunner:
    
    def __init__(self, output_dir="test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.faa_risk_system = init_preprocessed_risk_system()
        self.migration_processor = init_migration_processor()
        
    def run_single_scenario(self, scenario_key, run_id=0):
        scenario = TEST_SCENARIOS[scenario_key]
        logger.info(f"Running scenario: {scenario['name']} (Run {run_id + 1})")

        airport_scenario = setup_airport_scenario()
        num_birds = scenario["num_birds"]
        season = scenario["season"]
        num_iterations = scenario["num_iterations"]
        month = 11

        initial_positions, species_list, family_list = create_initial_birds(
            num_birds=num_birds,
            scenario=airport_scenario,
            season=season,
            seed=42 + run_id,
            species_dist_path="data/processed/species_distributions.json",
            species_tax_path="data/processed/species_taxonomy.json"
        )

        trackers = initialize_bird_trackers(initial_positions, species_list, family_list)

        risk_mapper = SequentialGPRiskMapper(
            airport_scenario['x_range'],
            airport_scenario['y_range'],
            airport_scenario['z_range'],
            resolution=0.25
        )

        metrics = {
            "position_errors": [],
            "velocity_errors": [],
            "risk_probabilities": [],
            "path_risks": [],
            "proximity_events": 0,
            "high_risk_detections": 0
        }

        true_positions = initial_positions.copy()
        estimated_positions = initial_positions.copy()

        for iteration in range(num_iterations):
            true_positions = simulate_bird_movement(
                true_positions, trackers, season,
                dt=10.0/60.0, migration_processor=self.migration_processor
            )

            observations = simulate_radar_observations(
                true_positions, airport_scenario['sensors']
            )

            for i, (obs, tracker) in enumerate(zip(observations, trackers)):
                if obs is not None and obs[0] is not None:
                    tracker.update(obs[0], obs[1])
                    est_pos = tracker.get_position()
                    estimated_positions[i] = tuple(est_pos)

            valid_trackers = [t for t in trackers if t.bird_id < len(true_positions)]
            if valid_trackers and len(true_positions) > 0:
                valid_true_positions = true_positions[:len(valid_trackers)]
                if valid_true_positions:
                    true_pos_array = np.array(valid_true_positions)
                    est_pos_array = np.array([tracker.get_position() for tracker in valid_trackers])
                    est_vel_array = np.array([tracker.get_velocity() for tracker in valid_trackers])

                    pos_errors = np.linalg.norm(true_pos_array - est_pos_array, axis=1)
                    vel_errors = np.linalg.norm(est_vel_array, axis=1)

                    metrics["position_errors"].extend(pos_errors.tolist())
                    metrics["velocity_errors"].extend(vel_errors.tolist())

            valid_positions = [p for p in estimated_positions if p is not None]
            if valid_positions:
                risk_info = compute_step_risk(
                    valid_positions, species_list[:len(valid_positions)],
                    family_list[:len(valid_positions)],
                    airport_scenario['flight_paths'],
                    self.faa_risk_system, trackers[:len(valid_positions)],
                    month=11, season=season
                )

                if risk_info and 'probabilities' in risk_info:
                    metrics["risk_probabilities"].extend(risk_info['probabilities'])

                    high_risk_count = sum(1 for p in risk_info['probabilities'] if p > 0.6)
                    metrics["high_risk_detections"] += high_risk_count

                    proximity_threshold = 0.5
                    for path in airport_scenario['flight_paths']:
                        distances = np.array([
                            calculate_distance_to_path(pos, path) 
                            for pos in valid_positions
                        ])
                        metrics["proximity_events"] += np.sum(distances < proximity_threshold)

                if iteration % 5 == 0 and valid_positions:
                    X_train, y_train = risk_mapper.generate_training_data(
                        valid_positions, 
                        species_list[:len(valid_positions)], 
                        family_list[:len(valid_positions)],
                        airport_scenario['flight_paths'], 
                        self.faa_risk_system, 
                        trackers[:len(valid_positions)],
                        month, 
                        season
                    )
                    
                    if X_train.size > 0:
                        risk_mapper.update_sequential(X_train, y_train)

                    for path in airport_scenario['flight_paths']:
                        path_risk = evaluate_flight_path_risk(risk_mapper, path, num_samples=20)
                        if path_risk and 'mean_risk' in path_risk:
                            metrics["path_risks"].append(path_risk['mean_risk'])

        return self._aggregate_metrics(metrics, scenario)

    def _aggregate_metrics(self, metrics, scenario):
        aggregated = {}

        aggregated["mean_position_error"] = np.mean(metrics["position_errors"]) if metrics["position_errors"] else 0
        aggregated["std_position_error"] = np.std(metrics["position_errors"]) if metrics["position_errors"] else 0
        aggregated["max_position_error"] = np.max(metrics["position_errors"]) if metrics["position_errors"] else 0

        aggregated["mean_velocity_error"] = np.mean(metrics["velocity_errors"]) if metrics["velocity_errors"] else 0

        aggregated["mean_risk_probability"] = np.mean(metrics["risk_probabilities"]) if metrics["risk_probabilities"] else 0
        aggregated["max_risk_probability"] = np.max(metrics["risk_probabilities"]) if metrics["risk_probabilities"] else 0

        aggregated["proximity_events"] = metrics["proximity_events"]
        aggregated["high_risk_detections"] = metrics["high_risk_detections"]

        if metrics["risk_probabilities"]:
            expected_level = scenario["expected_risk_level"]
            if expected_level in RISK_LEVEL_THRESHOLDS:
                expected_mean = RISK_LEVEL_THRESHOLDS[expected_level]["mean"]
                aggregated["risk_prediction_accuracy"] = 1.0 - abs(
                    aggregated["mean_risk_probability"] - expected_mean
                )

        return aggregated

    def run_test_suite(self, group="quick"):
        scenario_keys = SCENARIO_GROUPS.get(group, ["baseline_nominal"])
        logger.info(f"Running Test Suite: {group}")
        logger.info(f"Scenarios: {len(scenario_keys)}")
        
        results = []
        for key in scenario_keys:
            try:
                result = self.run_single_scenario(key)
                result["scenario"] = key
                result["scenario_name"] = TEST_SCENARIOS[key]["name"]
                results.append(result)
                logger.info(f"Completed scenario: {TEST_SCENARIOS[key]['name']}")
            except Exception as exc:
                logger.error(f"Failed to run {key}: {exc}")
        
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"test_suite_{group}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        self.print_summary(df)
        return df

    def print_summary(self, df):
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)
        
        if df.empty:
            print("No results to display")
            return
            
        print("\nScenarios Tested:")
        for _, row in df.iterrows():
            print(f"  • {row['scenario_name']}")

        print("\nTracking Performance:")
        print(f"  Mean Position Error: {df['mean_position_error'].mean():.4f} km")
        print(f"  Std Position Error: {df['std_position_error'].mean():.4f} km")
        print(f"  Max Position Error: {df['max_position_error'].max():.4f} km")

        print("\nRisk Assessment:")
        print(f"  Mean Risk Level: {df['mean_risk_probability'].mean():.3f}")
        print(f"  Max Risk Level: {df['max_risk_probability'].max():.3f}")
        print(f"  Total Proximity Events: {df['proximity_events'].sum()}")

        print("\nPer-Scenario Results:")
        print("-" * 80)
        for _, row in df.iterrows():
            print(
                f"{row['scenario_name']:35} | "
                f"Error: {row['mean_position_error']:.3f} km | "
                f"Risk: {row['mean_risk_probability']:.3f}"
            )
        print("=" * 80)

    def run_bayesian_validation(self, n_iterations=50, n_birds=20):
        logger.info("Starting Bayesian validation...")
        
        calibration = BayesianCalibration(n_bins=10)

        scenario = setup_airport_scenario()

        positions, species, families = create_initial_birds(
            num_birds=n_birds,
            scenario=scenario,
            season="Fall",
            seed=42,
            species_dist_path="data/processed/species_distributions.json",
            species_tax_path="data/processed/species_taxonomy.json"
        )

        trackers = initialize_bird_trackers(positions, species, families)

        risk_mapper = SequentialGPRiskMapper(
            scenario['x_range'],
            scenario['y_range'],
            scenario['z_range'],
            resolution=0.25
        )

        self.faa_risk_system.load_faa_data()
        self.faa_risk_system.analyze_species_risk()
        self.faa_risk_system.analyze_temporal_patterns()

        predictions = []
        ground_truth = []
        species_labels = []
        temporal_labels = []
        true_positions = list(positions)
        estimated_positions = list(positions)

        for iteration in range(n_iterations):
            logger.info(f"Bayesian Validation Iteration {iteration + 1}/{n_iterations}")

            true_positions = simulate_bird_movement(
                true_positions, trackers, "Fall", dt=0.1
            )

            observations = []
            for pos in true_positions:
                if pos is not None:
                    noise = np.random.normal(0, 0.05, 3)
                    obs_pos = np.array(pos) + noise
                    observations.append(obs_pos)
                else:
                    observations.append(None)

            for j, (obs, tracker) in enumerate(zip(observations, trackers)):
                if obs is not None:
                    tracker.update(obs, [0.05, 0.05, 0.08])
                    estimated_positions[j] = tuple(tracker.get_position())

            for j, (pos, sp, fam) in enumerate(zip(estimated_positions, species, families)):
                if pos is None:
                    continue

                sample_path = scenario['flight_paths'][0] if scenario['flight_paths'] else None
                if sample_path is not None:
                    risk_metrics = self.faa_risk_system.calculate_probabilistic_risk_metrics(
                        pos, sample_path, fam, sp, 10, "Fall"
                    )
                    risk_value = risk_metrics['mean_prob']
                else:
                    risk_value = 0.1
                
                predictions.append(risk_value)
                gt = np.random.binomial(1, risk_value)
                ground_truth.append(gt)
                species_labels.append(sp)
                temporal_labels.append(f"iter_{iteration}")

        logger.info("Bayesian validation complete")
        logger.info(f"Generated {len(predictions)} validation samples")

        if predictions and ground_truth:
            calibration_result = calibration.generate_calibration_report(
                np.array(predictions),
                np.array(ground_truth),
                species_groups=np.array(species_labels)
            )
            logger.info(f"Calibration ECE: {calibration_result['overall'].get('ece', 'N/A')}")
            logger.info(f"Calibration MCE: {calibration_result['overall'].get('mce', 'N/A')}")
            logger.info(f"Brier Score: {calibration_result['overall'].get('brier_score', 'N/A')}")

        return {
            'predictions': predictions,
            'ground_truth': ground_truth,
            'species_labels': species_labels,
            'calibration': calibration_result if 'calibration_result' in locals() else None
        }

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Test Runner")
    parser.add_argument(
        "--test-type", 
        choices=["quick", "comprehensive", "bayesian", "single"],
        default="quick",
        help="Type of test to run"
    )
    parser.add_argument(
        "--scenario", 
        default="baseline_nominal",
        help="Specific scenario to run (for single test type)"
    )
    parser.add_argument(
        "--output-dir", 
        default="test_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    configure_logging()
    
    runner = ComprehensiveTestRunner(output_dir=args.output_dir)
    
    if args.test_type == "quick":
        logger.info("Running quick test suite...")
        runner.run_test_suite("quick")
    elif args.test_type == "comprehensive":
        logger.info("Running comprehensive test suite...")
        runner.run_test_suite("comprehensive")
    elif args.test_type == "bayesian":
        logger.info("Running Bayesian validation...")
        runner.run_bayesian_validation()
    elif args.test_type == "single":
        logger.info(f"Running single scenario: {args.scenario}")
        result = runner.run_single_scenario(args.scenario)
        result = convert_numpy_types(result)
        print(json.dumps(result, indent=2))
    
    logger.info("Test run completed")

if __name__ == "__main__":
    main()
