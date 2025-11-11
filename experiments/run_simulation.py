import argparse
import datetime
import json
import logging
import pathlib
import numpy as np
from models.gp_mapper import SequentialGPRiskMapper, evaluate_flight_path_risk
from models.kalman_tracker import MigrationDataProcessor, BirdStateTracker, initialize_bird_trackers, simulate_bird_movement
from models.risk_model import BirdStrikeRiskSystem
from utils.geometry import calculate_distance_3d
from utils.logger import configure_logging, get_logger
from experiments.constants import FORCE_DEBUG_MODULES
from experiments.utils import (
    create_initial_birds,
    setup_airport_scenario,
    simulate_radar_observations,
)
from experiments.visualization import (
    configure_visualization,
    plot_difference_maps,
    plot_risk_map_snapshot,
    plot_simulation_results,
)

logger = get_logger(__name__)

def generate_report(results, output_dir, enable_plots=True):
    logger.info("Generating probabilistic report...")

    if results.get("position_errors"):
        errs = [e for s in results["position_errors"] if s for e in s if e is not None]
        avg_err = np.mean(errs) if errs else float('nan')
        logger.info(f"Avg Pos Error: {avg_err:.4f} km")

    if results.get("path_risk_assessments"):
        if results["path_risk_assessments"]:
            final_risks = results["path_risk_assessments"][-1]
            logger.info("Final Path Risk Assessment (Probabilistic):")
            for pid, r in final_risks.items():
                logger.info(
                    f"  - {r.get('path_name', pid)}: "
                    f"Avg P(>=Med)={r.get('avg_prob_medium_or_high', 0):.1%}, "
                    f"Avg P(High)={r.get('avg_prob_high', 0):.1%}, "
                    f"Avg Unc={r.get('mean_std_dev', 0):.3f}"
                )
        else:
            logger.info("Path risk list empty.")
    else:
        logger.info("No path risk data.")

    results_file = str(pathlib.Path(output_dir) / "sim_summary_prob.json")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer, np.floating)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (datetime.datetime, datetime.date)):
                return o.isoformat()
            if isinstance(o, BirdStateTracker):
                return f"<Tracker id={o.bird_id}>"
            return str(o)

    serializable_results = json.loads(json.dumps(results, cls=NumpyEncoder))
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Results saved: {results_file}")

    if enable_plots:
        plot_simulation_results(results, output_dir)
    else:
        logger.info("Skipping plot generation (plotting disabled)")

def run_simulation(num_iterations, num_birds, timestep, output_dir,
                   faa_data_path, migration_data_path, seed, plot_interval, enable_plots=True):
    scenario = setup_airport_scenario()
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Init FAA system (probabilistic V3): {faa_data_path}")
    faa_risk_system = BirdStrikeRiskSystem(
        faa_data_path=faa_data_path,
        use_preprocessed=False,
        spatial_profiles_path="data/processed/spatial_risk_profiles.json"
    )
    faa_risk_system.load_faa_data()
    faa_risk_system.analyze_species_risk()
    faa_risk_system.analyze_temporal_patterns()
    faa_risk_system.analyze_spatial_patterns()

    migration_processor = None
    if migration_data_path:
        logger.info(f"Init Migration processor: {migration_data_path}")
        migration_processor = MigrationDataProcessor(
            seasonal_factors_path="data/processed/seasonal_factors.json",
            species_taxonomy_path="data/processed/species_taxonomy.json",
            use_preprocessed=True
        )
        logger.info("OK: Migration Proc Init.")
    else:
        logger.info("No migration data path provided; skipping migration processor initialization.")

    sim_time = datetime.datetime.now()
    current_month = sim_time.month
    current_season = (
        "Winter" if current_month in [12, 1, 2] else
        "Spring" if current_month in [3, 4, 5] else
        "Summer" if current_month in [6, 7, 8] else "Fall"
    )
    logger.info(f"Context: Season='{current_season}', Month={current_month}")

    initial_positions, species_list, family_list = create_initial_birds(
        num_birds, scenario, season=current_season, seed=seed
    )
    trackers = initialize_bird_trackers(initial_positions, species_list, family_list)
    true_positions = list(initial_positions)
    risk_mapper = SequentialGPRiskMapper(
        scenario['x_range'], scenario['y_range'], scenario['z_range'], resolution=0.25
    )

    results = {
        "timestamps": [],
        "true_positions": [],
        "estimated_positions": [],
        "observations": [],
        "position_errors": [],
        "path_risk_assessments": []
    }

    prev_risk_maps = {
        'mean': None,
        'std': None,
        'prob_high': None,
        'prob_med_or_high': None,
    }

    logger.info("Starting simulation loop...")
    for i in range(num_iterations):
        timestamp = datetime.datetime.now()
        logger.info(f"-- Iteration {i + 1}/{num_iterations} | Season: {current_season}, Month: {current_month} --")

        true_positions = simulate_bird_movement(
            true_positions, trackers, current_season,
            dt=timestep / 60., migration_processor=migration_processor
        )
        results["true_positions"].append([p for p in true_positions])

        observations_step = simulate_radar_observations(true_positions, scenario['sensors'])
        results["observations"].append(observations_step)

        estimated_positions_step = []
        position_errors_step = []
        for j, tracker in enumerate(trackers):
            obs_pos_tuple, obs_unc_std_tuple = observations_step[j]
            if obs_pos_tuple is not None:
                tracker.update(
                    np.array(obs_pos_tuple),
                    np.array(obs_unc_std_tuple) if obs_unc_std_tuple is not None else None
                )
            est_pos = tracker.get_position()
            estimated_positions_step.append(est_pos)
            if est_pos is None or true_positions[j] is None:
                position_errors_step.append(None)
            else:
                position_errors_step.append(calculate_distance_3d(*true_positions[j], *est_pos))

        results["estimated_positions"].append(estimated_positions_step)
        results["position_errors"].append(position_errors_step)

        valid_errors = [e for e in position_errors_step if e is not None]
        avg_err = np.mean(valid_errors) if valid_errors else float('nan')
        max_err = np.max(valid_errors) if valid_errors else float('nan')
        logger.info(f"  KF Update: Avg Error={avg_err:.3f} km, Max Error={max_err:.3f} km")

        valid_est_pos = [tuple(p) for p in estimated_positions_step if p is not None]
        valid_trackers = [t for t, p in zip(trackers, estimated_positions_step) if p is not None]

        if valid_est_pos:
            valid_species = [t.species_group for t in valid_trackers]
            valid_families = [t.family_group for t in valid_trackers]

            X_train, y_train = risk_mapper.generate_training_data(
                valid_est_pos, valid_species, valid_families,
                scenario['flight_paths'], faa_risk_system, valid_trackers,
                current_month, current_season
            )

            if X_train.size > 0:
                risk_mapper.update_sequential(X_train, y_train, timestamp)
                logger.info(f"  GP map updated. LML={risk_mapper.log_marginal_likelihood:.1f}")
            else:
                logger.debug("  No GP training data.")
        else:
            logger.info("  No valid birds for GP update.")

        path_risks_step = {}
        for path in scenario['flight_paths']:
            path_risk_info = evaluate_flight_path_risk(risk_mapper, path, num_samples=50)
            path_risks_step[path['id']] = path_risk_info
            logger.debug(
                f"  Path '{path['name']}': "
                f"AvgP(>=Med)={path_risk_info['avg_prob_medium_or_high']:.1%}, "
                f"AvgP(High)={path_risk_info['avg_prob_high']:.1%}, "
                f"AvgUnc={path_risk_info['mean_std_dev']:.3f}"
            )

        results["path_risk_assessments"].append(path_risks_step)

        if enable_plots and ((i + 1) % plot_interval == 0 or i == num_iterations - 1):
            iteration_index = i + 1
            logger.info(f"Generating plots iter {iteration_index}...")
            plot_risk_map_snapshot(
                risk_mapper=risk_mapper,
                scenario=scenario,
                estimated_positions=estimated_positions_step,
                trackers=trackers,
                true_positions=true_positions,
                output_dir=output_dir,
                iteration_index=iteration_index,
            )

            prev_risk_maps = plot_difference_maps(
                risk_mapper=risk_mapper,
                prev_risk_maps=prev_risk_maps,
                scenario=scenario,
                trackers=trackers,
                estimated_positions=estimated_positions_step,
                output_dir=output_dir,
                iteration_index=iteration_index,
            )

        results["timestamps"].append(timestamp)
        logger.info(f"Iteration {i + 1} completed.\n")

    logger.info("Simulation loop completed.")
    generate_report(results, output_dir, enable_plots)
    logger.info("Simulation finished.")

def main():
    parser = argparse.ArgumentParser(
        description='Bird Strike Risk Simulation (Probabilistic Risk V3)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--iterations', type=int, default=30,
                        help='Number of simulation iterations (default: 30)')
    parser.add_argument('--birds', type=int, default=15,
                        help='Number of birds in simulation (default: 15)')
    parser.add_argument('--timestep', type=int, default=10,
                        help='Timestep in seconds (default: 10)')
    parser.add_argument('--output-dir', type=str, default='simulation_output_prob_risk_v3',
                        help='Output directory (default: simulation_output_prob_risk_v3)')
    parser.add_argument('--faa-data', type=str, default='data/raw/wildlife_strikes.csv',
                        help='Path to FAA wildlife strikes data (default: data/raw/wildlife_strikes.csv)')
    parser.add_argument('--migration-data', type=str, default='data/raw/bird_migration.csv',
                        help='Path to migration data (default: data/raw/bird_migration.csv)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    parser.add_argument('--plot-interval', type=int, default=5,
                        help='Interval for generating plots (default: 5)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable all plot generation for faster execution')

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    configure_logging(log_level)
    global logger
    logger.setLevel(log_level)
    logger.info(f"Default Log Level set to: {args.log_level.upper()}")

    for module_name in FORCE_DEBUG_MODULES:
        module_logger = logging.getLogger(module_name)
        if module_logger.level > logging.DEBUG:
            module_logger.setLevel(logging.DEBUG)
            logger.info(f"Forcing DEBUG level for module: {module_name}")
    enable_plots = configure_visualization(enable_plots=not args.no_plots)

    logger.info("--- Bird Strike Risk Simulation ---")
    logger.info(f"Params: Iter={args.iterations}, Birds={args.birds}, TS={args.timestep}s, Out='{args.output_dir}'")
    logger.info(f"FAA='{args.faa_data}', Mig='{args.migration_data if args.migration_data else 'None'}'")
    logger.info(f"Plotting: {'Disabled' if args.no_plots else 'Enabled'}")

    actual_mig_path = args.migration_data if args.migration_data else None
    if actual_mig_path:
        logger.info(f"Using migration data: {actual_mig_path}")
    else:
        logger.info("No migration data path specified.")

    if args.seed is not None:
        np.random.seed(args.seed)
        logger.info(f"Seed set: {args.seed}")

    run_simulation(
        args.iterations, args.birds, args.timestep, args.output_dir,
        args.faa_data, actual_mig_path, args.seed, args.plot_interval, enable_plots
    )

    logger.info("--- Simulation Complete ---")

if __name__ == "__main__":
    main()
