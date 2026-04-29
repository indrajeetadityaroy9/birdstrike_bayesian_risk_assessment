from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

def configure_visualization(enable_plots=True):
    if not enable_plots:
        logger.info("Plotting disabled via --no-plots flag")
        return False

    style_config = {
        'style': 'seaborn-v0_8-darkgrid',
        'figure.figsize': (11, 8),
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'text.usetex': False,
    }

    plt.style.use(style_config['style'])
    plt.rcParams['figure.figsize'] = style_config['figure.figsize']
    plt.rcParams['figure.dpi'] = style_config['figure.dpi']
    plt.rcParams['savefig.dpi'] = style_config['savefig.dpi']
    plt.rcParams['font.size'] = style_config['font.size']
    plt.rcParams['axes.titlesize'] = style_config['axes.titlesize']
    plt.rcParams['axes.labelsize'] = style_config['axes.labelsize']
    plt.rcParams['xtick.labelsize'] = style_config['xtick.labelsize']
    plt.rcParams['ytick.labelsize'] = style_config['ytick.labelsize']
    plt.rcParams['legend.fontsize'] = style_config['legend.fontsize']
    plt.rcParams['text.usetex'] = style_config['text.usetex']
    logger.info("Matplotlib configured.")
    return True

def plot_simulation_results(results, output_dir):
    logger.info("Generating probabilistic plots...")
    timestamps = results.get("timestamps", [])
    if not timestamps:
        logger.warning("No timestamps for plots.")
        return

    iterations = range(1, len(timestamps) + 1)
    output_path = Path(output_dir)

    position_errors = results.get("position_errors")
    if position_errors:
        cleaned_errors = [[err for err in step if err is not None] for step in position_errors if step]
        if cleaned_errors:
            avg_errors = [np.mean(step) if step else np.nan for step in cleaned_errors]
            max_errors = [np.max(step) if step else np.nan for step in cleaned_errors]

            plt.figure(figsize=(10, 5))
            plt.plot(iterations, avg_errors, label='Avg Error', marker='.')
            plt.plot(iterations, max_errors, label='Max Error', marker='.', alpha=0.7)
            plt.xlabel("Step")
            plt.ylabel("Error (km)")
            plt.title("Position Error")
            plt.legend()
            plt.grid(alpha=0.5)
            plt.ylim(bottom=0)
            plt.tight_layout()
            path = output_path / "pos_error.png"
            plt.savefig(path)
            plt.close()
            logger.info("Saved %s", path)

    path_risk_assessments = results.get("path_risk_assessments")
    if path_risk_assessments:
        plt.figure(figsize=(10, 5))

        if path_risk_assessments:
            path_ids = list(path_risk_assessments[0].keys())
            for path_id in path_ids:
                name = path_risk_assessments[0].get(path_id, {}).get('path_name', path_id)
                avg_prob = [
                    step.get(path_id, {}).get('avg_prob_medium_or_high', np.nan)
                    for step in path_risk_assessments
                ]
                plt.plot(iterations, avg_prob, label=f'{name} Avg P(Risk>=Med)', marker='.', ms=4)

            plt.legend(fontsize=8, loc='best')

        plt.xlabel("Simulation Step")
        plt.ylabel("Probability")
        plt.title("Average Path Risk Probability (P(Risk>=Medium))")
        plt.grid(alpha=0.5)
        plt.ylim(0, 1)
        plt.tight_layout()
        path = output_path / "path_risk_probability_plot.png"
        plt.savefig(path)
        plt.close()
        logger.info("Saved %s", path)

def plot_risk_map_snapshot(risk_mapper, scenario, estimated_positions, trackers, true_positions, output_dir, iteration_index, z_level=0.5):
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_est = [list(p) if p is not None else None for p in estimated_positions]
    plot_true = [list(p) if p is not None else None for p in true_positions]

    risk_mapper.plot_2d_risk_map(
        z_level=z_level,
        ax=ax,
        plot_type='probability',
        adaptive_contours=True,
        runways=scenario['runways'],
        flight_paths=scenario['flight_paths'],
        bird_positions=plot_est,
        bird_trackers=trackers,
        actual_positions=plot_true,
        show_confidence=False,
    )

    plot_path = Path(output_dir) / f"risk_map_iter_{iteration_index:03d}.png"
    plt.savefig(plot_path)
    plt.close(fig)
    logger.info("Saved %s", plot_path)

def plot_difference_maps(risk_mapper, prev_risk_maps, scenario, trackers, estimated_positions, output_dir, iteration_index, z_level=0.5):
    plot_est = [list(p) if p is not None else None for p in estimated_positions]

    previous_prob_map = prev_risk_maps.get('prob_med_or_high')
    if previous_prob_map is not None:
        fig, ax = plt.subplots(figsize=(12, 10))
        risk_mapper.plot_temporal_difference_map(
            previous_risk_map=previous_prob_map,
            z_level=z_level,
            ax=ax,
            runways=scenario['runways'],
            flight_paths=scenario['flight_paths'],
            bird_positions=plot_est,
            bird_trackers=trackers,
        )
        diff_plot_path = Path(output_dir) / f"risk_diff_map_iter_{iteration_index:03d}.png"
        plt.savefig(diff_plot_path)
        plt.close(fig)
        logger.info("Saved %s", diff_plot_path)

    mean_map, std_map, prob_high_map, prob_med_or_high_map = risk_mapper.compute_risk_map(z_level=z_level)
    return {
        'mean': mean_map.copy() if mean_map is not None else None,
        'std': std_map.copy() if std_map is not None else None,
        'prob_high': prob_high_map.copy() if prob_high_map is not None else None,
        'prob_med_or_high': prob_med_or_high_map.copy() if prob_med_or_high_map is not None else None,
    }
