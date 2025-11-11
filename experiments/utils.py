import json
import numpy as np
from experiments.constants import (
    DEFAULT_SCENARIO,
    DEFAULT_SEASONAL_DISTRIBUTIONS,
    DEFAULT_SPECIES_TO_FAMILY,
)
from utils.geometry import calculate_distance_3d
from utils.logger import get_logger

logger = get_logger(__name__)

def setup_airport_scenario():
    logger.debug("Setting up default airport scenario.")
    return DEFAULT_SCENARIO.copy()

def create_initial_birds(
    num_birds,
    scenario,
    season=None,
    seed=None,
    species_dist_path='data/processed/species_distributions.json',
    species_tax_path='data/processed/species_taxonomy.json',
    use_preprocessed=True,
):
    if seed is not None:
        np.random.seed(seed)

    if not use_preprocessed:
        raise NotImplementedError("create_initial_birds currently requires preprocessed species data")

    x_min, x_max = scenario["x_range"]
    y_min, y_max = scenario["y_range"]
    z_min, z_max = scenario["z_range"]

    positions = [
        (
            np.random.uniform(x_min * 0.8, x_max * 0.8),
            np.random.uniform(y_min * 0.8, y_max * 0.8),
            np.random.uniform(z_min + 0.1, z_max * 0.8),
        )
        for _ in range(num_birds)
    ]

    species_to_family = DEFAULT_SPECIES_TO_FAMILY.copy()
    seasonal_distributions = DEFAULT_SEASONAL_DISTRIBUTIONS.copy()

    try:
        with open(species_tax_path, 'r', encoding='utf-8') as f:
            species_taxonomy = json.load(f)
        species_to_family.update(species_taxonomy.get('species_to_family', {}))
        logger.info("Loaded species taxonomy: %d mappings", len(species_to_family))
    except FileNotFoundError:
        logger.warning("Species taxonomy file not found: %s", species_tax_path)
    except json.JSONDecodeError as exc:
        logger.warning("Species taxonomy JSON decode error: %s", exc)

    try:
        with open(species_dist_path, 'r', encoding='utf-8') as f:
            loaded_distributions = json.load(f)

        formatted = {
            season_key: dist_data
            for season_key, dist_data in loaded_distributions.items()
            if isinstance(dist_data, dict)
            and 'species' in dist_data
            and 'probabilities' in dist_data
        }

        if formatted:
            seasonal_distributions.update(formatted)
            logger.info("Loaded species distributions: %d seasons", len(seasonal_distributions))
    except FileNotFoundError:
        logger.warning("Species distributions file not found: %s", species_dist_path)
    except json.JSONDecodeError as exc:
        logger.warning("Species distributions JSON decode error: %s", exc)

    season_lower = season.lower() if season else "unknown"
    season_cap = season.capitalize() if season else "Unknown"

    if season_lower in seasonal_distributions:
        dist = seasonal_distributions[season_lower]
    else:
        dist = seasonal_distributions.get('default', DEFAULT_SEASONAL_DISTRIBUTIONS['default'])

    s_choices = dist['species']
    probabilities = dist['probabilities']
    if not np.isclose(sum(probabilities), 1.0):
        probabilities = np.array(probabilities) / sum(probabilities)

    species_list = np.random.choice(s_choices, num_birds, p=probabilities).tolist()
    family_list = [species_to_family.get(species, 'Unknown') for species in species_list]

    species_summary = {s: species_list.count(s) for s in set(species_list)}
    logger.info(
        "Created %d birds for season '%s'. Species distribution: %s",
        num_birds,
        season_cap,
        species_summary,
    )

    return positions, species_list, family_list


def simulate_radar_observations(true_positions, sensors):
    observations = []
    sensor_pos = np.array([s['position'] for s in sensors])
    accuracies = np.array([s['accuracy'] for s in sensors])
    ranges = np.array([s['range'] for s in sensors])

    for true_pos in true_positions:
        detected_by = []
        for i, s_pos in enumerate(sensor_pos):
            dist = calculate_distance_3d(*true_pos, *s_pos)
            detection_prob = max(0, 1.0 - (dist / ranges[i]) ** 2)
            if dist <= ranges[i] and np.random.rand() < detection_prob:
                noise = np.random.normal(0.0, accuracies[i], 3)
                detected_by.append((np.array(true_pos) + noise, accuracies[i]))

        if detected_by:
            fused_pos = np.mean([obs for obs, _ in detected_by], axis=0)
            avg_acc = np.mean([acc for _, acc in detected_by])
            observations.append((tuple(fused_pos), np.array([avg_acc] * 3)))
        else:
            observations.append((None, None))

    return observations
