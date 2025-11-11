import numpy as np
import pandas as pd
import json
import pathlib
from filterpy.kalman import KalmanFilter
from utils.logger import get_logger

logger = get_logger(__name__)
                                                                 
DEFAULT_SEASONAL_FACTORS = {
    'winter': {'tendency': 0.0, 'activity': 0.7},
    'spring': {'tendency': 0.1, 'activity': 1.2},
    'summer': {'tendency': 0.0, 'activity': 1.0},
    'fall': {'tendency': -0.1, 'activity': 1.2},
    'default': {'tendency': 0.0, 'activity': 1.0}
}

class MigrationDataProcessor:
    def __init__(self, migration_data_path=None, seasonal_factors_path='seasonal_factors.json',
                 species_taxonomy_path='species_taxonomy.json', use_preprocessed=True):
        
        self.data_path = migration_data_path
        self.use_preprocessed = use_preprocessed
        self.data = None
        self.movement_patterns = {}                                        
        self.species_map = {}
        self.species_taxonomy = {}

                                             
        if use_preprocessed:
            self._load_preprocessed_data(seasonal_factors_path, species_taxonomy_path)
        else:
            if not migration_data_path:
                raise ValueError("migration_data_path is required when use_preprocessed is False")

            logger.info(f"Loading migration data: {migration_data_path}")
            self.data = pd.read_csv(migration_data_path)
            logger.info(f"Loaded {len(self.data)} records.")
            self._process_migration_data()

    def _load_preprocessed_data(self, seasonal_factors_path, species_taxonomy_path):
        
                               
        seasonal_path = pathlib.Path(seasonal_factors_path)
        with open(seasonal_path, 'r') as f:
            self.movement_patterns = json.load(f)
        logger.info(f"Loaded preprocessed seasonal factors: {len(self.movement_patterns)} families")

                               
        taxonomy_path = pathlib.Path(species_taxonomy_path)
        with open(taxonomy_path, 'r') as f:
            self.species_taxonomy = json.load(f)
                                       
        species_to_family = self.species_taxonomy.get('species_to_family', {})
                                                       
        for species, family in species_to_family.items():
            if family not in self.species_map:
                self.species_map[family] = species
        logger.info(f"Loaded species taxonomy: {len(species_to_family)} species mapped")

    def _process_migration_data(self):
        family_col = 'Bird families'
        species_col = 'Bird species'
        month_col = 'Migration start month'

        self.species_map =        self.data.drop_duplicates(subset=[family_col])[[family_col, species_col]].set_index(family_col)[
            species_col].to_dict()

        self.data[month_col] = pd.to_numeric(self.data[month_col], errors='coerce')
        self.data.dropna(subset=[month_col], inplace=True)
        self.data[month_col] = self.data[month_col].astype(int)

        for family in self.data[family_col].unique():
            if pd.isna(family):
                continue

            self.movement_patterns[family] = {}
            family_data = self.data[self.data[family_col] == family]

            for season in ['winter', 'spring', 'summer', 'fall']:
                months = {'winter': [12, 1, 2], 'spring': [3, 4, 5], 'summer': [6, 7, 8], 'fall': [9, 10, 11]}[season]
                season_data = family_data[family_data[month_col].isin(months)]
                base_pattern = DEFAULT_SEASONAL_FACTORS.get(season, DEFAULT_SEASONAL_FACTORS['default'])

                self.movement_patterns[family][season] = {
                    'tendency': base_pattern['tendency'],
                    'activity': base_pattern['activity'] * (1.1 if not season_data.empty else 1.0)
                }

        logger.info("Processed migration data.")

    def get_seasonal_pattern(self, family, season):
        default_pattern = DEFAULT_SEASONAL_FACTORS.get(season.lower(), DEFAULT_SEASONAL_FACTORS['default']).copy()

        if pd.isna(family):
            return default_pattern

        family_patterns = self.movement_patterns.get(family)
        return family_patterns.get(season.lower(), default_pattern).copy() if family_patterns else default_pattern


class BirdStateTracker:
    def __init__(self, bird_id, initial_position, initial_uncertainty=None,
                 species_group='Unknown', family_group='Unknown'):
        self.bird_id = bird_id
        self.species_group = species_group
        self.family_group = family_group
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.x = np.zeros(6)
        self.kf.x[:3] = np.array(initial_position).flatten()
        self.kf.P = np.eye(6) * 5.0
        pos_vars = np.array(initial_uncertainty) ** 2 if initial_uncertainty else np.array(
            [0.1 ** 2, 0.1 ** 2, 0.05 ** 2])
        np.fill_diagonal(self.kf.P[:3, :3], pos_vars)
        np.fill_diagonal(self.kf.P[3:, 3:], [0.02 ** 2, 0.02 ** 2, 0.01 ** 2])

        self.kf.F = np.eye(6)

        self.kf.H = np.array([
            [1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.]
        ])

        self.kf.R = np.diag([0.05 ** 2, 0.05 ** 2, 0.08 ** 2])

        q_pos_var = 0.005 ** 2
        q_vel_var = 0.01 ** 2
        self.Q_base_vars = np.diag([q_pos_var, q_pos_var, q_pos_var, q_vel_var, q_vel_var, q_vel_var])

        self.position_history = [self.kf.x[:3].flatten().copy()]
        self.covariance_history = [self.kf.P.copy()]
        self.update_count = 0

    def predict(self, dt=1.0, seasonal_pattern=None):
        self.kf.F[0, 3] = dt
        self.kf.F[1, 4] = dt
        self.kf.F[2, 5] = dt

        Q_dt = self.Q_base_vars * dt

        self.kf.predict(Q=Q_dt)

        if seasonal_pattern:
            tendency = seasonal_pattern.get('tendency', 0.0)
            activity = seasonal_pattern.get('activity', 1.0)

            self.kf.x[4] += tendency * activity * dt * 0.1

            curr_vel = self.kf.x[3:].flatten()
            speed = np.linalg.norm(curr_vel)

            if speed > 1e-6:
                self.kf.x[3:] *= (speed * activity / speed)

        self.kf.x[2] = max(0.0, self.kf.x[2])

    def update(self, observation_pos, observation_uncertainty_std=None):
        z = np.array(observation_pos).reshape(3, 1)

        if observation_uncertainty_std is not None:
            self.kf.R = np.diag(np.array(observation_uncertainty_std) ** 2)

        try:
            self.kf.update(z)
        except np.linalg.LinAlgError:
            logger.warning(f"KF update fail bird {self.bird_id}")
            self.kf.P *= 1.01
            return

        self.kf.x[2] = max(0.0, self.kf.x[2])
        self.position_history.append(self.kf.x[:3].flatten().copy())
        self.covariance_history.append(self.kf.P.copy())
        self.update_count += 1

    def get_position(self):
        return self.kf.x[:3].flatten()

    def get_velocity(self):
        return self.kf.x[3:].flatten()

    def get_position_uncertainty(self):
        return np.sqrt(np.diag(self.kf.P[:3, :3]))


def get_bird_priors(position, season, family='Unknown', species='Unknown', migration_processor=None):
    prior_mu = tuple(position)
    seasonal_pattern = SEASONAL_FACTORS.get(season.capitalize(), SEASONAL_FACTORS['default']).copy()

    if migration_processor and family != 'Unknown':
        pattern_data = migration_processor.get_seasonal_pattern(family, season.capitalize())
        seasonal_pattern = pattern_data

    return {'mu': prior_mu, 'seasonal_pattern': seasonal_pattern}


def simulate_bird_movement(current_positions, trackers, season, dt=1.0, migration_processor=None):
    new_true_positions = []

    if len(current_positions) != len(trackers):
        logger.error("Pos/tracker mismatch.")
        return current_positions

    for i, tracker in enumerate(trackers):
        seasonal_pattern = None

        if migration_processor:
            family = getattr(tracker, 'family_group', 'Unknown')
            seasonal_pattern = migration_processor.get_seasonal_pattern(family, season.capitalize())

        tracker.predict(dt=dt, seasonal_pattern=seasonal_pattern)
        pred_pos = tracker.get_position()

        pos_noise_std = np.sqrt(np.diag(tracker.Q_base_vars[:3, :3])) * np.sqrt(dt)
        noise = np.random.normal(0.0, pos_noise_std)
        new_true_pos = pred_pos + noise

        new_true_pos[2] = max(0.0, new_true_pos[2])
        new_true_positions.append(tuple(new_true_pos))

    return new_true_positions


def initialize_bird_trackers(initial_positions, species_list=None, family_list=None):
    trackers = []
    num_birds = len(initial_positions)

    if species_list is None or len(species_list) != num_birds:
        species_list = ['Unknown'] * num_birds

    if family_list is None or len(family_list) != num_birds:
        family_list = ['Unknown'] * num_birds

    for i, pos in enumerate(initial_positions):
        tracker = BirdStateTracker(
            bird_id=i,
            initial_position=pos,
            species_group=species_list[i],
            family_group=family_list[i]
        )
        trackers.append(tracker)

    logger.info(f"Initialized {len(trackers)} KF trackers.")
    return trackers
