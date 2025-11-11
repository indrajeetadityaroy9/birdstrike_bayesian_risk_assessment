DEFAULT_SCENARIO = {
    "x_range": (-5, 5),
    "y_range": (-5, 5),
    "z_range": (0, 3),
    "runways": [
        {"name": "RWY 09/27", "start": (-2., 0., 0.), "end": (2., 0., 0.), "width": .06},
        {"name": "RWY 18/36", "start": (0., -2., 0.), "end": (0., 2., 0.), "width": .06}
    ],
    "flight_paths": [
        {"id": "APP09", "name": "Approach 09", "start": (-4.5, 0., 1.), "end": (-2., 0., 0.),
         "width": .3, "type": "approach"},
        {"id": "DEP27", "name": "Departure 27", "start": (2., 0., 0.), "end": (4.5, 0., 1.),
         "width": .4, "type": "departure"},
        {"id": "APP18", "name": "Approach 18", "start": (0., -4.5, 1.), "end": (0., -2., 0.),
         "width": .3, "type": "approach"},
        {"id": "DEP36", "name": "Departure 36", "start": (0., 2., 0.), "end": (0., 4.5, 1.),
         "width": .4, "type": "departure"}
    ],
    "sensors": [
        {"id": "radar_main", "position": (.1, -.1, .05), "range": 10., "accuracy": .03},
        {"id": "radar_north", "position": (0., 4., .05), "range": 8., "accuracy": .04},
        {"id": "radar_south", "position": (0., -4., .05), "range": 8., "accuracy": .04},
        {"id": "radar_east", "position": (4., 0., .05), "range": 8., "accuracy": .04},
        {"id": "radar_west", "position": (-4., 0., .05), "range": 8., "accuracy": .04}
    ],
    "airport_reference": {"lat": 40., "lon": -75., "alt": 100}
}

DEFAULT_SPECIES_TO_FAMILY = {
    'Goose': 'Anatidae',
    'Duck': 'Anatidae',
    'Gull': 'Laridae',
    'Hawk': 'Accipitridae',
    'Small Bird': 'Passeriformes',
    'Pigeon/Dove': 'Columbidae',
    'Heron/Egret': 'Ardeidae',
    'Owl': 'Strigidae',
    'Eagle': 'Accipitridae',
    'Vulture': 'Cathartidae',
    'Unknown': 'Unknown',
    'Other Bird': 'Unknown'
}

DEFAULT_SEASONAL_DISTRIBUTIONS = {
    'spring': {
        'species': ['Goose', 'Duck', 'Gull', 'Hawk', 'Small Bird'],
        'probabilities': [0.3, 0.25, 0.2, 0.1, 0.15]
    },
    'fall': {
        'species': ['Goose', 'Duck', 'Gull', 'Hawk', 'Small Bird'],
        'probabilities': [0.3, 0.25, 0.2, 0.1, 0.15]
    },
    'summer': {
        'species': ['Gull', 'Pigeon/Dove', 'Small Bird', 'Hawk', 'Heron/Egret'],
        'probabilities': [0.3, 0.2, 0.25, 0.1, 0.15]
    },
    'winter': {
        'species': ['Goose', 'Duck', 'Gull', 'Owl', 'Hawk'],
        'probabilities': [0.35, 0.3, 0.15, 0.05, 0.15]
    },
    'default': {
        'species': ['Gull', 'Goose', 'Small Bird', 'Hawk', 'Pigeon/Dove'],
        'probabilities': [0.25, 0.25, 0.2, 0.15, 0.15]
    }
}

FORCE_DEBUG_MODULES = ['risk_mapping', 'bird_strike_system']

TEST_SCENARIOS = {
    "low_risk_sparse": {
        "name": "Low Risk - Sparse Birds",
        "description": "Few birds, calm conditions, low activity season",
        "num_birds": 5,
        "season": "Winter",
        "num_iterations": 20,
        "expected_risk_level": "low",
        "parameters": {
            "bird_speed_multiplier": 0.8,
            "proximity_threshold": 2.0,
        },
    },
    "moderate_risk_spring": {
        "name": "Moderate Risk - Spring Migration",
        "description": "Moderate bird count during spring migration",
        "num_birds": 15,
        "season": "Spring",
        "num_iterations": 30,
        "expected_risk_level": "moderate",
        "parameters": {
            "bird_speed_multiplier": 1.2,
            "proximity_threshold": 1.5,
        },
    },
    "high_risk_fall": {
        "name": "High Risk - Fall Migration",
        "description": "High bird density during fall migration period",
        "num_birds": 25,
        "season": "Fall",
        "num_iterations": 30,
        "expected_risk_level": "high",
        "parameters": {
            "bird_speed_multiplier": 1.3,
            "proximity_threshold": 1.0,
        },
    },
    "baseline_nominal": {
        "name": "Baseline - Nominal Conditions",
        "description": "Standard operational conditions for baseline",
        "num_birds": 10,
        "season": "Summer",
        "num_iterations": 30,
        "expected_risk_level": "low-moderate",
        "parameters": {
            "bird_speed_multiplier": 1.0,
            "proximity_threshold": 1.5,
        },
    },
}

SCENARIO_GROUPS = {
    "quick": ["low_risk_sparse", "baseline_nominal", "high_risk_fall"],
    "comprehensive": ["low_risk_sparse", "moderate_risk_spring", "high_risk_fall", "baseline_nominal"],
}

RISK_LEVEL_THRESHOLDS = {
    "low": {"mean": 0.1, "max": 0.25},
    "low-moderate": {"mean": 0.2, "max": 0.40},
    "moderate": {"mean": 0.3, "max": 0.50},
    "high": {"mean": 0.5, "max": 0.80},
}
