REPORT_SEPARATOR = "=" * 80
STEP_SEPARATOR = "-" * 80

DEFAULT_FAA_DATA_PATH = 'data/raw/wildlife_strikes.csv'
DEFAULT_MIGRATION_DATA_PATH = 'bird_migration.csv'

FAA_COLUMN_CANDIDATES = {
    'species_id': ['Species ID', 'SPECIES_ID', 'species_id', 'SpeciesID'],
    'species_name': ['Species Name', 'SPECIES', 'species_name', 'SpeciesName'],
    'month': ['Incident Month', 'INCIDENT_MONTH', 'month', 'Month'],
    'damage': ['Aircraft Damage', 'DAMAGE', 'damage'],
    'airport': ['Airport', 'AIRPORT', 'airport_name'],
    'state': ['Airport State', 'State', 'STATE'],
    'year': ['Incident Year', 'INCIDENT_YEAR', 'year'],
}

FAMILY_KEYWORDS = {
    'Anatidae': ['duck', 'goose', 'geese', 'swan', 'mallard', 'teal'],
    'Laridae': ['gull', 'tern'],
    'Columbidae': ['dove', 'pigeon'],
    'Accipitridae': ['hawk', 'eagle', 'kite'],
    'Falconidae': ['falcon', 'kestrel'],
    'Strigidae': ['owl'],
    'Corvidae': ['crow', 'raven', 'jay'],
    'Cathartidae': ['vulture', 'buzzard'],
    'Passeridae': ['sparrow', 'finch', 'starling', 'blackbird'],
}

EXTENDED_FAMILY_KEYWORDS = {
    'Anatidae': ['duck', 'goose', 'geese', 'swan', 'mallard', 'teal', 'merganser'],
    'Laridae': ['gull', 'tern'],
    'Columbidae': ['dove', 'pigeon'],
    'Accipitridae': ['hawk', 'eagle', 'kite', 'harrier'],
    'Falconidae': ['falcon', 'kestrel'],
    'Strigidae': ['owl'],
    'Corvidae': ['crow', 'raven', 'jay', 'magpie'],
    'Cathartidae': ['vulture', 'condor'],
    'Icteridae': ['blackbird', 'grackle', 'cowbird', 'oriole'],
    'Passeridae': ['sparrow'],
    'Sturnidae': ['starling'],
    'Fringillidae': ['finch'],
    'Ardeidae': ['heron', 'egret'],
    'Pelecanidae': ['pelican'],
    'Phalacrocoracidae': ['cormorant'],
}

MIGRATION_COLUMN_CANDIDATES = {
    'routes': ['Migration routes', 'migration_routes', 'routes', 'ROUTES'],
    'gps_x': ['GPS_xx', 'gps_xx', 'GPS_X', 'longitude'],
    'gps_y': ['GPS_yy', 'gps_yy', 'GPS_Y', 'latitude'],
    'family': ['Bird families', 'bird_families', 'family', 'FAMILY'],
    'start_month': ['Migration start month', 'migration_start_month', 'start_month']
}

ROUTE_DIRECTION_MAP = {
    1: {'tendency': 0.15, 'direction': 'North', 'season': 'Spring'},
    2: {'tendency': 0.12, 'direction': 'North-Northeast', 'season': 'Spring'},
    3: {'tendency': 0.18, 'direction': 'Northeast', 'season': 'Spring'},
    4: {'tendency': 0.10, 'direction': 'Northwest', 'season': 'Spring'},
    5: {'tendency': 0.14, 'direction': 'North', 'season': 'Spring'},
    6: {'tendency': -0.15, 'direction': 'South', 'season': 'Fall'},
    7: {'tendency': -0.12, 'direction': 'South-Southeast', 'season': 'Fall'},
    8: {'tendency': -0.18, 'direction': 'Southeast', 'season': 'Fall'},
    9: {'tendency': -0.10, 'direction': 'Southwest', 'season': 'Fall'},
    10: {'tendency': -0.14, 'direction': 'South', 'season': 'Fall'},
    11: {'tendency': 0.05, 'direction': 'East', 'season': 'Spring'},
    12: {'tendency': -0.05, 'direction': 'West', 'season': 'Fall'},
    13: {'tendency': 0.0, 'direction': 'Resident', 'season': 'Winter'},
    14: {'tendency': 0.0, 'direction': 'Resident', 'season': 'Summer'},
    15: {'tendency': 0.20, 'direction': 'Far North', 'season': 'Spring'},
    16: {'tendency': -0.20, 'direction': 'Far South', 'season': 'Fall'},
    17: {'tendency': 0.08, 'direction': 'Coastal North', 'season': 'Spring'},
    18: {'tendency': -0.08, 'direction': 'Coastal South', 'season': 'Fall'},
    19: {'tendency': 0.12, 'direction': 'Interior North', 'season': 'Spring'},
    20: {'tendency': -0.12, 'direction': 'Interior South', 'season': 'Fall'},
    21: {'tendency': 0.0, 'direction': 'Mixed/Partial', 'season': 'Various'}
}

SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Fall']
SEASON_ORDER_LOWER = [season.lower() for season in SEASON_ORDER]

MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

SEASON_MONTHS = {
    'winter': [12, 1, 2],
    'spring': [3, 4, 5],
    'summer': [6, 7, 8],
    'fall': [9, 10, 11]
}

SEASON_BASELINE_ACTIVITY = {
    'winter': 0.7,
    'spring': 1.2,
    'summer': 1.0,
    'fall': 1.2
}

DEFAULT_SEASONAL_FACTORS = {
    'winter': {'tendency': 0.0, 'activity': 0.7},
    'spring': {'tendency': 0.1, 'activity': 1.2},
    'summer': {'tendency': 0.0, 'activity': 1.0},
    'fall': {'tendency': -0.1, 'activity': 1.2},
}

BETA_PRIOR_ALPHA = 1
BETA_PRIOR_BETA = 9

MIGRATION_ANALYSIS_PARAMS = {
    'max_families_to_display': 15,
    'max_route_codes_to_display': 10,
    'gps_tendency_clip_min': -0.3,
    'gps_tendency_clip_max': 0.3,
    'gps_tendency_divisor': 100.0,
    'gps_activity_min_multiplier': 0.5,
    'gps_activity_min': 0.5,
    'gps_activity_max': 2.0,
    'route_activity_base': 1.0,
    'route_activity_multiplier': 0.5,
    'gps_tendency_weight': 0.7,
    'gps_from_gps_weight': 0.3,
    'gps_displacement_divisor': 50.0,
}

SEASONAL_FACTORS_PRELIM_FILE = 'seasonal_factors_preliminary.json'
