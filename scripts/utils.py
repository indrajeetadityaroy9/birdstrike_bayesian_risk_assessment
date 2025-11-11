import pandas as pd
from scripts.constants import SEASON_MONTHS

def resolve_columns(df, column_candidates):
    resolved = {}
    for key, candidates in column_candidates.items():
        for col in candidates:
            if col in df.columns:
                resolved[key] = col
                break
    return resolved

def month_to_season(month):
    season = _season_from_month(month)
    return season.capitalize() if season else 'Unknown'

def month_to_season_lower(month):
    season = _season_from_month(month)
    return season if season else 'Unknown'

def categorize_species(species_name):
    if pd.isna(species_name):
        return 'Unknown'

    name_lower = str(species_name).lower()

    if any(kw in name_lower for kw in ['gull', 'tern']):
        return 'Gull'
    elif any(kw in name_lower for kw in ['goose', 'geese']):
        return 'Goose'
    elif 'duck' in name_lower:
        return 'Duck'
    elif any(kw in name_lower for kw in ['hawk', 'eagle', 'kite']):
        return 'Hawk'
    elif any(kw in name_lower for kw in ['dove', 'pigeon']):
        return 'Pigeon'
    elif any(kw in name_lower for kw in ['sparrow', 'starling', 'finch', 'swallow']):
        return 'Small Bird'
    elif any(kw in name_lower for kw in ['heron', 'egret']):
        return 'Heron'
    elif 'owl' in name_lower:
        return 'Owl'
    elif 'crow' in name_lower or 'raven' in name_lower:
        return 'Crow'
    else:
        return 'Other'

def _season_from_month(month):
    if pd.isna(month):
        return None
    try:
        month = int(month)
    except (TypeError, ValueError):
        return None

    for season, season_months in SEASON_MONTHS.items():
        if month in season_months:
            return season
    return None
