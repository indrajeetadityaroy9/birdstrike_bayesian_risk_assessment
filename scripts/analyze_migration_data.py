import json
import pandas as pd
import numpy as np
from collections import defaultdict

from scripts.constants import (
    ROUTE_DIRECTION_MAP,
    MIGRATION_COLUMN_CANDIDATES,
    SEASON_BASELINE_ACTIVITY,
    MIGRATION_ANALYSIS_PARAMS,
    SEASON_ORDER_LOWER,
    DEFAULT_MIGRATION_DATA_PATH,
    SEASONAL_FACTORS_PRELIM_FILE,
)

from scripts.utils import (
    resolve_columns,
    month_to_season_lower,
)
from utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


def _print_route_distribution(routes_col, route_counts):
    print(f"\nRoute Code Distribution:")
    print(f"  Total non-null routes: {len(route_counts):,}")
    print(f"  Unique route codes: {route_counts.nunique()}")
    print(f"\n  Route Code Frequencies:")
    for route_code, count in route_counts.items():
        pct = count / len(route_counts) * 100
        print(f"Route {route_code:>2}: {count:>6,} ({pct:5.2f}%)")

def _print_route_mapping(route_counts):
    for route_code in sorted(route_counts.index):
        info = ROUTE_DIRECTION_MAP.get(route_code)
        if info:
            print(f"Route {route_code:>2} → {info['direction']:<20} "f"(tendency={info['tendency']:+.2f}, peak={info['season']})")
        else:
            print(f"Route {route_code:>2} → Unknown (needs classification)")


def _validate_route_gps_consistency(df, routes_col, gps_y_col):
    validation = []
    print("\nROUTE ↔ GPS CONSISTENCY VALIDATION")
    for route_code, info in ROUTE_DIRECTION_MAP.items():
        route_data = df[df[routes_col] == route_code][[gps_y_col]].dropna()
        if len(route_data) > 1:
            y_mean = route_data[gps_y_col].mean()
            gps_sign = 'positive' if y_mean > 0 else 'negative' if y_mean < 0 else 'neutral'
            expected = info['tendency']
            expected_sign = 'positive' if expected > 0 else 'negative' if expected < 0 else 'neutral'
            match = gps_sign == expected_sign
            label = '[OK]' if match else '[WARN]'
            print(f"  {label} Route {route_code:>2} ({info['direction']:<20}): "
                  f"Expected={expected:+.2f}, GPS mean Y={y_mean:+6.2f} ({gps_sign})")
            validation.append({'route': route_code, 'match': match})

    matched = sum(1 for r in validation if r['match'])
    total = len(validation)
    if total:
        print(f"\n  Overall consistency: {matched}/{total} ({matched/total*100:.1f}%)")

    return validation


def analyze_routes(df, cols):
    routes_col = cols.get('routes')
    if routes_col is None:
        print("Migration routes column not found")
        print(f"  Available columns: {df.columns.tolist()}")
        return {}

    logger.info("MIGRATION ROUTE CODE ANALYSIS")
    print(f"Using column: '{routes_col}'")

    route_values = df[routes_col].dropna()
    route_counts = route_values.value_counts().sort_index()
    _print_route_distribution(routes_col, route_counts)
    _print_route_mapping(route_counts)

    validation = []
    gps_y_col = cols.get('gps_y')
    if gps_y_col:
        validation = _validate_route_gps_consistency(df, routes_col, gps_y_col)

    return {
        'route_counts': route_counts.to_dict(),
        'route_direction_map': ROUTE_DIRECTION_MAP,
        'validation': validation
    }


def _print_gps_coverage(gps_data, gps_x_col, gps_y_col, total_records):
    print(f"\nGPS Data Coverage:")
    print(f"  Records with GPS: {len(gps_data):,} ({len(gps_data)/total_records*100:.1f}%)")
    print(f"  X range: [{gps_data[gps_x_col].min():.2f}, {gps_data[gps_x_col].max():.2f}]")
    print(f"  Y range: [{gps_data[gps_y_col].min():.2f}, {gps_data[gps_y_col].max():.2f}]")


def _calculate_family_movement_metrics(family_data, gps_x_col, gps_y_col):
    x_displ = family_data[gps_x_col].max() - family_data[gps_x_col].min()
    y_displ = family_data[gps_y_col].max() - family_data[gps_y_col].min()

    tendency = np.clip(
        y_displ / MIGRATION_ANALYSIS_PARAMS['gps_tendency_divisor'],
        MIGRATION_ANALYSIS_PARAMS['gps_tendency_clip_min'],
        MIGRATION_ANALYSIS_PARAMS['gps_tendency_clip_max']
    )

    total_range = np.sqrt(x_displ**2 + y_displ**2)
    activity = np.clip(
        MIGRATION_ANALYSIS_PARAMS['route_activity_base'] +
        (total_range / MIGRATION_ANALYSIS_PARAMS['gps_displacement_divisor']),
        MIGRATION_ANALYSIS_PARAMS['gps_activity_min'],
        MIGRATION_ANALYSIS_PARAMS['gps_activity_max']
    )

    return tendency, activity


def _print_family_movement_patterns(df, family_col, gps_x_col, gps_y_col):
    print(f"\nMovement Patterns by Family:")
    max_families = MIGRATION_ANALYSIS_PARAMS['max_families_to_display']
    for family in df[family_col].dropna().unique()[:max_families]:
        family_data = df[df[family_col] == family][[gps_x_col, gps_y_col]].dropna()
        if len(family_data) > 1:
            tendency, activity = _calculate_family_movement_metrics(
                family_data, gps_x_col, gps_y_col
            )
            print(f"  {family[:30]:<30}")
            print(f"    GPS points: {len(family_data):>4}, "
                  f"Y displacement: {family_data[gps_y_col].max() - family_data[gps_y_col].min():+7.2f}, "
                  f"Tendency: {tendency:+.3f}, Activity: {activity:.2f}x")


def analyze_gps_movements(df, cols):
    gps_x_col = cols.get('gps_x')
    gps_y_col = cols.get('gps_y')
    if not gps_x_col or not gps_y_col:
        print("\nGPS coordinate columns not found")
        print(f"  Available columns: {df.columns.tolist()}")
        return {}

    logger.info("GPS MOVEMENT VECTOR ANALYSIS")
    print(f"Using GPS columns: '{gps_x_col}', '{gps_y_col}'")

    gps_data = df[[gps_x_col, gps_y_col]].dropna()
    _print_gps_coverage(gps_data, gps_x_col, gps_y_col, len(df))

    family_col = cols.get('family')
    if family_col:
        _print_family_movement_patterns(df, family_col, gps_x_col, gps_y_col)

    return {'gps_coverage': len(gps_data) / len(df), 'has_gps': True}


def _extract_route_tendency_and_activity(season_data, routes_col, route_direction_map, family_data):
    tendency_from_routes = 0.0
    activity_from_routes = MIGRATION_ANALYSIS_PARAMS['route_activity_base']

    if routes_col and routes_col in season_data.columns:
        season_routes = season_data[routes_col].dropna()
        if len(season_routes) > 0:
            most_common_route = season_routes.mode()
            if len(most_common_route) > 0:
                most_common_route = most_common_route.iloc[0]
                if most_common_route in route_direction_map:
                    route_info = route_direction_map[most_common_route]
                    tendency_from_routes = route_info['tendency']
                    route_freq = len(season_routes) / len(family_data)
                    activity_from_routes = (
                        MIGRATION_ANALYSIS_PARAMS['route_activity_base'] +
                        route_freq * MIGRATION_ANALYSIS_PARAMS['route_activity_multiplier']
                    )

    return tendency_from_routes, activity_from_routes


def _extract_gps_tendency(season_data, gps_x_col, gps_y_col):
    if gps_x_col and gps_y_col:
        if gps_x_col in season_data.columns and gps_y_col in season_data.columns:
            season_gps = season_data[[gps_x_col, gps_y_col]].dropna()
            if len(season_gps) > 1:
                y_displ = season_gps[gps_y_col].max() - season_gps[gps_y_col].min()
                return np.clip(
                    y_displ / MIGRATION_ANALYSIS_PARAMS['gps_tendency_divisor'],
                    MIGRATION_ANALYSIS_PARAMS['gps_tendency_clip_min'],
                    MIGRATION_ANALYSIS_PARAMS['gps_tendency_clip_max']
                )
    return None


def _calculate_final_seasonal_metrics(tendency_from_routes, tendency_from_gps, activity_from_routes, season):
    final_tendency = tendency_from_routes
    if tendency_from_gps is not None:
        final_tendency = (
            MIGRATION_ANALYSIS_PARAMS['gps_tendency_weight'] * tendency_from_routes +
            MIGRATION_ANALYSIS_PARAMS['gps_from_gps_weight'] * tendency_from_gps
        )

    final_activity = activity_from_routes * SEASON_BASELINE_ACTIVITY.get(season, 1.0)

    return final_tendency, final_activity


def derive_seasonal_factors(df, route_direction_map, cols):
    family_col = cols.get('family')
    if not family_col:
        print("Family column not found")
        return {}

    start_month_col = cols.get('start_month')
    routes_col = cols.get('routes')
    gps_x_col = cols.get('gps_x')
    gps_y_col = cols.get('gps_y')

    logger.info("SEASONAL FACTOR DERIVATION (HYBRID APPROACH)")
    print(f"Using columns: Family='{family_col}', Start Month='{start_month_col}', "
          f"Routes='{routes_col}'")

    seasonal_factors = defaultdict(lambda: defaultdict(dict))
    print(f"\nDeriving Factors per Family × Season:")

    families = df[family_col].dropna().unique()
    max_families = MIGRATION_ANALYSIS_PARAMS['max_families_to_display']

    for family in sorted(families)[:max_families]:
        family_data = df[df[family_col] == family]
        if start_month_col not in family_data.columns:
            continue

        family_data = family_data.copy()
        family_data['season'] = family_data[start_month_col].apply(month_to_season_lower)

        for season in SEASON_ORDER_LOWER:
            season_data = family_data[family_data['season'] == season]
            if len(season_data) == 0:
                continue

            tendency_from_routes, activity_from_routes = _extract_route_tendency_and_activity(
                season_data, routes_col, route_direction_map, family_data
            )

            tendency_from_gps = _extract_gps_tendency(season_data, gps_x_col, gps_y_col)

            final_tendency, final_activity = _calculate_final_seasonal_metrics(
                tendency_from_routes, tendency_from_gps, activity_from_routes, season
            )

            seasonal_factors[family][season] = {
                'tendency': round(final_tendency, 3),
                'activity': round(final_activity, 3)
            }
            print(f"  {family[:25]:<25} × {season:<6} → "
                  f"tendency={final_tendency:+.3f}, activity={final_activity:.3f}")

    return dict(seasonal_factors)


def _report_route_analysis(routes):
    if not routes or 'route_counts' not in routes:
        return

    route_counts = routes['route_counts']
    print(f"Migration Routes Analyzed:")
    print(f"  Total unique routes: {len(route_counts)}")
    print(f"  Total route records: {sum(route_counts.values()):,}")

    if route_counts:
        most_common = max(route_counts.items(), key=lambda x: x[1])
        least_common = min(route_counts.items(), key=lambda x: x[1])
        print(f"  Most common route: {most_common[0]} ({most_common[1]:,} records)")
        print(f"  Least common route: {least_common[0]} ({least_common[1]:,} records)")


def _report_route_gps_validation(routes):
    if not routes or not routes.get('validation'):
        return

    validation = routes['validation']
    valid_count = sum(1 for r in validation if r['match'])
    total = len(validation)

    if total:
        consistency_pct = valid_count / total * 100
        print(f"\nRoute-GPS Consistency Validation:")
        print(f"  Routes validated: {total}")
        print(f"  Consistent routes: {valid_count} ({consistency_pct:.1f}%)")
        print(f"  Inconsistent routes: {total - valid_count} ({100 - consistency_pct:.1f}%)")


def _report_gps_coverage(gps):
    if not gps or 'gps_coverage' not in gps:
        return

    print(f"\nGPS Data Coverage:")
    print(f"  Records with GPS coordinates: {gps['gps_coverage']*100:.1f}%")


def _calculate_seasonal_factor_statistics(factors):
    all_tendencies = []
    all_activities = []
    season_counts = {season: 0 for season in SEASON_ORDER_LOWER}

    for seasons in factors.values():
        for season, values in seasons.items():
            if 'tendency' in values:
                all_tendencies.append(values['tendency'])
            if 'activity' in values:
                all_activities.append(values['activity'])
            if season in season_counts:
                season_counts[season] += 1

    return all_tendencies, all_activities, season_counts


def _report_seasonal_factors(factors):
    if not factors:
        return

    print(f"\nSeasonal Factors Derived:")
    print(f"  Total families analyzed: {len(factors)}")

    all_tendencies, all_activities, season_counts = _calculate_seasonal_factor_statistics(factors)

    if all_tendencies:
        print(f"  Tendency range: [{min(all_tendencies):+.3f}, {max(all_tendencies):+.3f}]")
        print(f"  Mean tendency: {np.mean(all_tendencies):+.3f}")

    if all_activities:
        print(f"  Activity range: [{min(all_activities):.3f}, {max(all_activities):.3f}]")
        print(f"  Mean activity: {np.mean(all_activities):.3f}")

    print(f"  Factors per season: {dict(season_counts)}")

    with open(SEASONAL_FACTORS_PRELIM_FILE, 'w') as f:
        json.dump(factors, f, indent=2)
    print(f"\nSeasonal factors saved to: {SEASONAL_FACTORS_PRELIM_FILE}")


def generate_report(analysis_results):
    logger.info("ANALYSIS SUMMARY")
    print()

    routes = analysis_results.get('routes')
    if routes:
        _report_route_analysis(routes)
        _report_route_gps_validation(routes)

    gps = analysis_results.get('gps')
    _report_gps_coverage(gps)

    factors = analysis_results.get('seasonal_factors')
    _report_seasonal_factors(factors)


def _load_migration_data(filepath=DEFAULT_MIGRATION_DATA_PATH):
    print(f"Loading migration data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded {len(df):,} records with {len(df.columns)} columns")
    cols = resolve_columns(df, MIGRATION_COLUMN_CANDIDATES)
    return df, cols


def _run_all_analyses(df, cols):
    analysis_results = {}
    analysis_results['routes'] = analyze_routes(df, cols)
    analysis_results['gps'] = analyze_gps_movements(df, cols)
    route_map = analysis_results['routes'].get('route_direction_map', ROUTE_DIRECTION_MAP)
    analysis_results['seasonal_factors'] = derive_seasonal_factors(df, route_map, cols)

    return analysis_results


def main():
    configure_logging()
    df, cols = _load_migration_data()
    analysis_results = _run_all_analyses(df, cols)
    generate_report(analysis_results)


if __name__ == "__main__":
    main()
