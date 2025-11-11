import pandas as pd
from collections import defaultdict
from scripts.constants import (
    FAA_COLUMN_CANDIDATES,
    FAMILY_KEYWORDS,
    MONTH_LABELS,
    SEASON_ORDER,
    DEFAULT_FAA_DATA_PATH,
)
from scripts.utils import resolve_columns, month_to_season
from utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


def analyze_taxonomy(df, cols):
    logger.info("SPECIES & TAXONOMY ANALYSIS")

    taxonomy_summary = {}
    id_col = cols.get('species_id')
    name_col = cols.get('species_name')

    if id_col:
        print(f"\nSpecies ID analysis using column '{id_col}':")
        species_ids = df[id_col].dropna().astype(str)
        print(f"  Non-null IDs: {len(species_ids):,}")
        print(f"  Unique IDs: {species_ids.nunique():,}")

        lengths = species_ids.str.len()
        length_mode = lengths.mode().iloc[0] if not lengths.mode().empty else 'N/A'
        print(f"  Length range: {lengths.min()}–{lengths.max()} (mode={length_mode})")

        first_chars = species_ids.str[0]
        first_char_counts = first_chars.value_counts().head(10)
        for char, count in first_char_counts.items():
            print(f"    '{char}': {count:,} ({count/len(species_ids)*100:.1f}%)")

        pattern_checks = {
            'All letters': species_ids.str.match(r'^[A-Za-z]+$'),
            'All digits': species_ids.str.match(r'^\d+$'),
            'Letter + digits': species_ids.str.match(r'^[A-Za-z]\d+$'),
            'Letters + digits': species_ids.str.match(r'^[A-Za-z]+\d+$'),
            'Mixed': species_ids.str.match(r'^[A-Za-z0-9]+$')
        }
        for label, matches in pattern_checks.items():
            count = matches.sum()
            if count:
                print(f"    {label:<15}: {count:,} ({count/len(species_ids)*100:.1f}%)")

        encoding_schema = None
        if 'B' in first_char_counts and first_char_counts.iloc[0] == first_char_counts['B']:
            encoding_schema = 'CLASS_PREFIX'
        if pattern_checks['Letter + digits'].sum() > len(species_ids) * 0.5:
            encoding_schema = 'CLASS_NUMERIC'

        examples = []
        for char in first_char_counts.head(5).index:
            examples.append((char, species_ids[species_ids.str[0] == char].head(5).tolist()))

        cross_ref = {}
        if name_col:
            id_name_pairs = df[[id_col, name_col]].dropna().drop_duplicates()
            cross_ref['unique_pairs'] = len(id_name_pairs)
            ids_per_name = id_name_pairs.groupby(name_col)[id_col].nunique()
            names_per_id = id_name_pairs.groupby(id_col)[name_col].nunique()
            cross_ref['multi_id_names'] = ids_per_name[ids_per_name > 1].head(3).to_dict()
            cross_ref['multi_name_ids'] = names_per_id[names_per_id > 1].head(3).to_dict()

        taxonomy_summary['encoding'] = {
            'patterns': first_char_counts.to_dict(),
            'examples': examples,
            'encoding_schema': encoding_schema,
            'cross_reference': cross_ref
        }
    else:
        print("Species ID column not found.")

    if name_col:
        print(f"\nSpecies frequency using column '{name_col}':")
        species_counts = df[name_col].value_counts()
        for i, (species, count) in enumerate(species_counts.head(20).items(), 1):
            pct = count / len(df) * 100
            print(f"  {i:2d}. {species[:40]:<40} {count:>6,} ({pct:5.2f}%)")

        family_counts = defaultdict(int)
        species_to_family_map = {}
        for species in df[name_col].dropna():
            species_lower = species.lower()
            assigned = False
            for family, keywords in FAMILY_KEYWORDS.items():
                if any(kw in species_lower for kw in keywords):
                    family_counts[family] += 1
                    species_to_family_map[species] = family
                    assigned = True
                    break
            if not assigned:
                family_counts['Other'] += 1
                species_to_family_map[species] = 'Other'

        print("\nTop inferred families:")
        for family, count in sorted(family_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            pct = count / len(df) * 100
            print(f"    {family:<20} {count:>8,} ({pct:5.2f}%)")

        taxonomy_summary['families'] = {
            'species_counts': species_counts.to_dict(),
            'family_counts': dict(family_counts),
            'species_to_family_map': species_to_family_map
        }
    else:
        print("Species name column not found.")

    return taxonomy_summary


def analyze_temporal_and_quality(df, cols):
    logger.info("TEMPORAL & DATA QUALITY ANALYSIS")

    month_col = cols.get('month')
    damage_col = cols.get('damage')
    month_summary = {}

    if month_col:
        print(f"\nTemporal analysis using '{month_col}':")
        monthly_counts = df[month_col].value_counts().sort_index()
        for month, count in monthly_counts.items():
            if pd.notna(month) and 1 <= month <= 12:
                pct = count / len(df) * 100
                month_name = MONTH_LABELS[int(month) - 1]
                bar = '#' * int(pct)
                print(f"  {month_name} (M{int(month):02d}): {count:>7,} ({pct:5.2f}%) {bar}")

        df['Season'] = df[month_col].apply(month_to_season)
        seasonal_counts = df['Season'].value_counts()
        print("\nSeasonal distribution:")
        for season in SEASON_ORDER:
            if season in seasonal_counts:
                count = seasonal_counts[season]
                pct = count / len(df) * 100
                print(f"  {season:<10} {count:>8,} ({pct:5.2f}%)")

        damage_by_season = {}
        if damage_col:
            print("\nDamage rate by season:")
            for season in SEASON_ORDER:
                season_data = df[df['Season'] == season]
                if len(season_data) > 0:
                    damage_rate = season_data[damage_col].notna().sum() / len(season_data) * 100
                    damage_by_season[season] = damage_rate
                    print(f"  {season:<10} {damage_rate:5.2f}% incidents with damage reported")

        month_summary = {
            'monthly_counts': monthly_counts.to_dict(),
            'seasonal_counts': seasonal_counts.to_dict(),
            'damage_rate_by_season': damage_by_season
        }
    else:
        print("Month column not found.")

    print("\nDataset overview:")
    print(f"  Total records: {len(df):,}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    missing = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
    print("\nMissing values (top 20):")
    for col in missing_pct.head(20).index:
        count = missing[col]
        pct = missing_pct[col]
        if count > 0:
            bar = '-' * int(pct / 5)
            print(f"  {col[:40]:<40} {count:>8,} ({pct:5.1f}%) {bar}")

    key_column_stats = {}
    for label, col_name in {
        'Species Name': cols.get('species_name'),
        'Species ID': cols.get('species_id'),
        'Incident Month': month_col,
        'Aircraft Damage': damage_col,
        'Incident Year': cols.get('year'),
        'Airport': cols.get('airport'),
        'State': cols.get('state')
    }.items():
        if col_name and col_name in df.columns:
            completeness = (1 - df[col_name].isnull().sum() / len(df)) * 100
            status = 'OK' if completeness > 90 else 'WARN' if completeness > 70 else 'POOR'
            key_column_stats[label] = {'column': col_name, 'completeness': completeness, 'status': status}
            print(f"  {status:4} {label:<25} {completeness:5.1f}% complete")

    quality_summary = {
        'total_records': len(df),
        'missing_summary': missing.to_dict(),
        'key_columns': key_column_stats
    }

    return {'temporal': month_summary, 'quality': quality_summary}


def generate_report(taxonomy_results, tq_results):
    logger.info("ANALYSIS SUMMARY")
    print()

    encoding = taxonomy_results.get('encoding')
    if encoding and encoding.get('encoding_schema'):
        print(f"Species ID Encoding Schema: {encoding['encoding_schema']}")

    families = taxonomy_results.get('families')
    if families:
        total_families = len(families['family_counts'])
        print(f"Taxonomic Families Identified: {total_families}")
        family_counts = families['family_counts']
        if family_counts:
            top_families = sorted(family_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print("Top 5 Families by Strike Count:")
            for family, count in top_families:
                print(f"  {family}: {count:,}")

    temporal = tq_results.get('temporal', {})
    if temporal.get('seasonal_counts'):
        seasonal_counts = temporal['seasonal_counts']
        peak_season = max(seasonal_counts.items(), key=lambda x: x[1])[0]
        peak_count = seasonal_counts[peak_season]
        print(f"\nPeak Strike Season: {peak_season} ({peak_count:,} strikes)")
        print("Seasonal Distribution:")
        for season in SEASON_ORDER:
            if season in seasonal_counts:
                print(f"  {season}: {seasonal_counts[season]:,}")

    quality = tq_results.get('quality', {})
    total_records = quality.get('total_records')
    if total_records is not None:
        print(f"\nTotal Records Analyzed: {total_records:,}")


def main():
    configure_logging()
    logger.info("Loading FAA wildlife strike data...")
    df = pd.read_csv(DEFAULT_FAA_DATA_PATH, low_memory=False)
    print(f"Loaded {len(df):,} records with {len(df.columns)} columns\n")

    cols = resolve_columns(df, FAA_COLUMN_CANDIDATES)
    taxonomy_results = analyze_taxonomy(df, cols)
    tq_results = analyze_temporal_and_quality(df, cols)
    generate_report(taxonomy_results, tq_results)


if __name__ == "__main__":
    main()
