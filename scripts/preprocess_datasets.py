import pandas as pd
import numpy as np
from scipy.stats import beta
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
import json
import warnings
from scripts.constants import (
    EXTENDED_FAMILY_KEYWORDS,
    ROUTE_DIRECTION_MAP,
    DEFAULT_SEASONAL_FACTORS,
    BETA_PRIOR_ALPHA,
    BETA_PRIOR_BETA,
    SEASON_MONTHS,
    DEFAULT_FAA_DATA_PATH,
    DEFAULT_MIGRATION_DATA_PATH,
)
from scripts.utils import categorize_species, month_to_season
from utils.logger import configure_logging, get_logger

logger = get_logger(__name__)

warnings.filterwarnings('ignore')

class DataPreprocessor:

    def __init__(self, faa_path=DEFAULT_FAA_DATA_PATH, migration_path=DEFAULT_MIGRATION_DATA_PATH):
        self.faa_path = faa_path
        self.migration_path = migration_path

        self.beta_prior_alpha = BETA_PRIOR_ALPHA
        self.beta_prior_beta = BETA_PRIOR_BETA

    def ensure_season_column(self, df):
        if 'Season' in df.columns:
            return df['Season']

        if 'Incident Month' in df.columns:
            df['Season'] = df['Incident Month'].apply(month_to_season)
            return df['Season']

        if 'IncidentDate' in df.columns and 'Season' not in df.columns:
            df['Incident Month'] = df['IncidentDate'].dt.month
            df['Season'] = df['Incident Month'].apply(month_to_season)
            return df['Season']

        return None

    def aggregate_damage_stats(self, faa_data, group_col, skip_unknown=True):
        if 'DamageSeverity' not in faa_data.columns or group_col not in faa_data.columns:
            return {}

        stats = {}
        grouped = faa_data.groupby(group_col)
        for key, group in grouped:
            if pd.isna(key):
                continue
            if skip_unknown and key == 'Unknown':
                continue
            strike_count = len(group)
            damage_count = group['DamageSeverity'].sum()
            stats[key] = {
                'n_strikes': int(strike_count),
                'n_damage': int(damage_count)
            }
        return stats

    def run(self):
        logger.info("STEP 1: Load and Clean FAA Wildlife Strike Data")
        faa_data = self.load_and_clean_faa_data()

        logger.info("STEP 2: Load and Process Migration Data")
        print(f"Loading migration data from {self.migration_path}...")
        migration_data = pd.read_csv(self.migration_path, low_memory=False)
        print(f"Loaded {len(migration_data):,} records with {len(migration_data.columns)} columns")

        logger.info("STEP 3: Extract Species Taxonomy")
        species_taxonomy = self.extract_species_taxonomy(faa_data, migration_data)

        logger.info("STEP 4: Derive Seasonal Behavioral Factors")
        seasonal_factors = self.derive_seasonal_factors(migration_data, species_taxonomy)

        logger.info("STEP 5: Calculate Seasonal Species Distributions")
        species_distributions = self.calculate_species_distributions(faa_data)

        logger.info("STEP 6: Compute Spatial Risk Profiles")
        spatial_profiles = self.compute_spatial_risk_profiles(faa_data)

        logger.info("STEP 7: Compute Beta-Binomial Damage Posteriors")
        damage_posteriors = self.compute_damage_posteriors(faa_data, species_taxonomy)

        logger.info("STEP 8: Save All Output Files")
        self.save_outputs(faa_data, seasonal_factors, species_distributions, spatial_profiles, species_taxonomy, damage_posteriors)

    def load_and_clean_faa_data(self):
        print(f"Loading FAA data from {self.faa_path}...")
        df = pd.read_csv(self.faa_path, low_memory=False)
        print(f"Loaded {len(df):,} records with {len(df.columns)} columns")

        print("\n  Applying imputation (median numerical, iterative categorical)...")

        numerical_cols = ['Incident Year', 'Incident Month', 'Incident Day', 'Height', 'Speed']
        numerical_cols = [c for c in numerical_cols if c in df.columns]

        if numerical_cols:
            num_imputer = SimpleImputer(strategy='median')
            df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
            print(f"  Imputed {len(numerical_cols)} numerical columns")

        categorical_cols = ['Species Name', 'Species ID', 'State', 'Airport']
        categorical_cols = [c for c in categorical_cols if c in df.columns]

        if categorical_cols:
            for col in categorical_cols:
                df[col] = df[col].fillna('Unknown').astype(str)

            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            encoded = encoder.fit_transform(df[categorical_cols])

            iter_imputer = IterativeImputer(max_iter=10, random_state=42)
            imputed = iter_imputer.fit_transform(encoded)

            imputed_rounded = np.round(imputed).astype(int)
            imputed_rounded = np.clip(imputed_rounded, 0, None)

            for i, col in enumerate(categorical_cols):
                try:
                    df[col] = encoder.inverse_transform(imputed_rounded[:, [i]]).flatten()
                except:
                    pass

            print(f"  Imputed {len(categorical_cols)} categorical columns")

        print("\n  Creating derived columns...")

        if 'Aircraft Damage' in df.columns:
            df['DamageSeverity'] = df['Aircraft Damage'].notna().astype(int)
            print(f"  DamageSeverity: {df['DamageSeverity'].sum():,} damaging strikes")

        if 'Species ID' in df.columns:
            df['SpeciesIDClass'] = df['Species ID'].astype(str).str[0].map({
                'B': 'Birds',
                'M': 'Mammals',
                'R': 'Reptiles'
            }).fillna('Unknown')

            df['FamilyGroup'] = 'Unknown'

        if 'Species Name' in df.columns:
            df['SpeciesGroup'] = df['Species Name'].apply(categorize_species)

        if all(c in df.columns for c in ['Incident Year', 'Incident Month', 'Incident Day']):
            try:
                df['IncidentDate'] = pd.to_datetime(
                    df[['Incident Year', 'Incident Month', 'Incident Day']].rename(
                        columns={'Incident Year': 'year', 'Incident Month': 'month', 'Incident Day': 'day'}),
                    errors='coerce'
                )
                print(f"  IncidentDate created ({df['IncidentDate'].notna().sum():,} valid dates)")
            except:
                pass

        season_series = self.ensure_season_column(df)
        if season_series is not None:
            print(f"  Season derived from month")

        print(f"\nFAA data cleaned: {len(df):,} records ready")
        return df

    def extract_species_taxonomy(self, faa_data, migration_data):
        print("Extracting species taxonomy mappings...")

        taxonomy = {
            'species_to_family': {},
            'family_list': [],
            'species_list': []
        }

        migration_families = set()
        if 'Bird families' in migration_data.columns:
            migration_families = set(migration_data['Bird families'].dropna().unique())
            print(f"  Found {len(migration_families)} families in migration data")

        family_keywords = EXTENDED_FAMILY_KEYWORDS

        if 'Species Name' in faa_data.columns:
            for species_name in faa_data['Species Name'].dropna().unique():
                species_lower = species_name.lower()
                matched = False

                for family, keywords in family_keywords.items():
                    if any(kw in species_lower for kw in keywords):
                        taxonomy['species_to_family'][species_name] = family
                        matched = True
                        break

                if not matched:
                    for mig_family in migration_families:
                        if mig_family.lower() in species_lower or species_lower in mig_family.lower():
                            taxonomy['species_to_family'][species_name] = mig_family
                            matched = True
                            break

                if not matched:
                    taxonomy['species_to_family'][species_name] = 'Other'

        taxonomy['family_list'] = sorted(set(taxonomy['species_to_family'].values()))
        taxonomy['species_list'] = sorted(taxonomy['species_to_family'].keys())

        family_to_species = {}
        for species, family in taxonomy['species_to_family'].items():
            if family not in family_to_species:
                family_to_species[family] = []
            family_to_species[family].append(species)

        taxonomy['family_to_species'] = family_to_species

        print(f"  Mapped {len(taxonomy['species_to_family'])} species to {len(taxonomy['family_list'])} families")

        if 'Species Name' in faa_data.columns:
            faa_data['FamilyGroup'] = faa_data['Species Name'].map(taxonomy['species_to_family']).fillna('Other')

        return taxonomy

    def derive_seasonal_factors(self, migration_data, taxonomy):
        print("Deriving seasonal factors (hybrid routes + GPS)...")

        seasonal_factors = DEFAULT_SEASONAL_FACTORS

        if migration_data.empty:
            print("  Using default factors (no migration data)")
            result = {}
            for family in taxonomy['family_list']:
                result[family] = seasonal_factors.copy()
            return result

        route_map = ROUTE_DIRECTION_MAP

        family_col = 'Bird families' if 'Bird families' in migration_data.columns else None
        routes_col = 'Migration routes' if 'Migration routes' in migration_data.columns else None
        start_month_col = 'Migration start month' if 'Migration start month' in migration_data.columns else None

        if not family_col:
            print("  Family column not found, using defaults")
            result = {}
            for family in taxonomy['family_list']:
                result[family] = seasonal_factors.copy()
            return result

        result = {}

        for family in migration_data[family_col].unique():
            if pd.isna(family):
                continue

            family_data = migration_data[migration_data[family_col] == family]
            result[family] = {}

            for season_name in ['winter', 'spring', 'summer', 'fall']:
                tendency = seasonal_factors[season_name]['tendency']
                activity = seasonal_factors[season_name]['activity']

                if start_month_col:
                    season_data = family_data[family_data[start_month_col].isin(SEASON_MONTHS[season_name])]

                    if len(season_data) > 0:
                        if routes_col and routes_col in season_data.columns:
                            routes = season_data[routes_col].dropna()
                            if len(routes) > 0:
                                common_route = routes.mode().iloc[0] if len(routes.mode()) > 0 else None

                                if common_route in route_map:
                                    route_info = route_map[common_route]
                                    tendency = route_info['tendency']

                                migration_intensity = len(routes) / len(family_data)
                                activity *= (1.0 + migration_intensity * 0.3)

                result[family][season_name] = {
                    'tendency': round(np.clip(tendency, -0.3, 0.3), 3),
                    'activity': round(np.clip(activity, 0.5, 2.0), 3)
                }

        for family in taxonomy['family_list']:
            if family not in result:
                result[family] = seasonal_factors.copy()

        print(f"  Derived factors for {len(result)} families x 4 seasons")
        return result

    def calculate_species_distributions(self, faa_data):
        print("Calculating seasonal species distributions...")

        distributions = {
            'spring': {},
            'summer': {},
            'fall': {},
            'winter': {}
        }

        season_series = self.ensure_season_column(faa_data)
        if season_series is None or 'SpeciesGroup' not in faa_data.columns:
            print("  Missing Season or SpeciesGroup columns, using defaults")
            return distributions

        for season in distributions.keys():
            season_label = season.capitalize()
            season_data = faa_data[faa_data['Season'] == season_label]

            if len(season_data) > 0:
                species_counts = season_data['SpeciesGroup'].value_counts()
                total = species_counts.sum()

                species_probs = {}
                for species, count in species_counts.items():
                    species_probs[species] = float(count / total)

                total_prob = sum(species_probs.values())
                if total_prob > 0:
                    for species in species_probs.keys():
                        species_probs[species] /= total_prob

                sorted_species = sorted(species_probs.items(), key=lambda x: x[1], reverse=True)[:10]

                distributions[season] = {
                    'species': [sp for sp, prob in sorted_species],
                    'probabilities': [float(prob) for sp, prob in sorted_species]
                }

        total_species = sum(len(d.get('species', [])) for d in distributions.values())
        print(f"  Calculated distributions: {total_species} species x season combinations")

        return distributions

    def compute_spatial_risk_profiles(self, faa_data):
        print("Computing spatial risk profiles...")

        profiles = {
            'airports': {},
            'states': {}
        }

        airport_col = 'Airport' if 'Airport' in faa_data.columns else None
        state_col = None
        for candidate in ['State', 'Airport State']:
            if candidate in faa_data.columns:
                state_col = candidate
                break

        if airport_col:
            airport_stats = self.aggregate_damage_stats(faa_data, airport_col)
            for airport, stats in airport_stats.items():
                strike_count = stats['n_strikes']
                damage_rate = stats['n_damage'] / strike_count if strike_count else 0
                risk_score = strike_count * damage_rate
                profiles['airports'][airport] = {
                    'strike_count': strike_count,
                    'damage_rate': float(damage_rate),
                    'risk_score': float(risk_score),
                    'log_odds_contribution': float(np.log1p(risk_score) * 0.5)
                }
            print(f"  Computed risk for {len(profiles['airports'])} airports")

        if state_col:
            state_stats = self.aggregate_damage_stats(faa_data, state_col)
            for state, stats in state_stats.items():
                strike_count = stats['n_strikes']
                damage_rate = stats['n_damage'] / strike_count if strike_count else 0
                risk_score = strike_count * damage_rate
                profiles['states'][state] = {
                    'strike_count': strike_count,
                    'damage_rate': float(damage_rate),
                    'risk_score': float(risk_score),
                    'log_odds_contribution': float(np.log1p(risk_score) * 0.5)
                }
            print(f"  Computed risk for {len(profiles['states'])} states")

        return profiles

    def compute_damage_posteriors(self, faa_data, species_taxonomy):
        print("Computing Beta-Binomial damage posteriors...")

        prior_dist = beta(self.beta_prior_alpha, self.beta_prior_beta)

        posteriors = {
            'by_species_group': {},
            'by_family': {},
            'by_season': {},
            'by_month': {},
            'prior': {
                'alpha': self.beta_prior_alpha,
                'beta': self.beta_prior_beta,
                'mean': float(prior_dist.mean()),
                'variance': float(prior_dist.var())
            }
        }

        if 'DamageSeverity' not in faa_data.columns:
            print("  Missing DamageSeverity column, returning prior only")
            return posteriors

        if 'SpeciesGroup' in faa_data.columns:
            species_stats = self.aggregate_damage_stats(faa_data, 'SpeciesGroup', skip_unknown=False)
            for species_group, stats in species_stats.items():
                n_damage = stats['n_damage']
                n_no_damage = stats['n_strikes'] - n_damage
                alpha_post = self.beta_prior_alpha + n_damage
                beta_post = self.beta_prior_beta + n_no_damage
                posterior_dist = beta(alpha_post, beta_post)
                posteriors['by_species_group'][species_group] = {
                    'alpha': int(alpha_post),
                    'beta': int(beta_post),
                    'mean': float(posterior_dist.mean()),
                    'variance': float(posterior_dist.var()),
                    'n_strikes': stats['n_strikes'],
                    'n_damage': stats['n_damage']
                }
            print(f"  Computed posteriors for {len(posteriors['by_species_group'])} species groups")

        if 'FamilyGroup' in faa_data.columns:
            family_stats = self.aggregate_damage_stats(faa_data, 'FamilyGroup')
            for family, stats in family_stats.items():
                n_damage = stats['n_damage']
                n_no_damage = stats['n_strikes'] - n_damage
                alpha_post = self.beta_prior_alpha + n_damage
                beta_post = self.beta_prior_beta + n_no_damage
                posterior_dist = beta(alpha_post, beta_post)
                posteriors['by_family'][family] = {
                    'alpha': int(alpha_post),
                    'beta': int(beta_post),
                    'mean': float(posterior_dist.mean()),
                    'variance': float(posterior_dist.var()),
                    'n_strikes': stats['n_strikes'],
                    'n_damage': stats['n_damage']
                }
            print(f"  Computed posteriors for {len(posteriors['by_family'])} families")

        season_series = self.ensure_season_column(faa_data)
        if season_series is not None:
            season_stats = self.aggregate_damage_stats(faa_data, 'Season')
            for season, stats in season_stats.items():
                n_damage = stats['n_damage']
                n_no_damage = stats['n_strikes'] - n_damage
                alpha_post = self.beta_prior_alpha + n_damage
                beta_post = self.beta_prior_beta + n_no_damage
                posterior_dist = beta(alpha_post, beta_post)
                posteriors['by_season'][season.lower()] = {
                    'alpha': int(alpha_post),
                    'beta': int(beta_post),
                    'mean': float(posterior_dist.mean()),
                    'variance': float(posterior_dist.var()),
                    'n_strikes': stats['n_strikes'],
                    'n_damage': stats['n_damage']
                }
            print(f"  Computed posteriors for {len(posteriors['by_season'])} seasons")

        if 'Incident Month' in faa_data.columns:
            month_stats = self.aggregate_damage_stats(faa_data, 'Incident Month', skip_unknown=False)
            for month, stats in month_stats.items():
                n_damage = stats['n_damage']
                n_no_damage = stats['n_strikes'] - n_damage
                alpha_post = self.beta_prior_alpha + n_damage
                beta_post = self.beta_prior_beta + n_no_damage
                posterior_dist = beta(alpha_post, beta_post)
                posteriors['by_month'][str(int(month))] = {
                    'alpha': int(alpha_post),
                    'beta': int(beta_post),
                    'mean': float(posterior_dist.mean()),
                    'variance': float(posterior_dist.var()),
                    'n_strikes': stats['n_strikes'],
                    'n_damage': stats['n_damage']
                }
            print(f"  Computed posteriors for {len(posteriors['by_month'])} months")

        return posteriors

    def save_outputs(self, faa_data, seasonal_factors, species_distributions, spatial_profiles, species_taxonomy, damage_posteriors):

        output_file = 'processed_faa_strikes.csv'
        faa_data.to_csv(output_file, index=False)
        print(f"  Saved: {output_file} ({len(faa_data):,} records)")

        output_file = 'seasonal_factors.json'
        with open(output_file, 'w') as f:
            json.dump(seasonal_factors, f, indent=2)
        print(f"  Saved: {output_file}")

        output_file = 'species_distributions.json'
        with open(output_file, 'w') as f:
            json.dump(species_distributions, f, indent=2)
        print(f"  Saved: {output_file}")

        output_file = 'spatial_risk_profiles.json'
        with open(output_file, 'w') as f:
            json.dump(spatial_profiles, f, indent=2)
        print(f"  Saved: {output_file}")

        output_file = 'species_taxonomy.json'
        with open(output_file, 'w') as f:
            json.dump(species_taxonomy, f, indent=2)
        print(f"  Saved: {output_file}")

        output_file = 'damage_posteriors.json'
        with open(output_file, 'w') as f:
            json.dump(damage_posteriors, f, indent=2)
        print(f"  Saved: {output_file}")

def main():
    configure_logging()
    preprocessor = DataPreprocessor()
    preprocessor.run()

if __name__ == "__main__":
    main()
