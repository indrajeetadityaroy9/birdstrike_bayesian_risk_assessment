import numpy as np
import pandas as pd
import warnings
from scipy.stats import beta
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
import pathlib
from utils.logger import get_logger
from utils.geometry import calculate_distance_to_path

logger = get_logger(__name__)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def update_beta_params(prior_alpha, prior_beta, successes, failures):
    epsilon = 1e-6
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + failures + epsilon
    return posterior_alpha, posterior_beta


class BirdStrikeRiskSystem:
    def __init__(self, faa_data_path=None, use_preprocessed=True, spatial_profiles_path='spatial_risk_profiles.json'):
        self.faa_data_path = faa_data_path
        self.use_preprocessed = use_preprocessed
        self.spatial_profiles_path = spatial_profiles_path

        self.faa_data = None
        self.species_damage_prob_profiles = None
        self.family_damage_prob_profiles = None
        self.location_risk_profiles = {}
        self.temporal_damage_prob_profiles = {}
        self.spatial_risk_profiles = {}

        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-999)
        self.iterative_imputer = IterativeImputer(max_iter=10, random_state=42, missing_values=np.nan)

        self.beta_prior_alpha = 1
        self.beta_prior_beta = 9

        self.risk_params = {
            'base_log_odds': -2.5,
            'proximity_log_odds_scale': 2.0,
            'species_log_odds_scale': 1.0,
            'temporal_log_odds_scale': 0.8,
            'spatial_log_odds_scale': 0.5,
            'mc_samples': 500,
            'high_prob_threshold': 0.60,
            'medium_prob_threshold': 0.25,
        }
        logger.info(f"Initialized BirdStrikeRiskSystem (preprocessed={use_preprocessed}) with risk parameters: {self.risk_params}")
        self._load_spatial_profiles()

    def _load_spatial_profiles(self):
        
        import json

        spatial_path = pathlib.Path(self.spatial_profiles_path)
        with open(spatial_path, 'r') as f:
            self.spatial_risk_profiles = json.load(f)

        airport_count = len(self.spatial_risk_profiles.get('airports', {}))
        state_count = len(self.spatial_risk_profiles.get('states', {}))
        logger.info(f"Loaded spatial risk profiles: {airport_count} airports, {state_count} states")

    def load_faa_data(self):
        if not self.faa_data_path:
            raise ValueError("faa_data_path must be provided")

        logger.info(f"Loading FAA data: {self.faa_data_path}")
        self.faa_data = pd.read_csv(self.faa_data_path, low_memory=False)
        logger.info(f"Loaded {len(self.faa_data)} records.")

                                                       
        if not self.use_preprocessed:
            self._preprocess_faa_data()
        else:
            logger.info("Using preprocessed FAA data (skipping imputation)")
                                                                                               

        return True

    def _preprocess_faa_data(self):
        logger.info("Preprocessing FAA data with imputation...")
        if self.faa_data is None:
            logger.error("FAA data not loaded.")
            return

        num_cols = []
        date_parts = ['Incident Year', 'Incident Month', 'Incident Day']
        potential_num = ['Aircraft Damage', 'Height', 'Speed', 'Species Quantity']

        for col in date_parts + potential_num:
            if col in self.faa_data.columns:
                self.faa_data[col] = pd.to_numeric(self.faa_data[col], errors='coerce')
                num_cols.append(col)

        if num_cols and self.numerical_imputer:
            try:
                logger.debug(f"Numerical imputation columns: {num_cols}")
                original_index = self.faa_data.index
                imputed_data = self.numerical_imputer.fit_transform(self.faa_data[num_cols])
                imputed_df = pd.DataFrame(imputed_data, columns=num_cols, index=original_index)
                self.faa_data[num_cols] = imputed_df
                logger.debug("Numerical imputation completed.")
            except Exception as e:
                logger.error(f"Numerical impute error: {e}", exc_info=True)

        cat_cols_to_encode = []
        if 'Species Name' in self.faa_data.columns:
            cat_cols_to_encode.append('Species Name')
        if 'Species ID' in self.faa_data.columns:
            cat_cols_to_encode.append('Species ID')
        if 'Airport State' in self.faa_data.columns:
            cat_cols_to_encode.append('Airport State')

        if not cat_cols_to_encode:
            logger.warning("No categorical columns found for model imputation.")
        else:
            logger.debug(f"Preparing model imputation for columns: {cat_cols_to_encode}")
            impute_placeholder = '_NaN_'
            df_cat_impute = self.faa_data[cat_cols_to_encode].astype(str).fillna(impute_placeholder).copy()

            try:
                encoded_cats = self.categorical_encoder.fit_transform(df_cat_impute)
                encoded_df = pd.DataFrame(encoded_cats, columns=cat_cols_to_encode, index=self.faa_data.index)
                encoded_df.replace(-999, np.nan, inplace=True)
                logger.debug("Categorical encoding for imputation completed.")
            except Exception as e:
                logger.error(f"OrdinalEncode error: {e}. Skipping categorical imputation.", exc_info=True)
                encoded_df = None

            if encoded_df is not None:
                predictor_num_cols = [col for col in num_cols if col in self.faa_data.columns]
                df_for_iterative = pd.concat([encoded_df, self.faa_data[predictor_num_cols]], axis=1)
                logger.debug(f"Applying IterativeImputer (Predictors: {predictor_num_cols})...")

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        imputed_iterative_array = self.iterative_imputer.fit_transform(df_for_iterative)

                    imputed_iterative_df = pd.DataFrame(
                        imputed_iterative_array,
                        columns=df_for_iterative.columns,
                        index=df_for_iterative.index
                    )
                    logger.debug("Iterative imputation completed.")

                    imputed_encoded_cats = imputed_iterative_df[cat_cols_to_encode]
                    imputed_encoded_cats_rounded = np.round(imputed_encoded_cats).astype(int)

                    for i, col in enumerate(cat_cols_to_encode):
                        max_code = len(self.categorical_encoder.categories_[i]) - 1
                        imputed_encoded_cats_rounded[col] = np.clip(
                            imputed_encoded_cats_rounded[col], 0, max_code
                        )

                    imputed_cat_labels = self.categorical_encoder.inverse_transform(imputed_encoded_cats_rounded)
                    imputed_cat_df = pd.DataFrame(
                        imputed_cat_labels,
                        columns=cat_cols_to_encode,
                        index=self.faa_data.index
                    )
                    self.faa_data[cat_cols_to_encode] = imputed_cat_df
                    logger.debug(f"Categorical columns imputed and decoded: {cat_cols_to_encode}")
                except Exception as e:
                    logger.error(f"Iterative impute processing error: {e}.", exc_info=True)

        if all(c in self.faa_data.columns for c in date_parts):
            try:
                self.faa_data[date_parts] = self.faa_data[date_parts].astype(int)
                self.faa_data['IncidentDate'] = pd.to_datetime(
                    self.faa_data[['Incident Year', 'Incident Month', 'Incident Day']],
                    errors='coerce'
                )
                n_null_dates = self.faa_data['IncidentDate'].isnull().sum()
                if n_null_dates > 0:
                    logger.warning(f"Dropping {n_null_dates} rows with invalid dates post-imputation.")
                    self.faa_data.dropna(subset=['IncidentDate'], inplace=True)
                if 'IncidentDate' in self.faa_data.columns:
                    logger.debug("Created 'IncidentDate' column.")
            except Exception as e:
                logger.warning(f"Date creation failed: {e}. Check date columns.")
        elif 'Incident Month' in self.faa_data.columns:
            logger.debug("Using 'Incident Month' for temporal analysis.")
            self.faa_data['Incident Month'] = self.faa_data['Incident Month'].astype(int)
        else:
            logger.error("Missing critical Incident Month/Date information for temporal analysis.")

        primary_damage_col = 'Aircraft Damage'
        if primary_damage_col in self.faa_data.columns:
            self.faa_data['DamageSeverity'] = self.faa_data[primary_damage_col].astype(float).fillna(0).apply(
                lambda x: 1 if x > 0 else 0
            )
            logger.debug("Created 'DamageSeverity' column (0 or 1).")
        else:
            self.faa_data['DamageSeverity'] = 0
            logger.warning(f"Column '{primary_damage_col}' not found. 'DamageSeverity' set to 0 for all records.")

        species_col = 'Species Name'
        family_col = 'Species ID'

        if species_col in self.faa_data.columns:
            self.faa_data[species_col].fillna('Unknown', inplace=True)
            self.faa_data['SpeciesGroup'] = self.faa_data[species_col].apply(self._categorize_species)
            logger.debug("Created 'SpeciesGroup' column.")
        else:
            self.faa_data['SpeciesGroup'] = 'Unknown'
            logger.warning(f"Column '{species_col}' not found. 'SpeciesGroup' set to 'Unknown'.")

        if family_col in self.faa_data.columns:
            self.faa_data[family_col].fillna('Unknown', inplace=True)
            self.faa_data['FamilyGroup'] = self.faa_data[family_col].astype(str).str[0].map(
                {'B': 'Birds', 'M': 'Mammals', 'R': 'Reptiles'}
            ).fillna('Unknown')
            logger.debug("Created 'FamilyGroup' column based on 'Species ID'.")
        else:
            self.faa_data['FamilyGroup'] = 'Unknown'
            logger.warning(f"Column '{family_col}' not found. 'FamilyGroup' set to 'Unknown'.")

        self.faa_data['Strike'] = 1
        logger.info("FAA preprocessing complete.")

    def _categorize_species(self, species_name):
        if pd.isna(species_name): return "Unknown"
        name = str(species_name).lower()
        if 'gull' in name: return "Gull"
        if 'goose' in name or 'geese' in name: return "Goose"
        if 'duck' in name: return "Duck"
        if 'hawk' in name: return "Hawk"
        if 'eagle' in name: return "Eagle"
        if 'vulture' in name: return "Vulture"
        if 'heron' in name or 'egret' in name: return "Heron/Egret"
        if 'sparrow' in name or 'starling' in name or 'swallow' in name or 'blackbird' in name: return "Small Bird"
        if 'owl' in name: return "Owl"
        if 'pigeon' in name or 'dove' in name: return "Pigeon/Dove"
        if 'unknown' in name or 'bird' not in name: return "Unknown/Other"
        return "Other Bird"

    def analyze_species_risk(self):
        if self.faa_data is None or not all(c in self.faa_data.columns for c in
                                            ['FamilyGroup', 'SpeciesGroup', 'Strike', 'DamageSeverity']):
            logger.error("Missing columns for hierarchical species analysis.")
            return

        logger.info("Analyzing hierarchical species damage probability...")
        try:
            agg_cols = {'DamageSeverity': 'sum', 'Strike': 'count'}
            family_counts = self.faa_data.groupby('FamilyGroup').agg(agg_cols).reset_index().rename(
                columns={'DamageSeverity': 'DamageCount', 'Strike': 'TotalStrikes'}
            )
            species_counts = self.faa_data.groupby(['FamilyGroup', 'SpeciesGroup']).agg(agg_cols).reset_index().rename(
                columns={'DamageSeverity': 'DamageCount', 'Strike': 'TotalStrikes'}
            )

            family_counts['NoDamageCount'] = family_counts['TotalStrikes'] - family_counts['DamageCount']
            family_counts['alpha'], family_counts['beta'] = update_beta_params(
                self.beta_prior_alpha, self.beta_prior_beta,
                family_counts['DamageCount'], family_counts['NoDamageCount']
            )
            self.family_damage_prob_profiles = family_counts[['FamilyGroup', 'alpha', 'beta']].set_index('FamilyGroup')
            logger.debug(f"Family Beta Params calculated for {len(self.family_damage_prob_profiles)} families.")

            species_counts['NoDamageCount'] = species_counts['TotalStrikes'] - species_counts['DamageCount']
            species_counts['alpha'], species_counts['beta'] = update_beta_params(
                self.beta_prior_alpha, self.beta_prior_beta,
                species_counts['DamageCount'], species_counts['NoDamageCount']
            )
            self.species_damage_prob_profiles = species_counts[['FamilyGroup', 'SpeciesGroup', 'alpha', 'beta']]
            logger.debug(f"SpeciesGroup Beta Params calculated for {len(self.species_damage_prob_profiles)} groups.")
            logger.info("OK: SpeciesGroup Beta Params analyzed.")


        except Exception as e:
            logger.error(f"FAIL: Hierarchical species analysis: {e}", exc_info=True)
            self.family_damage_prob_profiles = pd.DataFrame(
                {'FamilyGroup': ['Unknown'], 'alpha': [self.beta_prior_alpha], 'beta': [self.beta_prior_beta]}
            ).set_index('FamilyGroup')
            self.species_damage_prob_profiles = pd.DataFrame(
                {'FamilyGroup': ['Unknown'], 'SpeciesGroup': ['Unknown'],
                 'alpha': [self.beta_prior_alpha], 'beta': [self.beta_prior_beta]}
            )

    def analyze_temporal_patterns(self):
        if self.faa_data is None or ('Incident Month' not in self.faa_data.columns and
                                     'IncidentDate' not in self.faa_data.columns):
            logger.error("Missing Month/Date for temporal analysis.")
            self.temporal_damage_prob_profiles = {}
            return {}

        logger.info("Analyzing temporal damage probability...")
        try:
            if 'IncidentDate' in self.faa_data.columns and not self.faa_data['IncidentDate'].isnull().all():
                temp_data = self.faa_data.dropna(subset=['IncidentDate']).copy()
                temp_data['Month'] = temp_data['IncidentDate'].dt.month
                logger.debug("Using 'IncidentDate' for temporal analysis.")
            elif 'Incident Month' in self.faa_data.columns:
                temp_data = self.faa_data.dropna(subset=['Incident Month']).copy()
                temp_data['Month'] = temp_data['Incident Month'].astype(int)
                temp_data = temp_data[temp_data['Month'].between(1, 12)]
                logger.debug("Using 'Incident Month' for temporal analysis.")
            else:
                logger.error("No valid month data found for temporal analysis.")
                self.temporal_damage_prob_profiles = {}
                return {}

            if temp_data.empty:
                logger.warning("No valid temporal data points after filtering.")
                self.temporal_damage_prob_profiles = {}
                return {}

            agg_cols = {'DamageSeverity': 'sum', 'Strike': 'count'}

            monthly_counts = temp_data.groupby('Month').agg(agg_cols).reset_index().rename(
                columns={'DamageSeverity': 'DamageCount', 'Strike': 'TotalStrikes'}
            )
            monthly_counts['NoDamageCount'] = monthly_counts['TotalStrikes'] - monthly_counts['DamageCount']
            monthly_counts['alpha'], monthly_counts['beta'] = update_beta_params(
                self.beta_prior_alpha, self.beta_prior_beta,
                monthly_counts['DamageCount'], monthly_counts['NoDamageCount']
            )
            logger.debug("Monthly temporal Beta Params calculated.")

            temp_data['Season'] = temp_data['Month'].apply(
                lambda m: 'Winter' if m in [12, 1, 2] else
                'Spring' if m in [3, 4, 5] else
                'Summer' if m in [6, 7, 8] else 'Fall'
            )
            seasonal_counts = temp_data.groupby('Season').agg(agg_cols).reindex(
                ['Winter', 'Spring', 'Summer', 'Fall']
            ).fillna(0).reset_index().rename(
                columns={'DamageSeverity': 'DamageCount', 'Strike': 'TotalStrikes'}
            )
            seasonal_counts['NoDamageCount'] = seasonal_counts['TotalStrikes'] - seasonal_counts['DamageCount']
            seasonal_counts['alpha'], seasonal_counts['beta'] = update_beta_params(
                self.beta_prior_alpha, self.beta_prior_beta,
                seasonal_counts['DamageCount'], seasonal_counts['NoDamageCount']
            )
            logger.debug("Seasonal temporal Beta Params calculated.")

            self.temporal_damage_prob_profiles = {
                'monthly': monthly_counts[['Month', 'alpha', 'beta']].set_index('Month'),
                'seasonal': seasonal_counts[['Season', 'alpha', 'beta']].set_index('Season')
            }
            logger.info("OK: Temporal damage probability analysis complete.")
            return self.temporal_damage_prob_profiles

        except Exception as e:
            logger.error(f"FAIL: Temporal analysis: {e}", exc_info=True)
            self.temporal_damage_prob_profiles = {}
            return {}

    def analyze_spatial_patterns(self):
        if self.faa_data is None:
            logger.warning("No FAA data loaded, skipping spatial analysis.")
            return {}

        logger.info("Analyzing spatial risk patterns...")
        self.location_risk_profiles = {}

        try:
            if 'Airport State' in self.faa_data.columns:
                state_risk = self.faa_data.groupby('Airport State').agg(
                    StrikeCount=('Strike', 'count'),
                    AvgDamageSeverity=('DamageSeverity', 'mean')
                ).reset_index().fillna({'AvgDamageSeverity': 0})
                state_risk['RiskScore'] = state_risk['StrikeCount'] * state_risk['AvgDamageSeverity']
                self.location_risk_profiles['state'] = state_risk.sort_values('RiskScore', ascending=False)
                logger.debug(f"State spatial risk calculated for {len(state_risk)} states.")

            if 'Airport ID' in self.faa_data.columns:
                airport_risk = self.faa_data.groupby('Airport ID').agg(
                    StrikeCount=('Strike', 'count'),
                    AvgDamageSeverity=('DamageSeverity', 'mean')
                ).reset_index().fillna({'AvgDamageSeverity': 0})
                airport_risk['RiskScore'] = airport_risk['StrikeCount'] * airport_risk['AvgDamageSeverity']
                self.location_risk_profiles['airport'] = airport_risk.sort_values('RiskScore', ascending=False)
                logger.debug(f"Airport spatial risk calculated for {len(airport_risk)} airports.")

            logger.info("Spatial analysis complete.")
            return self.location_risk_profiles

        except Exception as e:
            logger.error(f"FAIL: Spatial analysis encountered an error: {e}", exc_info=True)
            return {}

    def _get_hierarchical_species_damage_dist_params(self, family_group=None, species_group=None):
        alpha = self.beta_prior_alpha
        beta_val = self.beta_prior_beta
        source = "Base Prior"

        if family_group and self.family_damage_prob_profiles is not None and family_group in self.family_damage_prob_profiles.index:
            try:
                fa, fb = self.family_damage_prob_profiles.loc[family_group, ['alpha', 'beta']]
                if pd.notna(fa) and pd.notna(fb):
                    alpha, beta_val = float(fa), float(fb)
                    source = f"Family ({family_group})"
            except KeyError:
                logger.debug(f"Family '{family_group}' not found in profiles, using base/prior.")
            except Exception as e:
                logger.warning(f"Error accessing family profile '{family_group}': {e}. Using base/prior.")

        if species_group and self.species_damage_prob_profiles is not None:
            mask = (self.species_damage_prob_profiles['SpeciesGroup'] == species_group)
            if family_group:
                mask &= (self.species_damage_prob_profiles['FamilyGroup'] == family_group)
            match = self.species_damage_prob_profiles[mask]

            if not match.empty:
                try:
                    sa, sb = match[['alpha', 'beta']].iloc[0]
                    if pd.notna(sa) and pd.notna(sb):
                        alpha, beta_val = float(sa), float(sb)
                        source = f"Species ({species_group})"
                        if family_group:
                            source += f" in Family ({family_group})"

                except Exception as e:
                    logger.warning(f"Error accessing species profile '{species_group}': {e}. Using {source} params.")

        logger.debug(
            f"[RiskParams] Species/Family='{species_group}/{family_group}': Using Beta(a={alpha:.2f}, b={beta_val:.2f}) from {source}")
        return alpha, beta_val

    def _get_temporal_damage_dist_params(self, month=None, season=None):
        alpha = self.beta_prior_alpha
        beta_val = self.beta_prior_beta
        source = "Base Prior"

        if not self.temporal_damage_prob_profiles:
            logger.debug("[RiskParams] Temporal profiles not loaded, using Base Prior.")
            return alpha, beta_val

        if month is not None and isinstance(month, (int, float)) and 1 <= int(month) <= 12:
            month_int = int(month)
            monthly = self.temporal_damage_prob_profiles.get('monthly')
            if monthly is not None and not monthly.empty:
                try:
                    ma, mb = monthly.loc[month_int, ['alpha', 'beta']]
                    if pd.notna(ma) and pd.notna(mb):
                        alpha, beta_val = float(ma), float(mb)
                        source = f"Monthly ({month_int})"
                except KeyError:
                    logger.debug(f"Month '{month_int}' not found in monthly profiles, checking season/prior.")
                except Exception as e:
                    logger.warning(f"Error accessing monthly profile '{month_int}': {e}. Checking season/prior.")

        if source == "Base Prior" and season is not None:
            seasonal = self.temporal_damage_prob_profiles.get('seasonal')
            season_std = str(season).capitalize()
            if seasonal is not None and not seasonal.empty:
                try:
                    sa, sb = seasonal.loc[season_std, ['alpha', 'beta']]
                    if pd.notna(sa) and pd.notna(sb):
                        alpha, beta_val = float(sa), float(sb)
                        source = f"Seasonal ({season_std})"
                except KeyError:
                    logger.debug(f"Season '{season_std}' not found in seasonal profiles, using Base Prior.")
                except Exception as e:
                    logger.warning(f"Error accessing seasonal profile '{season_std}': {e}. Using Base Prior.")

        logger.debug(
            f"[RiskParams] Month/Season='{month}/{season}': Using Beta(a={alpha:.2f}, b={beta_val:.2f}) from {source}")
        return alpha, beta_val

    def calculate_probabilistic_risk_metrics(self, bird_position, flight_path,
                                             family_group=None, species_group=None,
                                             month=None, season=None,
                                             airport=None, state=None):
        n_samples = self.risk_params['mc_samples']
        log_prefix = f"[RiskCalc Pos={hash(tuple(bird_position))}]"

        logger.debug(f"{log_prefix} Calculating risk for Species='{species_group}', Family='{family_group}', Month='{month}', Season='{season}' near path='{flight_path.get('id', 'N/A')}'")

        distance = calculate_distance_to_path(bird_position, flight_path)
        path_width = flight_path.get("width", 0.3)
        norm_distance = distance / path_width if path_width > 0 else distance / 0.3
                                                                      
        proximity_log_odds = 2.0 * ((1.0 - np.clip(norm_distance, 0.0, 1.0)) ** 2)
        logger.debug(f"{log_prefix} Distance={distance:.3f} km, NormDist={norm_distance:.3f}, ProxLogOdds={proximity_log_odds:.3f}")

        spec_alpha, spec_beta = self._get_hierarchical_species_damage_dist_params(family_group, species_group)
        temp_alpha, temp_beta = self._get_temporal_damage_dist_params(month, season)

        spec_alpha, spec_beta = max(1e-6, spec_alpha), max(1e-6, spec_beta)
        temp_alpha, temp_beta = max(1e-6, temp_alpha), max(1e-6, temp_beta)
        mean_p_damage_species = spec_alpha / (spec_alpha + spec_beta)
        mean_p_damage_temporal = temp_alpha / (temp_alpha + temp_beta)
        logger.debug(
            f"{log_prefix} Species Params: Beta(a={spec_alpha:.2f}, b={spec_beta:.2f}) -> MeanP={mean_p_damage_species:.3f}")
        logger.debug(
            f"{log_prefix} Temporal Params: Beta(a={temp_alpha:.2f}, b={temp_beta:.2f}) -> MeanP={mean_p_damage_temporal:.3f}")

        try:
            p_damage_samples_species = beta.rvs(spec_alpha, spec_beta, size=n_samples)
            p_damage_samples_temporal = beta.rvs(temp_alpha, temp_beta, size=n_samples)
        except ValueError as e:
            logger.error(f"{log_prefix} Error sampling from Beta distributions: {e}. Using mean values as fallback.")
            p_damage_samples_species = np.full(n_samples, mean_p_damage_species)
            p_damage_samples_temporal = np.full(n_samples, mean_p_damage_temporal)

        epsilon = 1e-9

        base_prob = self.beta_prior_alpha / (self.beta_prior_alpha + self.beta_prior_beta)
        base_log_odds_term = np.log((base_prob + epsilon) / (1 - base_prob + epsilon))

        log_odds_species = np.log((p_damage_samples_species + epsilon) / (1 - p_damage_samples_species + epsilon))
        log_odds_ratio_species = log_odds_species - base_log_odds_term
        species_log_odds_contrib = self.risk_params['species_log_odds_scale'] * log_odds_ratio_species

        log_odds_temporal = np.log((p_damage_samples_temporal + epsilon) / (1 - p_damage_samples_temporal + epsilon))
        log_odds_ratio_temporal = log_odds_temporal - base_log_odds_term
        temporal_log_odds_contrib = self.risk_params['temporal_log_odds_scale'] * log_odds_ratio_temporal

        logger.debug(f"{log_prefix} Mean LogOdds Contrib: Species={np.mean(species_log_odds_contrib):.3f}, Temporal={np.mean(temporal_log_odds_contrib):.3f}")
                                                                         
        spatial_log_odds = 0.0
        if airport and airport in self.spatial_risk_profiles.get('airports', {}):
            spatial_log_odds = self.spatial_risk_profiles['airports'][airport]['log_odds_contribution']
            logger.debug(f"{log_prefix} Airport '{airport}' spatial contribution: {spatial_log_odds:.3f}")
        elif state and state in self.spatial_risk_profiles.get('states', {}):
            spatial_log_odds = self.spatial_risk_profiles['states'][state]['log_odds_contribution']
            logger.debug(f"{log_prefix} State '{state}' spatial contribution: {spatial_log_odds:.3f}")

        spatial_log_odds_contrib = self.risk_params['spatial_log_odds_scale'] * spatial_log_odds

        total_log_odds_samples = (
                self.risk_params['base_log_odds'] +
                proximity_log_odds +
                species_log_odds_contrib +
                temporal_log_odds_contrib +
                spatial_log_odds_contrib                          
        )
        logger.debug(
            f"{log_prefix} Combined LogOdds: Mean={np.mean(total_log_odds_samples):.3f}, Std={np.std(total_log_odds_samples):.3f}")

        posterior_prob_samples = 1.0 / (1.0 + np.exp(-total_log_odds_samples))
                                                                                             
        posterior_prob_samples = np.clip(posterior_prob_samples, 1e-8, 1.0 - 1e-8)

        mean_prob = np.mean(posterior_prob_samples)
        std_dev_prob = np.std(posterior_prob_samples)
        ci_lower = np.percentile(posterior_prob_samples, 5)
        ci_upper = np.percentile(posterior_prob_samples, 95)
        logger.debug(f"{log_prefix} Final Posterior Prob: Mean={mean_prob:.3f}, Std={std_dev_prob:.3f}, CI90=({ci_lower:.3f}, {ci_upper:.3f})")

        if mean_prob >= self.risk_params['high_prob_threshold']:
            category = "HIGH"
        elif mean_prob >= self.risk_params['medium_prob_threshold']:
            category = "MEDIUM"
        else:
            category = "LOW"
        logger.debug(f"{log_prefix} Assigned Category: {category} (based on MeanProb={mean_prob:.3f})")

        return {
            "mean_prob": mean_prob,
            "std_dev_prob": std_dev_prob,
            "prob_ci_90": (ci_lower, ci_upper),
            "category": category,
        }
