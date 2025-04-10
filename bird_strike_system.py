import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class BirdStrikeRiskSystem:
    """
    Enhanced Bird Strike Risk Assessment System
    Integrates FAA strike data with Bayesian radar-based detection approach
    """
    
    def __init__(self, faa_data_path=None):
        """Initialize the system with paths to datasets"""
        self.faa_data_path = faa_data_path
        self.faa_data = None
        self.species_risk_profiles = {}
        self.location_risk_profiles = {}
        self.temporal_risk_profiles = {}
        self.sigma_x = 0.25  # Standard deviations for Gaussian priors
        self.sigma_y = 0.25
        self.sigma_z = 0.15
        self.sigma_noise = 0.3  # Standard deviation for measurement noise
        
    def load_faa_data(self):
        """Load and preprocess the FAA wildlife strike dataset"""
        if self.faa_data_path is None:
            print("No FAA data path provided. Skipping FAA data integration.")
            return False
        
        try:
            print(f"Loading FAA wildlife strike data from {self.faa_data_path}")
            self.faa_data = pd.read_csv(self.faa_data_path)
            
            # Show available columns for debugging
            print(f"Available columns: {self.faa_data.columns.tolist()}")
            
            # Basic preprocessing
            # Convert date columns to datetime if needed
            if all(col in self.faa_data.columns for col in ['Incident Year', 'Incident Month', 'Incident Day']):
                try:
                    self.faa_data['IncidentDate'] = pd.to_datetime(
                        self.faa_data[['Incident Year', 'Incident Month', 'Incident Day']])
                    print("Created IncidentDate from year/month/day columns")
                except Exception as e:
                    print(f"Warning: Could not create IncidentDate: {str(e)}")
            
            # Create damage severity indicator if possible
            damage_cols = [col for col in self.faa_data.columns if 'Damage' in col]
            if damage_cols:
                try:
                    # Make sure damage columns are numeric
                    for col in damage_cols:
                        if self.faa_data[col].dtype not in ['int64', 'float64']:
                            self.faa_data[col] = pd.to_numeric(self.faa_data[col], errors='coerce').fillna(0)
                    
                    self.faa_data['DamageSeverity'] = self.faa_data[damage_cols].sum(axis=1)
                    print(f"Created DamageSeverity from {len(damage_cols)} damage columns")
                except Exception as e:
                    print(f"Warning: Could not create DamageSeverity: {str(e)}")
                    # Create a default DamageSeverity column
                    self.faa_data['DamageSeverity'] = 0
                    print("Created a default DamageSeverity column with zeros")
            else:
                print("Warning: No damage columns found. Creating default DamageSeverity column.")
                self.faa_data['DamageSeverity'] = 0
            
            # Extract bird species information
            if 'Species Name' in self.faa_data.columns:
                self.faa_data['SpeciesGroup'] = self.faa_data['Species Name'].apply(self._categorize_species)
                print("Created SpeciesGroup from Species Name")
            else:
                print("Warning: No Species Name column found. Creating default SpeciesGroup.")
                self.faa_data['SpeciesGroup'] = 'Unknown'
            
            print(f"Successfully loaded {len(self.faa_data)} strike records")
            return True
                
        except Exception as e:
            print(f"Error loading FAA data: {str(e)}")
            # Create a minimal default dataframe
            self.faa_data = pd.DataFrame({
                'Record ID': range(1, 11),
                'SpeciesGroup': ['Unknown'] * 10,
                'DamageSeverity': [0] * 10,
                'Aircraft Damage': [0] * 10
            })
            print("Created fallback dataset with minimal structure")
        return True  # Return True so processing can continue with default data
    
    def _categorize_species(self, species_name):
        """Categorize bird species into groups based on name"""
        if pd.isna(species_name):
            return "Unknown"
        
        species_lower = str(species_name).lower()
        
        if 'gull' in species_lower:
            return "Gull"
        elif 'goose' in species_lower or 'geese' in species_lower:
            return "Goose"
        elif 'hawk' in species_lower:
            return "Hawk"
        elif 'eagle' in species_lower:
            return "Eagle"
        elif 'owl' in species_lower:
            return "Owl"
        elif 'duck' in species_lower:
            return "Duck"
        elif 'sparrow' in species_lower or 'finch' in species_lower or 'swallow' in species_lower:
            return "Small Passerine"
        elif 'unknown' in species_lower:
            return "Unknown"
        else:
            return "Other"
    
    def analyze_species_risk(self):
        """Analyze risk profiles of different bird species based on FAA data"""
        if self.faa_data is None:
            print("No FAA data loaded. Please load data first.")
            return
        
        print("Analyzing species risk profiles...")
        
        # Check if SpeciesGroup column exists
        if 'SpeciesGroup' not in self.faa_data.columns:
            # Try to create it if Species Name column exists
            if 'Species Name' in self.faa_data.columns:
                print("Creating SpeciesGroup column from Species Name...")
                self.faa_data['SpeciesGroup'] = self.faa_data['Species Name'].apply(self._categorize_species)
            else:
                # If no Species Name column, use a simpler approach with available data
                print("WARNING: 'Species Name' column not found. Using simplified risk analysis.")
                # Print column names for debugging
                print(f"Available columns: {self.faa_data.columns.tolist()}")
                
                # Create a default species group
                self.faa_data['SpeciesGroup'] = 'Unknown'
        
        # Now check if we have the required columns for the analysis
        required_columns = ['SpeciesGroup', 'Record ID']
        damage_indicator = None
        
        # Find a column we can use to indicate damage
        for col in self.faa_data.columns:
            if 'Damage' in col:
                damage_indicator = col
                break
        
        if damage_indicator is None:
            print("WARNING: No damage column found. Adding placeholder data.")
            self.faa_data['Aircraft Damage'] = 0
            damage_indicator = 'Aircraft Damage'
        
        if not all(col in self.faa_data.columns for col in required_columns):
            print(f"ERROR: Missing required columns for analysis. Available columns: {self.faa_data.columns.tolist()}")
            # Create a minimal species risk profile
            self.species_risk_profiles = pd.DataFrame({
                'SpeciesGroup': ['Unknown'],
                'StrikeCount': [1],
                'AvgDamageSeverity': [0],
                'TotalDamageSeverity': [0],
                'MaxDamageSeverity': [0],
                'DamagePercentage': [0],
                'FrequencyScore': [1.0],
                'SeverityScore': [0],
                'RiskScore': [0.4]  # Default risk score
            })
            print("Created fallback species risk profiles.")
            return self.species_risk_profiles
            
        try:
            # Group by species and calculate risk metrics
            species_risk = self.faa_data.groupby('SpeciesGroup').agg({
                'Record ID': 'count',
                damage_indicator: [
                    lambda x: x.mean() if x.dtype.kind in 'if' else 0,  # Mean if numeric
                    lambda x: x.sum() if x.dtype.kind in 'if' else 0,   # Sum if numeric
                    lambda x: x.max() if x.dtype.kind in 'if' else 0    # Max if numeric
                ]
            }).reset_index()
            
            # Rename columns for clarity
            col_names = ['SpeciesGroup', 'StrikeCount', 'AvgDamageSeverity', 'TotalDamageSeverity', 'MaxDamageSeverity']
            species_risk.columns = col_names
            
            # Add damage percentage (using a safer approach)
            if damage_indicator in self.faa_data.columns and self.faa_data[damage_indicator].dtype.kind in 'if':
                damage_pct = self.faa_data.groupby('SpeciesGroup')[damage_indicator].apply(
                    lambda x: (x > 0).mean() * 100 if x.dtype.kind in 'if' else 0
                ).reset_index()
                
                species_risk = species_risk.merge(
                    damage_pct.rename(columns={damage_indicator: 'DamagePercentage'}),
                    on='SpeciesGroup'
                )
            else:
                species_risk['DamagePercentage'] = 0
            
            # Calculate risk score based on frequency and severity
            total_strikes = species_risk['StrikeCount'].sum()
            species_risk['FrequencyScore'] = species_risk['StrikeCount'] / total_strikes if total_strikes > 0 else 0
            
            severity_max = species_risk['AvgDamageSeverity'].max()
            species_risk['SeverityScore'] = (
                species_risk['AvgDamageSeverity'] / severity_max if severity_max > 0 else 0
            )
            
            species_risk['RiskScore'] = (species_risk['FrequencyScore'] * 0.4) + (species_risk['SeverityScore'] * 0.6)
            
            # Sort by risk score
            species_risk = species_risk.sort_values('RiskScore', ascending=False)
            
            self.species_risk_profiles = species_risk
            print("Species risk analysis complete.")
            
            return species_risk
            
        except Exception as e:
            print(f"Error in species risk analysis: {str(e)}")
            # Create a minimal fallback dataframe
            self.species_risk_profiles = pd.DataFrame({
                'SpeciesGroup': ['Unknown'],
                'StrikeCount': [1],
                'AvgDamageSeverity': [0],
                'TotalDamageSeverity': [0],
                'MaxDamageSeverity': [0],
                'DamagePercentage': [0],
                'FrequencyScore': [1.0],
                'SeverityScore': [0],
                'RiskScore': [0.4]  # Default risk score
            })
            print("Created fallback species risk profiles due to error.")
            return self.species_risk_profiles
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in bird strikes"""
        if self.faa_data is None:
            print("No FAA data loaded. Please load data first.")
            return {}
        
        print("Analyzing temporal patterns in bird strikes...")
        
        # Check for required columns
        required_cols = ['Record ID', 'DamageSeverity']
        missing_cols = [col for col in required_cols if col not in self.faa_data.columns]
        if missing_cols:
            print(f"Warning: Missing required columns: {missing_cols}")
            for col in missing_cols:
                self.faa_data[col] = 0  # Add default columns
        
        # Initialize empty result
        self.temporal_risk_profiles = {}
        
        # Extract temporal components if possible
        if 'IncidentDate' in self.faa_data.columns:
            try:
                self.faa_data['Month'] = self.faa_data['IncidentDate'].dt.month
                self.faa_data['Season'] = self.faa_data['IncidentDate'].dt.month.apply(
                    lambda x: 'Winter' if x in [12, 1, 2] else
                            'Spring' if x in [3, 4, 5] else
                            'Summer' if x in [6, 7, 8] else 'Fall'
                )
                
                # Monthly patterns
                monthly_strikes = self.faa_data.groupby('Month').agg({
                    'Record ID': 'count',
                    'DamageSeverity': ['mean', 'sum']
                }).reset_index()
                
                monthly_strikes.columns = ['Month', 'StrikeCount', 'AvgDamageSeverity', 'TotalDamageSeverity']
                
                # Seasonal patterns
                seasonal_strikes = self.faa_data.groupby('Season').agg({
                    'Record ID': 'count',
                    'DamageSeverity': ['mean', 'sum']
                }).reset_index()
                
                seasonal_strikes.columns = ['Season', 'StrikeCount', 'AvgDamageSeverity', 'TotalDamageSeverity']
                
                self.temporal_risk_profiles = {
                    'monthly': monthly_strikes,
                    'seasonal': seasonal_strikes
                }
                
            except Exception as e:
                print(f"Error in temporal analysis: {str(e)}")
                # Create default seasonal data
                self.temporal_risk_profiles = {
                    'seasonal': pd.DataFrame({
                        'Season': ['Winter', 'Spring', 'Summer', 'Fall'],
                        'StrikeCount': [10, 20, 30, 20],
                        'AvgDamageSeverity': [1.0, 1.2, 0.8, 1.0],
                        'TotalDamageSeverity': [10, 24, 24, 20]
                    })
                }
        else:
            print("Date information not available. Creating default temporal patterns.")
            # Create default seasonal data
            self.temporal_risk_profiles = {
                'seasonal': pd.DataFrame({
                    'Season': ['Winter', 'Spring', 'Summer', 'Fall'],
                    'StrikeCount': [10, 20, 30, 20],
                    'AvgDamageSeverity': [1.0, 1.2, 0.8, 1.0],
                    'TotalDamageSeverity': [10, 24, 24, 20]
                })
            }
        
        print("Temporal pattern analysis complete.")
        return self.temporal_risk_profiles
    
    def analyze_spatial_patterns(self):
        """Analyze spatial patterns in bird strikes"""
        if self.faa_data is None:
            print("No FAA data loaded. Please load data first.")
            return
        
        print("Analyzing spatial patterns in bird strikes...")
        
        # Group by airport/state
        if 'Airport State' in self.faa_data.columns:
            state_strikes = self.faa_data.groupby('Airport State').agg({
                'Record ID': 'count',
                'DamageSeverity': ['mean', 'sum']
            }).reset_index()
            
            state_strikes.columns = ['State', 'StrikeCount', 'AvgDamageSeverity', 'TotalDamageSeverity']
            state_strikes['RiskScore'] = state_strikes['StrikeCount'] * state_strikes['AvgDamageSeverity']
            state_strikes = state_strikes.sort_values('RiskScore', ascending=False)
            
            self.location_risk_profiles['state'] = state_strikes
        
        if 'Airport ID' in self.faa_data.columns:
            airport_strikes = self.faa_data.groupby('Airport ID').agg({
                'Record ID': 'count',
                'DamageSeverity': ['mean', 'sum']
            }).reset_index()
            
            airport_strikes.columns = ['Airport', 'StrikeCount', 'AvgDamageSeverity', 'TotalDamageSeverity']
            airport_strikes['RiskScore'] = airport_strikes['StrikeCount'] * airport_strikes['AvgDamageSeverity']
            airport_strikes = airport_strikes.sort_values('RiskScore', ascending=False)
            
            self.location_risk_profiles['airport'] = airport_strikes
        
        print("Spatial pattern analysis complete.")
        return self.location_risk_profiles
    
    def extract_strike_conditions(self):
        """Extract conditions associated with strikes"""
        if self.faa_data is None:
            print("No FAA data loaded. Please load data first.")
            return
        
        print("Analyzing strike conditions...")
        
        condition_factors = {}
        
        # Flight phase analysis
        if 'Flight Phase' in self.faa_data.columns:
            flight_phase = self.faa_data.groupby('Flight Phase').agg({
                'Record ID': 'count',
                'DamageSeverity': ['mean', 'sum']
            }).reset_index()
            
            flight_phase.columns = ['FlightPhase', 'StrikeCount', 'AvgDamageSeverity', 'TotalDamageSeverity']
            condition_factors['flight_phase'] = flight_phase
        
        # Visibility analysis
        if 'Visibility' in self.faa_data.columns:
            visibility = self.faa_data.groupby('Visibility').agg({
                'Record ID': 'count',
                'DamageSeverity': ['mean', 'sum']
            }).reset_index()
            
            visibility.columns = ['Visibility', 'StrikeCount', 'AvgDamageSeverity', 'TotalDamageSeverity']
            condition_factors['visibility'] = visibility
        
        # Height analysis
        if 'Height' in self.faa_data.columns:
            # Convert to numeric if needed
            if self.faa_data['Height'].dtype == 'object':
                self.faa_data['Height'] = pd.to_numeric(self.faa_data['Height'], errors='coerce')
            
            # Bin heights
            height_bins = [0, 500, 1000, 3000, 10000, 100000]
            labels = ['0-500', '500-1000', '1000-3000', '3000-10000', '10000+']
            self.faa_data['HeightBin'] = pd.cut(self.faa_data['Height'], bins=height_bins, labels=labels)
            
            height = self.faa_data.groupby('HeightBin').agg({
                'Record ID': 'count',
                'DamageSeverity': ['mean', 'sum']
            }).reset_index()
            
            height.columns = ['HeightBin', 'StrikeCount', 'AvgDamageSeverity', 'TotalDamageSeverity']
            condition_factors['height'] = height
        
        return condition_factors
    
    def generate_risk_matrix(self):
        """Generate a risk matrix based on species and conditions"""
        if self.faa_data is None:
            print("No FAA data loaded. Please load data first.")
            return
        
        print("Generating comprehensive risk matrix...")
        
        # Create a risk matrix with species groups and flight phases
        if 'SpeciesGroup' in self.faa_data.columns and 'Flight Phase' in self.faa_data.columns:
            risk_matrix = self.faa_data.pivot_table(
                index='SpeciesGroup',
                columns='Flight Phase',
                values='DamageSeverity',
                aggfunc='mean',
                fill_value=0
            )
            
            return risk_matrix
        else:
            print("Required columns for risk matrix not available.")
            return None
    
    def calculate_distance_3d(self, x1, y1, z1, x2, y2, z2):
        """Calculate 3D Euclidean distance between two points"""
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

    def compute_map_objective(self, bird_position, sensors_x, sensors_y, sensors_z, 
                             measurements, sigma_x, sigma_y, sigma_z, sigma_noise, prior_params):
        """
        Compute the MAP objective function for bird position estimation.
        This is the negative log of posterior probability.
        """
        x, y, z = bird_position
        
        # Extract prior parameters
        mu_x, mu_y, mu_z = prior_params['mu']
        
        # Prior probability based on bird behavior (negative log of Gaussian)
        prior = ((x - mu_x)**2 / sigma_x**2 + 
                 (y - mu_y)**2 / sigma_y**2 + 
                 (z - mu_z)**2 / sigma_z**2)
        
        # Likelihood from sensor measurements
        likelihood = 0
        for i in range(len(sensors_x)):
            measured_range = measurements[i]
            true_range = self.calculate_distance_3d(x, y, z, sensors_x[i], sensors_y[i], sensors_z[i])
            likelihood += ((measured_range - true_range)**2) / sigma_noise**2
        
        # Total objective function (negative log posterior)
        return prior + likelihood
    
    def calculate_risk(self, bird_position, flight_path, species_group=None, time_of_day=None, season=None):
        """
        Calculate collision risk based on distance to flight path 
        and integrated FAA risk factors
        """
        x, y, z = bird_position
        start_x, start_y, start_z = flight_path["start"]
        end_x, end_y, end_z = flight_path["end"]
        width = flight_path["width"]
        
        # Calculate the closest point on the flight path line segment
        v_x = end_x - start_x
        v_y = end_y - start_y
        v_z = end_z - start_z
        
        # Parameter of closest point on line
        t = ((x - start_x) * v_x + (y - start_y) * v_y + (z - start_z) * v_z) / \
            (v_x * v_x + v_y * v_y + v_z * v_z)
        
        # Constrain to line segment
        t = max(0, min(1, t))
        
        # Calculate closest point coordinates
        closest_x = start_x + t * v_x
        closest_y = start_y + t * v_y
        closest_z = start_z + t * v_z
        
        # Calculate distance from bird to closest point
        distance = np.sqrt((x - closest_x)**2 + (y - closest_y)**2 + (z - closest_z)**2)
        
        # Base risk level based on distance
        if distance < width / 2:
            base_risk = "HIGH"
            risk_score = 3
        elif distance < width:
            base_risk = "MEDIUM"
            risk_score = 2
        else:
            base_risk = "LOW"
            risk_score = 1
        
        # Apply risk modifiers from FAA data if available
        risk_modifier = 1.0
        
        # Apply species risk modifier if available
        if species_group:
            # Here is where we call the helper function
            species_risk_factor = self._get_species_risk_factor(species_group)
            risk_modifier *= species_risk_factor
        
        # Apply seasonal risk modifier if available
        if season:
            # Here is where we call the helper function
            seasonal_risk_factor = self._get_seasonal_risk_factor(season)
            risk_modifier *= seasonal_risk_factor
        
        # Apply time of day modifier based on established aviation patterns
        if time_of_day:
            if time_of_day == "dawn" or time_of_day == "dusk":
                risk_modifier *= 1.5  # Higher risk during dawn/dusk (bird activity peaks)
            elif time_of_day == "night":
                risk_modifier *= 0.8  # Lower risk during night (reduced visibility but less bird activity)
        
        # Adjust final risk score
        adjusted_risk_score = risk_score * risk_modifier
        
        # Determine final risk category
        if adjusted_risk_score >= 2.5:
            return "HIGH"
        elif adjusted_risk_score >= 1.5:
            return "MEDIUM"
        else:
            return "LOW"
        
    def _get_species_risk_factor(self, species_group):
        """Get risk factor for a bird species based on FAA historical data"""
        if not hasattr(self, 'species_risk_profiles') or self.species_risk_profiles.empty:
            return 1.0
        
        if species_group in self.species_risk_profiles['SpeciesGroup'].values:
            return 1.0 + self.species_risk_profiles[
                self.species_risk_profiles['SpeciesGroup'] == species_group
            ]['RiskScore'].values[0]
        else:
            return 1.0

    def _get_seasonal_risk_factor(self, season):
        """Get risk factor for a season based on FAA historical data"""
        if 'temporal_risk_profiles' not in self.__dict__ or not self.temporal_risk_profiles:
            return 1.0
        
        if 'seasonal' in self.temporal_risk_profiles:
            seasonal_data = self.temporal_risk_profiles['seasonal']
            if season in seasonal_data['Season'].values:
                season_strikes = seasonal_data[seasonal_data['Season'] == season]['StrikeCount'].values[0]
                total_strikes = seasonal_data['StrikeCount'].sum()
                return season_strikes / total_strikes * 4  # Normalize to 4 seasons
        
        return 1.0
    
    def visualize_temporal_risk(self):
        """Visualize temporal patterns in bird strike risk"""
        if 'temporal_risk_profiles' not in self.__dict__ or not self.temporal_risk_profiles:
            print("No temporal risk profiles available. Run analyze_temporal_patterns() first.")
            return
        
        plt.figure(figsize=(15, 7))
        
        # Monthly patterns
        if 'monthly' in self.temporal_risk_profiles:
            monthly_data = self.temporal_risk_profiles['monthly']
            
            plt.subplot(121)
            sns.lineplot(x='Month', y='StrikeCount', data=monthly_data, marker='o', linewidth=2)
            plt.title('Monthly Bird Strike Frequency')
            plt.xlabel('Month')
            plt.ylabel('Number of Strikes')
            plt.xticks(range(1, 13))
        
        # Seasonal patterns
        if 'seasonal' in self.temporal_risk_profiles:
            seasonal_data = self.temporal_risk_profiles['seasonal']
            
            plt.subplot(122)
            sns.barplot(x='Season', y='StrikeCount', data=seasonal_data, 
                       order=['Winter', 'Spring', 'Summer', 'Fall'])
            plt.title('Seasonal Bird Strike Patterns')
            plt.xlabel('Season')
            plt.ylabel('Number of Strikes')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_risk_matrix(self, risk_matrix):
        """Visualize the risk matrix as a heatmap"""
        if risk_matrix is None:
            print("No risk matrix available.")
            return
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(risk_matrix, annot=True, cmap='YlOrRd', linewidths=.5, fmt='.2f')
        plt.title('Bird Strike Risk Matrix: Species vs. Flight Phase')
        plt.ylabel('Bird Species Group')
        plt.xlabel('Flight Phase')
        plt.tight_layout()
        plt.show()


# Example usage
def demo_faa_integration(faa_data_path="wildlife_strikes.csv"):
    """
    Demonstrate the integration of FAA wildlife strike data 
    with the Bird Strike Risk Assessment System
    """
    # Initialize the system with FAA data
    risk_system = BirdStrikeRiskSystem(faa_data_path)
    
    # Load and preprocess the FAA data
    if risk_system.load_faa_data():
        # Analyze risk patterns
        species_risk = risk_system.analyze_species_risk()
        temporal_patterns = risk_system.analyze_temporal_patterns()
        spatial_patterns = risk_system.analyze_spatial_patterns()
        condition_factors = risk_system.extract_strike_conditions()
        risk_matrix = risk_system.generate_risk_matrix()
        
        # Visualize results
        print("\nTop 5 highest risk bird species:")
        print(species_risk.head(5)[['SpeciesGroup', 'StrikeCount', 'DamagePercentage', 'RiskScore']])
        
        print("\nSeasonal risk patterns:")
        if temporal_patterns and 'seasonal' in temporal_patterns:
            print(temporal_patterns['seasonal'])
        
        print("\nTop 5 highest risk states:")
        if spatial_patterns and 'state' in spatial_patterns:
            print(spatial_patterns['state'].head(5))
        
        # Output visualization
        risk_system.visualize_species_risk()
        risk_system.visualize_temporal_risk()
        if risk_matrix is not None:
            risk_system.visualize_risk_matrix(risk_matrix)
        
        return risk_system
    else:
        print("Failed to load FAA data. Demo aborted.")
        return None


# This would be called in the main execution
if __name__ == "__main__":
    # Example path to the FAA wildlife strike CSV file
    faa_data_path = "wildlife_strikes.csv"
    
    # Run the demonstration
    risk_system = demo_faa_integration(faa_data_path)