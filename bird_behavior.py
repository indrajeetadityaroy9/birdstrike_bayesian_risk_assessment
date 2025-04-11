# bird_behavior.py
import numpy as np
import pandas as pd

class MigrationDataProcessor:
    def __init__(self, migration_data_path):
        """Initialize with bird migration dataset"""
        self.data = pd.read_csv(migration_data_path)
        self.data['date_time'] = pd.to_datetime(self.data['date_time'])
        self._preprocess_data()
        self._extract_movement_patterns()
        
    def _preprocess_data(self):
        """Preprocess the migration data"""
        # Add derived features
        self.data['hour_of_day'] = self.data['date_time'].dt.hour
        self.data['day_of_year'] = self.data['date_time'].dt.dayofyear
        
        # Determine time of day
        self.data['time_of_day'] = self.data['hour_of_day'].apply(
            lambda h: 'dawn' if 5 <= h < 8 else
                     'day' if 8 <= h < 18 else
                     'dusk' if 18 <= h < 21 else 'night'
        )
        
        # Determine season based on day of year
        self.data['season'] = self.data['day_of_year'].apply(
            lambda d: 'spring' if 80 <= d < 172 else
                     'summer' if 172 <= d < 266 else
                     'fall' if 266 <= d < 355 else 'winter'
        )
        
        # Calculate movement deltas (with coordinate conversion)
        self.data = self.data.sort_values(['bird_name', 'date_time'])
        
        # Calculate position changes (per hour)
        self.data['delta_lat'] = self.data.groupby('bird_name')['latitude'].diff()
        self.data['delta_lon'] = self.data.groupby('bird_name')['longitude'].diff() 
        self.data['delta_alt'] = self.data.groupby('bird_name')['altitude'].diff()
        
        # Calculate time differences in hours
        self.data['time_diff'] = self.data.groupby('bird_name')['date_time'].diff().dt.total_seconds() / 3600
        
        # Calculate movement rates (position change per hour)
        self.data['lat_rate'] = self.data['delta_lat'] / self.data['time_diff']
        self.data['lon_rate'] = self.data['delta_lon'] / self.data['time_diff']
        self.data['alt_rate'] = self.data['delta_alt'] / self.data['time_diff']
        
        # Clean up NaNs
        self.data = self.data.dropna(subset=['lat_rate', 'lon_rate', 'alt_rate'])
    
    def _extract_movement_patterns(self):
        """Extract statistical patterns of bird movements"""
        # Create multi-level dictionary to store patterns
        self.movement_patterns = {}
        
        # Group by bird, season, and time of day
        for bird in self.data['bird_name'].unique():
            self.movement_patterns[bird] = {}
            bird_data = self.data[self.data['bird_name'] == bird]
            
            for season in ['spring', 'summer', 'fall', 'winter']:
                self.movement_patterns[bird][season] = {}
                season_data = bird_data[bird_data['season'] == season]
                
                for time in ['dawn', 'day', 'dusk', 'night']:
                    time_data = season_data[season_data['time_of_day'] == time]
                    
                    # If we have enough data points, calculate parameters
                    if len(time_data) >= 5:
                        # Calculate means and standard deviations for each dimension
                        self.movement_patterns[bird][season][time] = {
                            # Movement rates
                            'mean_lat_rate': time_data['lat_rate'].mean(),
                            'mean_lon_rate': time_data['lon_rate'].mean(),
                            'mean_alt_rate': time_data['alt_rate'].mean(),
                            'std_lat_rate': max(0.001, time_data['lat_rate'].std()),
                            'std_lon_rate': max(0.001, time_data['lon_rate'].std()),
                            'std_alt_rate': max(0.001, time_data['alt_rate'].std()),
                            
                            # Position preferences
                            'mean_alt': time_data['altitude'].mean(),
                            'std_alt': max(1.0, time_data['altitude'].std()),
                            
                            # Other movement characteristics
                            'mean_speed': time_data['speed_2d'].mean(),
                            'activity_level': time_data['speed_2d'].mean() / 5.0  # Normalize
                        }
                    else:
                        # Default patterns if not enough data
                        self.movement_patterns[bird][season][time] = {
                            'mean_lat_rate': 0.0,
                            'mean_lon_rate': 0.0,
                            'mean_alt_rate': 0.0,
                            'std_lat_rate': 0.01,
                            'std_lon_rate': 0.01,
                            'std_alt_rate': 0.01,
                            'mean_alt': 100.0,
                            'std_alt': 20.0,
                            'mean_speed': 0.5,
                            'activity_level': 0.1
                        }
    
    def get_movement_pattern(self, bird_type, time_of_day, season):
        """Get movement pattern for a specific bird type, time, and season"""
        # Map bird type to available data
        if bird_type.lower() in ['gull', 'seagull']:
            model_bird = 'Eric'  # Map to closest tracked bird
        elif bird_type.lower() in ['goose', 'geese']:
            model_bird = 'Sanne'
        else:
            model_bird = 'Nico'
        
        # Return the movement pattern
        return self.movement_patterns[model_bird][season][time_of_day]

migration_data = MigrationDataProcessor('bird_migration.csv')

# Update these functions in bird_behavior.py

def get_bird_priors(x, y, z, time_of_day, season, weather, bird_type='Gull', migration_data=None):
    """
    Generate prior distribution parameters for bird positions
    based on real migration data to enhance existing Bayesian approach
    """
    # If migration_data is not provided, try to use global migration_data if available
    if migration_data is None:
        return {
                'mu': (x, y, z),
                'activity_level': 0.5
            }
    
    # Get movement patterns from real migration data
    pattern = migration_data.get_movement_pattern(bird_type, time_of_day, season)
    
    # Weather adjustment factors
    weather_factors = {
        'clear': 1.0,
        'cloudy': 0.9,
        'rainy': 0.7,
        'foggy': 0.6
    }
    weather_factor = weather_factors.get(weather, 1.0)
    
    # Adjust means based on movement tendencies (convert to our coordinate system)
    # Note: This is a simplified conversion between lat/lon and x/y
    # In a real system, you'd use proper coordinate conversion
    mu_x = x + pattern['mean_lon_rate'] * weather_factor * 0.01
    mu_y = y + pattern['mean_lat_rate'] * weather_factor * 0.01
    
    # Blend current altitude with preferred altitude from data
    preferred_z = pattern['mean_alt'] / 1000.0  # Convert to km
    mu_z = z * 0.7 + preferred_z * 0.3  # Weighted average
    
    # Adjust for altitude tendency
    mu_z += pattern['mean_alt_rate'] * weather_factor * 0.0001  # Small adjustment
    
    # Ensure altitude is reasonable
    mu_z = max(0.05, min(2.0, mu_z))
    
    # Return prior parameters in the format expected by the existing system
    return {
        'mu': (mu_x, mu_y, mu_z),
        'activity_level': pattern['activity_level'] * weather_factor
    }

def simulate_bird_movement(current_positions, time_of_day, season, weather, flight_paths, migration_data=None):
    """
    Simulate bird movement using statistics from real migration data
    while maintaining the existing approach
    """
    # If migration_data is not provided, try to use global migration_data if available
    if migration_data is None:
       return [(x + np.random.normal(0, 0.1), 
                    y + np.random.normal(0, 0.1), 
                    max(0.05, min(2.0, z + np.random.normal(0, 0.05)))) 
                    for x, y, z in current_positions]
    
    bird_types = ["Gull", "Goose", "Hawk"]  # Assigned to flocks
    new_positions = []
    
    for i, (x, y, z) in enumerate(current_positions):
        bird_type = bird_types[i % len(bird_types)]
        
        # Get real-data informed movement pattern
        pattern = migration_data.get_movement_pattern(bird_type, time_of_day, season)
        
        # Weather adjustment factors
        weather_factors = {
            'clear': 1.0,
            'cloudy': 0.9,
            'rainy': 0.7,
            'foggy': 0.6
        }
        weather_factor = weather_factors.get(weather, 1.0)
        
        # Calculate movement based on patterns from real data
        delta_x = np.random.normal(
            pattern['mean_lon_rate'] * 0.01 * weather_factor,
            pattern['std_lon_rate'] * 0.01
        )
        delta_y = np.random.normal(
            pattern['mean_lat_rate'] * 0.01 * weather_factor,
            pattern['std_lat_rate'] * 0.01
        )
        delta_z = np.random.normal(
            pattern['mean_alt_rate'] * 0.0001 * weather_factor,
            pattern['std_alt_rate'] * 0.0001
        )
        
        # Apply movement
        new_x = x + delta_x
        new_y = y + delta_y
        new_z = z + delta_z
        
        # Add flight path awareness (birds try to avoid aircraft)
        for path in flight_paths:
            # Check if bird is near a flight path
            distance_to_path = calculate_distance_to_path((new_x, new_y, new_z), path)
            if distance_to_path < path["width"] * 1.5:
                # If too close, adjust position to move away
                start_x, start_y, start_z = path["start"]
                end_x, end_y, end_z = path["end"]
                
                # Calculate vector from path to bird
                path_center_x = (start_x + end_x) / 2
                path_center_y = (start_y + end_y) / 2
                path_center_z = (start_z + end_z) / 2
                
                away_vector_x = new_x - path_center_x
                away_vector_y = new_y - path_center_y
                away_vector_z = new_z - path_center_z
                
                # Normalize the vector
                magnitude = np.sqrt(away_vector_x**2 + away_vector_y**2 + away_vector_z**2)
                if magnitude > 0:
                    away_vector_x /= magnitude
                    away_vector_y /= magnitude
                    away_vector_z /= magnitude
                
                # Adjust position to move away slightly
                evasion_strength = 0.1 * (path["width"] * 1.5 - distance_to_path)
                new_x += away_vector_x * evasion_strength
                new_y += away_vector_y * evasion_strength
                new_z += away_vector_z * evasion_strength
        
        # Ensure position is valid
        new_z = max(0.05, min(2.0, new_z))
        
        new_positions.append((new_x, new_y, new_z))
    
    return new_positions

def calculate_distance_to_path(position, path):
    """Calculate distance from a position to a flight path"""
    x, y, z = position
    start_x, start_y, start_z = path["start"]
    end_x, end_y, end_z = path["end"]
    
    # Calculate vector along path
    v_x = end_x - start_x
    v_y = end_y - start_y
    v_z = end_z - start_z
    
    # Parameter of closest point on line
    denominator = v_x * v_x + v_y * v_y + v_z * v_z
    if denominator == 0:
        # Handle degenerate case (start and end are the same point)
        return np.sqrt((x - start_x)**2 + (y - start_y)**2 + (z - start_z)**2)
    
    t = ((x - start_x) * v_x + (y - start_y) * v_y + (z - start_z) * v_z) / denominator
    t = max(0, min(1, t))  # Constrain to line segment
    
    # Calculate closest point on path
    closest_x = start_x + t * v_x
    closest_y = start_y + t * v_y
    closest_z = start_z + t * v_z
    
    # Calculate distance to path
    return np.sqrt((x - closest_x)**2 + (y - closest_y)**2 + (z - closest_z)**2)