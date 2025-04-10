# bird_behavior.py
import numpy as np

def get_bird_priors(x, y, z, time_of_day, season, weather):
    """
    Generate prior distribution parameters for bird positions
    based on environmental factors and bird behavior models.
    """
    # Initialize prior means at current position
    mu_x, mu_y, mu_z = x, y, z
    
    # Adjust for time of day (activity levels)
    if time_of_day == "dawn" or time_of_day == "dusk":
        # Birds are more active at dawn/dusk
        activity_level = 0.8
    elif time_of_day == "day":
        activity_level = 0.6
    else:  # night
        activity_level = 0.3
    
    # Adjust for seasonal patterns
    seasonal_factor = 1.0
    if season == "spring" or season == "fall":
        # Migration seasons - more movement
        seasonal_factor = 1.2
    elif season == "winter":
        seasonal_factor = 0.8
    
    # Adjust for weather conditions
    weather_factor = 1.0
    if weather == "rainy" or weather == "foggy":
        # Birds fly lower in poor weather
        weather_factor = 0.7
        mu_z *= 0.8  # Lower altitude prior
    
    # Combined behavior factor
    combined_factor = activity_level * seasonal_factor * weather_factor
    
    return {
        'mu': (mu_x, mu_y, mu_z),
        'activity_level': combined_factor
    }

def simulate_bird_movement(current_positions, time_of_day, season, weather, flight_paths):
    """
    Simulate realistic bird movement patterns based on environmental factors.
    """
    new_positions = []
    
    # Calculate movement speed based on environmental factors
    speed_factor = 1.0
    if time_of_day == "dawn" or time_of_day == "dusk":
        speed_factor = 1.2
    elif time_of_day == "night":
        speed_factor = 0.6
        
    if season == "spring" or season == "fall":
        speed_factor *= 1.1
    
    if weather == "rainy" or weather == "foggy":
        speed_factor *= 0.8
    
    # Movement parameters
    base_speed = 0.2 * speed_factor
    noise_scale = 0.1
    
    # Update each flock's position
    for x, y, z in current_positions:
        # Random directional movement with inertia
        dx = np.random.normal(0, noise_scale)
        dy = np.random.normal(0, noise_scale)
        dz = np.random.normal(0, noise_scale * 0.5)  # Less vertical movement
        
        # Normalize and scale by speed
        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        if norm > 0:
            dx = (dx / norm) * base_speed
            dy = (dy / norm) * base_speed
            dz = (dz / norm) * base_speed
        
        # Apply constraints
        # 1. Stay above ground
        new_z = max(0.05, z + dz)
        
        # 2. Tendency to maintain reasonable altitudes
        if new_z > 1.5:
            new_z = min(new_z, z + dz * 0.5)  # Bias downward if too high
        
        # Update position
        new_x = x + dx
        new_y = y + dy
        
        new_positions.append((new_x, new_y, new_z))
    
    return new_positions