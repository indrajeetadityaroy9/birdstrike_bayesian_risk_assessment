# main.py
import numpy as np
from scipy.optimize import minimize
from utility import calculate_distance_3d, compute_map_objective, evaluate_map_objective_grid
from visualization import plot_risk_assessment, plot_statistics
from bird_behavior import get_bird_priors, simulate_bird_movement
from statistics_module import RiskStatistics

def main():
    # Define the airport reference point (origin)
    airport_x, airport_y, airport_z = 0, 0, 0

    # Define standard deviations for Gaussian priors and measurement noise
    sigma_x, sigma_y, sigma_z, sigma_noise = 0.25, 0.25, 0.15, 0.3

    # Create evaluation grid for MAP objective function
    x_points = np.arange(-2, 2.05, 0.05)
    y_points = np.arange(-2, 2.05, 0.05)
    grid_x, grid_y = np.meshgrid(x_points, y_points)

    # Define airport runways
    runways = [
        {"name": "Runway 1", "start": (-1.5, 0, 0), "end": (1.5, 0, 0)},
        {"name": "Runway 2", "start": (0, -1.5, 0), "end": (0, 1.5, 0)}
    ]

    # Generate flight paths based on runways
    flight_paths = []
    for runway in runways:
        # Approach path (descending to runway)
        approach = {
            "name": f"Approach {runway['name']}",
            "start": (runway["start"][0] - 2, runway["start"][1], 0.5),
            "end": runway["start"],
            "width": 0.3
        }
        # Departure path (ascending from runway)
        departure = {
            "name": f"Departure {runway['name']}",
            "start": runway["end"],
            "end": (runway["end"][0] + 2, runway["end"][1], 0.5),
            "width": 0.3
        }
        flight_paths.extend([approach, departure])

    # Place K radar sensors in optimal positions around airport
    K = 8  # Number of sensors
    theta = np.linspace(0, 2 * np.pi, K + 1)[:-1]
    sensors_x = 2 * np.cos(theta)  # Larger radius to cover airport perimeter
    sensors_y = 2 * np.sin(theta)
    sensors_z = np.ones(K) * 0.1  # Height of sensors

    # Environmental factors that affect bird behavior
    time_of_day = "dawn"  # dawn, day, dusk, night
    season = "spring"     # spring, summer, fall, winter
    weather = "clear"     # clear, cloudy, rainy, foggy

    # Simulate multiple bird flocks
    num_flocks = 3
    np.random.seed(42)  # For reproducibility
    
    # Initialize bird flocks with positions outside airport perimeter
    true_flock_positions = []
    for i in range(num_flocks):
        r = 1.5 + 0.5 * np.random.rand()  # Distance from airport
        angle = np.random.rand() * 2 * np.pi
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        z = 0.2 + 0.4 * np.random.rand()  # Random altitude
        true_flock_positions.append((x, y, z))

    # Initialize statistics tracker
    stats = RiskStatistics(num_flocks, len(flight_paths))
        
    # Run simulation for multiple time steps
    num_time_steps = 5
    all_estimated_positions = []
    for t in range(num_time_steps):
        print(f"\n===== Time step {t+1}/{num_time_steps} =====")
        
        # Process each bird flock
        estimated_positions = []
        all_risk_levels = []
        
        for flock_idx, (x_t, y_t, z_t) in enumerate(true_flock_positions):
            print(f"\nProcessing Flock #{flock_idx+1}")
            print(f"True Position: ({x_t:.2f}, {y_t:.2f}, {z_t:.2f})")
            
            # Get bird behavior priors based on environmental factors
            prior_params = get_bird_priors(x_t, y_t, z_t, time_of_day, season, weather)
            
            # Generate noisy range measurements from each sensor
            measurements = []
            for i in range(K):
                true_distance = calculate_distance_3d(x_t, y_t, z_t, 
                                                     sensors_x[i], sensors_y[i], sensors_z[i])
                measured_range = np.random.normal(true_distance, sigma_noise)
                # Ensure measurements are physically meaningful (positive)
                while measured_range < 0:
                    measured_range = np.random.normal(true_distance, sigma_noise)
                measurements.append(measured_range)

            # Print measurement statistics
            print(f"Sensor Measurements:")
            print(f"  Min: {min(measurements):.2f} km")
            print(f"  Max: {max(measurements):.2f} km")
            print(f"  Avg: {np.mean(measurements):.2f} km")
            print(f"  Std: {np.std(measurements):.2f} km")

            # Evaluate MAP objective function on grid (for visualization)
            objective_values = evaluate_map_objective_grid(
                grid_x, grid_y, z_t, sensors_x, sensors_y, sensors_z, 
                measurements, sigma_x, sigma_y, sigma_z, sigma_noise, prior_params
            )

            # Use numerical optimization to find MAP estimate of bird position
            initial_guess = [0, 0, z_t]  # Start at airport with known altitude
            bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0.05, 2.0)]  # Constrain altitude
            
            # Find position that minimizes objective function (maximizes posterior probability)
            optimization_result = minimize(
                compute_map_objective,
                initial_guess,
                args=(sensors_x, sensors_y, sensors_z, measurements, 
                      sigma_x, sigma_y, sigma_z, sigma_noise, prior_params),
                bounds=bounds
            )
            
            estimated_position = optimization_result.x
            estimated_positions.append(estimated_position)
            
            # Print MAP estimation statistics
            print(f"MAP Estimation Results:")
            print(f"  Estimated Position: ({estimated_position[0]:.2f}, {estimated_position[1]:.2f}, {estimated_position[2]:.2f})")
            error_3d = calculate_distance_3d(x_t, y_t, z_t, 
                                            estimated_position[0], estimated_position[1], estimated_position[2])
            error_2d = calculate_distance_3d(x_t, y_t, 0, 
                                            estimated_position[0], estimated_position[1], 0)
            error_z = abs(z_t - estimated_position[2])
            
            print(f"  Position Error:")
            print(f"    3D Error: {error_3d:.2f} km")
            print(f"    2D Error: {error_2d:.2f} km")
            print(f"    Alt Error: {error_z:.2f} km")
            print(f"  Objective Function Value: {optimization_result.fun:.4f}")
            print(f"  Optimization Success: {optimization_result.success}")
            
            # Calculate risk levels for each flight path
            risk_levels = []
            print(f"Risk Assessment:")
            
            for path_idx, path in enumerate(flight_paths):
                risk = calculate_risk(estimated_position, path)
                risk_levels.append((path["name"], risk))
                print(f"  {path['name']}: {risk}")
                
                # Record statistics
                stats.record_risk(t, flock_idx, path_idx, risk)
            
            all_risk_levels.append(risk_levels)
            
            # Record position estimation error
            stats.record_position_error(t, flock_idx, error_3d, error_2d, error_z)

            # Visualize results for this flock
            plot_risk_assessment(
                grid_x, grid_y, objective_values, 
                true_flock_positions[flock_idx], estimated_position,
                sensors_x, sensors_y, sensors_z,
                runways, flight_paths, risk_levels, t, flock_idx
            )
        
        # Print time step summary
        high_risk_count = sum(1 for flock_risks in all_risk_levels 
                             for _, risk in flock_risks if risk == "HIGH")
        medium_risk_count = sum(1 for flock_risks in all_risk_levels 
                               for _, risk in flock_risks if risk == "MEDIUM")
        
        print(f"\nTime Step {t+1} Summary:")
        print(f"  HIGH Risk Incidents: {high_risk_count}")
        print(f"  MEDIUM Risk Incidents: {medium_risk_count}")
        print(f"  Total Flocks: {num_flocks}")
        
        # Update bird positions for next time step using movement model
        true_flock_positions = simulate_bird_movement(
            true_flock_positions, time_of_day, season, weather, flight_paths
        )
        
        # Record flock positions for movement analysis
        for flock_idx, pos in enumerate(true_flock_positions):
            stats.record_flock_position(t+1, flock_idx, pos[0], pos[1], pos[2])
    
    # Print final statistics
    print("\n==================================================")
    print("FINAL SIMULATION STATISTICS")
    print("==================================================")
    
    # Print risk statistics
    stats.print_risk_summary()
    
    # Print estimation accuracy
    stats.print_estimation_accuracy()
    
    # Print movement patterns
    stats.print_movement_patterns()
    
    # Plot final statistics
    plot_statistics(stats, flight_paths, num_time_steps)

def calculate_risk(bird_position, flight_path):
    """Calculate collision risk based on distance to flight path"""
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
    
    # Determine risk level based on distance and path width
    if distance < width / 2:
        return "HIGH"
    elif distance < width:
        return "MEDIUM"
    else:
        return "LOW"

if __name__ == "__main__":
    main()
