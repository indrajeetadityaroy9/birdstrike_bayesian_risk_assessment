# main.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from utility import calculate_distance_3d, evaluate_map_objective_grid, compute_map_objective
from visualization import plot_risk_assessment
from bird_behavior import MigrationDataProcessor, get_bird_priors, simulate_bird_movement, calculate_distance_to_path
from statistics_module import RiskStatistics
from bird_strike_system import BirdStrikeRiskSystem

def get_seasonal_risk_summary(faa_risk_system, season):
    """
    Generate a seasonal risk summary from the FAA data
    """
    if not hasattr(faa_risk_system, 'temporal_risk_profiles') or not faa_risk_system.temporal_risk_profiles:
        return "No seasonal data available"
        
    if 'seasonal' in faa_risk_system.temporal_risk_profiles:
        seasonal_data = faa_risk_system.temporal_risk_profiles['seasonal']
        if 'Season' in seasonal_data.columns and season in seasonal_data['Season'].values:
            season_data = seasonal_data[seasonal_data['Season'] == season]
            strikes = season_data['StrikeCount'].values[0]
            total_strikes = seasonal_data['StrikeCount'].sum()
            percentage = (strikes / total_strikes) * 100
            avg_damage = season_data['AvgDamageSeverity'].values[0]
            
            return f"{season} represents {percentage:.1f}% of annual bird strikes with average damage severity of {avg_damage:.2f}"
    
    return f"No specific data available for {season}"

def print_species_risk_summary(faa_risk_system, species):
    """
    Print a risk summary for a specific bird species
    """
    if not hasattr(faa_risk_system, 'species_risk_profiles') or not isinstance(faa_risk_system.species_risk_profiles, pd.DataFrame):
        print(f"No species risk data available for {species}")
        return
        
    if 'SpeciesGroup' in faa_risk_system.species_risk_profiles.columns and species in faa_risk_system.species_risk_profiles['SpeciesGroup'].values:
        species_data = faa_risk_system.species_risk_profiles[
            faa_risk_system.species_risk_profiles['SpeciesGroup'] == species]
        
        strikes = species_data['StrikeCount'].values[0]
        damage_pct = species_data['DamagePercentage'].values[0]
        risk_score = species_data['RiskScore'].values[0]
        
        print(f"Species: {species}")
        print(f"  Historical Strikes: {strikes}")
        print(f"  Damage Rate: {damage_pct:.1f}%")
        print(f"  Risk Score: {risk_score:.4f}")
    else:
        print(f"No specific risk data available for {species}")

def main():
    # Define the airport reference point (origin)
    airport_x, airport_y, airport_z = 0, 0, 0

    # Define standard deviations for Gaussian priors and measurement noise
    sigma_x, sigma_y, sigma_z, sigma_noise = 0.25, 0.25, 0.15, 0.3

    # Initialize the migration data processor with robust error handling
    try:
        migration_processor = MigrationDataProcessor('bird_migration.csv')
        print("Successfully loaded bird migration data")
    except Exception as e:
        print(f"Error loading bird migration data: {str(e)}")
        print("Creating fallback migration data")
        # Create a minimal dummy dataset
        dummy_data = pd.DataFrame({
            'bird_name': ['Eric', 'Sanne', 'Nico'] * 10,
            'date_time': pd.date_range(start='2023-01-01', periods=30, freq='D'),
            'latitude': np.random.normal(50, 0.5, 30),
            'longitude': np.random.normal(5, 0.5, 30),
            'altitude': np.random.normal(100, 20, 30),
            'speed_2d': np.random.normal(10, 2, 30)
        })
        # Save the dummy data as CSV
        dummy_data.to_csv('bird_migration.csv', index=False)
        # Try again with the dummy data
        migration_processor = MigrationDataProcessor('bird_migration.csv')

    # Initialize the FAA risk assessment system with robust error handling
    try:
        faa_risk_system = BirdStrikeRiskSystem("wildlife_strikes.csv")
        data_loaded = faa_risk_system.load_faa_data()
        if data_loaded:
            faa_risk_system.analyze_species_risk()
            faa_risk_system.analyze_temporal_patterns()
            faa_risk_system.analyze_spatial_patterns()
        else:
            print("Warning: Proceeding without FAA data integration")
    except Exception as e:
        print(f"Error initializing FAA risk system: {str(e)}")
        print("Continuing with Bayesian detection only")
        faa_risk_system = None
    
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

    # Assign species to flocks based on local observations or eBird data
    flock_species = ["Gull", "Goose", "Hawk"]
    
    # Initialize statistics tracker
    stats = RiskStatistics(num_flocks, len(flight_paths))
        
    # Run simulation for multiple time steps
    num_time_steps = 5
    all_risk_levels = []  # Store risk levels for each time step for summary
    
    for t in range(num_time_steps):
        print(f"\n===== Time step {t+1}/{num_time_steps} =====")
        time_step_risk_levels = []  # Store risk levels for this time step
        
        # Process each bird flock
        for flock_idx, (x_t, y_t, z_t) in enumerate(true_flock_positions):
            print(f"\nProcessing Flock #{flock_idx+1} - Species: {flock_species[flock_idx]}")
            print(f"True Position: ({x_t:.2f}, {y_t:.2f}, {z_t:.2f})")
            
            # Get bird behavior priors based on environmental factors
            # Pass the migration_processor instance explicitly
            prior_params = get_bird_priors(
                x_t, y_t, z_t, 
                time_of_day, 
                season, 
                weather, 
                bird_type=flock_species[flock_idx],
                migration_data=migration_processor
            )
            
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
            
            # Record position estimation error
            stats.record_position_error(t, flock_idx, error_3d, error_2d, error_z)
            
            # *** INTEGRATION POINT 1: Use FAA data to enhance risk assessment ***
            # Calculate risk levels for each flight path using enhanced risk calculation
            risk_levels = []
            print(f"Risk Assessment:")
            
            # Process each flight path
            for path_idx, path in enumerate(flight_paths):
                try:
                    # Try using FAA-enhanced risk calculation if available
                    if faa_risk_system is not None:
                        try:
                            risk = faa_risk_system.calculate_risk(
                                estimated_position, 
                                path, 
                                species_group=flock_species[flock_idx],
                                time_of_day=time_of_day,
                                season=season
                            )
                        except Exception as e:
                            print(f"Error in FAA risk calculation: {str(e)}")
                            # Fall back to basic distance calculation
                            distance = calculate_distance_to_path(estimated_position, path)
                            width = path["width"]
                            risk = "HIGH" if distance < width/2 else "MEDIUM" if distance < width else "LOW"
                    else:
                        # Fall back to basic distance calculation
                        distance = calculate_distance_to_path(estimated_position, path)
                        width = path["width"]
                        risk = "HIGH" if distance < width/2 else "MEDIUM" if distance < width else "LOW"
                except Exception as e:
                    print(f"Error calculating risk for path {path['name']}: {str(e)}")
                    risk = "LOW"  # Default to low in case of errors
                
                # Add the calculated risk to our list
                risk_levels.append((path["name"], risk))
                print(f"  {path['name']}: {risk}")
                
                # Record statistics - this is critical for the visualization
                stats.record_risk(t, flock_idx, path_idx, risk)
            
            time_step_risk_levels.append(risk_levels)
            
            # Visualize results for this flock
            try:
                plot_risk_assessment(
                    grid_x, grid_y, objective_values, 
                    true_flock_positions[flock_idx], estimated_position,
                    sensors_x, sensors_y, sensors_z,
                    runways, flight_paths, risk_levels, t, flock_idx
                )
            except Exception as e:
                print(f"Error in visualization: {str(e)}")
        
        all_risk_levels.append(time_step_risk_levels)
        
        # *** INTEGRATION POINT 2: Create enriched summary with FAA context ***
        # Count risk incidents at this time step
        try:
            if time_step_risk_levels:
                # Flatten the nested list of risks
                flat_risks = [risk for flock_risks in time_step_risk_levels for _, risk in flock_risks]
                high_risk_count = flat_risks.count("HIGH")
                medium_risk_count = flat_risks.count("MEDIUM")
                
                print(f"\nTime Step {t+1} Summary:")
                print(f"  HIGH Risk Incidents: {high_risk_count}")
                print(f"  MEDIUM Risk Incidents: {medium_risk_count}")
                print(f"  Total Flocks: {num_flocks}")
                
                # Add FAA context if available
                if faa_risk_system is not None:
                    context = get_seasonal_risk_summary(faa_risk_system, season)
                    print(f"  FAA Historical Context: {context}")
            else:
                print(f"\nTime Step {t+1} Summary: No risk data available")
        except Exception as e:
            print(f"Error generating time step summary: {str(e)}")
        
        # Update bird positions for next time step using movement model
        # Pass the migration_processor instance explicitly
        true_flock_positions = simulate_bird_movement(
            true_flock_positions,
            time_of_day,
            season,
            weather,
            flight_paths,
            migration_data=migration_processor
        )
        
        # Record flock positions for movement analysis
        for flock_idx, pos in enumerate(true_flock_positions):
            stats.record_flock_position(t+1, flock_idx, pos[0], pos[1], pos[2])
    
    # We need to add error checking to the statistics methods
    # Since we might have zero observations
    # Make sure there's at least one risk record
    if not stats.position_errors:
        # Add a dummy record to prevent empty array errors
        stats.position_errors[(0, 0)] = (0.1, 0.1, 0.0)
    
    # *** INTEGRATION POINT 3: Generate comprehensive final statistics ***
    # Print final statistics that include both real-time and historical data
    print("\n==================================================")
    print("FINAL SIMULATION STATISTICS WITH FAA CONTEXT")
    print("==================================================")
    
    # Print risk statistics with error handling
    try:
        stats.print_risk_summary()
    except Exception as e:
        print(f"Error in risk summary: {str(e)}")
        print("Risk data may be incomplete.")
    
    # Print estimation accuracy with error handling
    try:
        stats.print_estimation_accuracy()
    except Exception as e:
        print(f"Error in estimation accuracy: {str(e)}")
        print("Error statistics may be incomplete.")
    
    # Print species-specific risk analysis from FAA data
    if faa_risk_system is not None:
        for i, species in enumerate(flock_species):
            print(f"\nSpecies-specific risk for Flock #{i+1} ({species}):")
            print_species_risk_summary(faa_risk_system, species)
    
    # Print an overall assessment of the simulation
    print("\n==================================================")
    print("OVERALL RISK ASSESSMENT")
    print("==================================================")
    
    # Count total risk incidents across all time steps
    try:
        if all_risk_levels:
            all_flat_risks = []
            for time_risks in all_risk_levels:
                for flock_risks in time_risks:
                    for _, risk in flock_risks:
                        all_flat_risks.append(risk)
            
            total_high = all_flat_risks.count("HIGH")
            total_medium = all_flat_risks.count("MEDIUM")
            total_low = all_flat_risks.count("LOW")
            
            print(f"Total risk incidents across simulation:")
            print(f"  HIGH: {total_high}")
            print(f"  MEDIUM: {total_medium}")
            print(f"  LOW: {total_low}")
        else:
            print("No risk data available for summary.")
    except Exception as e:
        print(f"Error in risk calculation: {str(e)}")
    
    return {
        "statistics": stats,
        "faa_analysis": faa_risk_system,
        "risk_levels": all_risk_levels
    }

if __name__ == "__main__":
    main()