# utility.py
import numpy as np

def calculate_distance_3d(x1, y1, z1, x2, y2, z2):
    """Calculate 3D Euclidean distance between two points"""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

def compute_map_objective(bird_position, sensors_x, sensors_y, sensors_z, measurements, 
                          sigma_x, sigma_y, sigma_z, sigma_noise, prior_params):
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
        true_range = calculate_distance_3d(x, y, z, sensors_x[i], sensors_y[i], sensors_z[i])
        likelihood += ((measured_range - true_range)**2) / sigma_noise**2
    
    # Total objective function (negative log posterior)
    return prior + likelihood

def evaluate_map_objective_grid(grid_x, grid_y, z, sensors_x, sensors_y, sensors_z, 
                               measurements, sigma_x, sigma_y, sigma_z, sigma_noise, prior_params):
    """Evaluate MAP objective function over a grid (for visualization)"""
    objective_values = np.zeros_like(grid_x)
    
    for i in range(grid_x.shape[0]):
        for j in range(grid_y.shape[1]):
            x, y = grid_x[i, j], grid_y[i, j]
            objective_values[i, j] = compute_map_objective(
                [x, y, z], sensors_x, sensors_y, sensors_z, measurements,
                sigma_x, sigma_y, sigma_z, sigma_noise, prior_params
            )
    
    return objective_values