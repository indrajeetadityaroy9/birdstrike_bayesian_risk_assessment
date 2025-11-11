import numpy as np

def calculate_distance_3d(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

def calculate_distance_to_path(position, path):
    p = np.array(position)
    a = np.array(path["start"])
    b = np.array(path["end"])
    ap = p - a
    ab = b - a
    ab_squared = np.dot(ab, ab)
    if ab_squared < 1e-12:
        return np.linalg.norm(p - a)
    t = np.dot(ap, ab) / ab_squared
    t_clamped = np.clip(t, 0.0, 1.0)
    closest_point = a + t_clamped * ab
    return np.linalg.norm(p - closest_point)

def compute_map_objective(bird_position, sensors_pos, measurements, sigma_prior_sq, sigma_noise_sq, prior_mu):
    x, y, z = bird_position
    mux, muy, muz = prior_mu
    spx2, spy2, spz2 = sigma_prior_sq
    prior_term = 0.5 * (((x - mux) ** 2 / spx2 if spx2 > 0 else 0) + ((y - muy) ** 2 / spy2 if spy2 > 0 else 0) + (
        (z - muz) ** 2 / spz2 if spz2 > 0 else 0))
    likelihood_term = 0
    num_sensors = sensors_pos.shape[0]
    for i in range(num_sensors):
        sensor_loc = sensors_pos[i, :]
        measured_range = measurements[i]
        if measured_range is not None and not np.isnan(measured_range):
            true_range = calculate_distance_3d(x, y, z, *sensor_loc)
            likelihood_term += 0.5 * ((measured_range - true_range) ** 2 / sigma_noise_sq)
    return likelihood_term + prior_term

def evaluate_map_objective_grid(grid_x, grid_y, z_level, sensors_pos, measurements, sigma_prior_sq, sigma_noise_sq, prior_mu):
    objective_values = np.zeros_like(grid_x)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            bird_pos_ij = np.array([grid_x[i, j], grid_y[i, j], z_level])
            objective_values[i, j] = compute_map_objective(bird_pos_ij, sensors_pos, measurements, sigma_prior_sq, sigma_noise_sq, prior_mu)
    return objective_values
