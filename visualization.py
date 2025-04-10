# visualization.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_risk_assessment(grid_x, grid_y, objective_values, true_position, estimated_position,
                        sensors_x, sensors_y, sensors_z, runways, flight_paths, risk_levels,
                        time_step, flock_idx):
    """Visualize the bird strike risk assessment with 2D and 3D views"""
    fig = plt.figure(figsize=(18, 9))
    
    # 2D top-down view (MAP objective contour)
    ax1 = fig.add_subplot(1, 2, 1)
    
    # Plot MAP objective function as contour
    contour_levels = 50
    contour = ax1.contourf(grid_x, grid_y, objective_values, levels=contour_levels, 
                         cmap='viridis', alpha=0.7)
    fig.colorbar(contour, ax=ax1, label='MAP objective value (lower is more probable)')
    
    # Plot airport and runways
    ax1.plot(0, 0, 'ko', markersize=10, label='Airport Reference')
    
    for runway in runways:
        ax1.plot([runway["start"][0], runway["end"][0]], 
                [runway["start"][1], runway["end"][1]], 
                'k-', linewidth=5, label=runway["name"] if runway == runways[0] else "")
    
    # Plot flight paths
    for path in flight_paths:
        if "Approach" in path["name"]:
            linestyle = '--'
            color = 'b'
        else:  # Departure
            linestyle = '-'
            color = 'g'
        
        ax1.plot([path["start"][0], path["end"][0]], 
                [path["start"][1], path["end"][1]], 
                color=color, linestyle=linestyle, linewidth=2, 
                label=path["name"] if path == flight_paths[0] or path == flight_paths[2] else "")
    
    # Plot sensors
    ax1.scatter(sensors_x, sensors_y, marker='^', s=80, color='cyan', label='Radar Sensors')
    
    # Plot bird positions
    ax1.scatter(true_position[0], true_position[1], marker='o', s=100, color='red', 
               label=f'True Flock Position (alt: {true_position[2]:.2f})')
    ax1.scatter(estimated_position[0], estimated_position[1], marker='x', s=100, color='white', 
               label=f'Estimated Position (alt: {estimated_position[2]:.2f})')
    
    # Add risk assessment text
    risk_text = "Risk Assessment:\n"
    for path_name, risk in risk_levels:
        color = 'red' if risk == "HIGH" else ('orange' if risk == "MEDIUM" else 'green')
        risk_text += f"{path_name}: {risk}\n"
    
    ax1.text(0.02, 0.02, risk_text, transform=ax1.transAxes, 
            bbox=dict(facecolor='white', alpha=0.7), verticalalignment='bottom')
    
    ax1.set_xlabel('x (km)')
    ax1.set_ylabel('y (km)')
    ax1.set_title(f'Bird Strike Risk Assessment - Time Step {time_step+1}, Flock {flock_idx+1}')
    ax1.grid(True)
    ax1.legend(loc='upper right')
    
    # 3D view
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Plot airport and sensors
    ax2.scatter(0, 0, 0, c='k', marker='o', s=100, label='Airport Reference')
    ax2.scatter(sensors_x, sensors_y, sensors_z, c='cyan', marker='^', s=50, label='Radar Sensors')
    
    # Plot runways
    for runway in runways:
        ax2.plot([runway["start"][0], runway["end"][0]], 
                [runway["start"][1], runway["end"][1]], 
                [runway["start"][2], runway["end"][2]], 
                'k-', linewidth=3)
    
    # Plot flight paths with risk zones
    for path in flight_paths:
        # Find risk level for this path
        current_risk = "LOW"  # Default
        for path_name, risk in risk_levels:
            if path["name"] == path_name:
                current_risk = risk
                break
        
        # Color based on risk
        if current_risk == "HIGH":
            color = 'red'
        elif current_risk == "MEDIUM":
            color = 'orange'
        else:
            color = 'green'
        
        # Plot path centerline
        ax2.plot([path["start"][0], path["end"][0]], 
                [path["start"][1], path["end"][1]], 
                [path["start"][2], path["end"][2]], 
                color=color, linewidth=2)
        
        # Simple tube representation for flight path
        # (This is a simplified visualization of flight path volume)
        width = path["width"]
        t_values = np.linspace(0, 1, 5)
        theta_values = np.linspace(0, 2*np.pi, 8)
        
        for t in t_values:
            for theta in theta_values:
                # Position along flight path
                px = path["start"][0] + t * (path["end"][0] - path["start"][0])
                py = path["start"][1] + t * (path["end"][1] - path["start"][1])
                pz = path["start"][2] + t * (path["end"][2] - path["start"][2])
                
                # Radial position (simplistic tube)
                px += width/2 * np.cos(theta)
                py += width/2 * np.sin(theta)
                
                ax2.scatter(px, py, pz, c=color, alpha=0.2, s=10)
    
    # Plot bird positions
    ax2.scatter(true_position[0], true_position[1], true_position[2], 
               c='red', marker='o', s=100, label='True Flock Position')
    ax2.scatter(estimated_position[0], estimated_position[1], estimated_position[2], 
               c='white', marker='x', s=100, label='Estimated Position')
    
    # Connect true and estimated positions
    ax2.plot([true_position[0], estimated_position[0]],
            [true_position[1], estimated_position[1]],
            [true_position[2], estimated_position[2]],
            'k--', alpha=0.5)
    
    ax2.set_xlabel('x (km)')
    ax2.set_ylabel('y (km)')
    ax2.set_zlabel('z (km)')
    ax2.set_title('3D View of Bird Positions and Flight Paths')
    
    ax2.set_xlim([-3, 3])
    ax2.set_ylim([-3, 3])
    ax2.set_zlim([0, 2])
    
    ax2.view_init(elev=30, azim=-60)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def plot_statistics(stats, flight_paths, num_time_steps):
    """Plot statistical summary of the simulation"""
    fig = plt.figure(figsize=(15, 15))
    
    # Plot 1: Risk trends over time
    ax1 = fig.add_subplot(2, 2, 1)
    time_steps = range(1, num_time_steps + 1)
    high_risks = []
    medium_risks = []
    low_risks = []
    
    for t in range(num_time_steps):
        high, medium, low = stats.get_risk_counts_by_time(t)
        high_risks.append(high)
        medium_risks.append(medium)
        low_risks.append(low)
    
    ax1.bar(time_steps, high_risks, label='HIGH', color='red', width=0.25, align='center')
    ax1.bar([t + 0.25 for t in time_steps], medium_risks, label='MEDIUM', color='orange', width=0.25, align='center')
    ax1.bar([t + 0.5 for t in time_steps], low_risks, label='LOW', color='green', width=0.25, align='center')
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Number of Incidents')
    ax1.set_title('Risk Level Trend Over Time')
    ax1.set_xticks(time_steps)
    ax1.legend()
    ax1.grid(axis='y')
    
    # Plot 2: Risk by flight path
    ax2 = fig.add_subplot(2, 2, 2)
    path_labels = [p["name"] for p in flight_paths]
    high_by_path = []
    medium_by_path = []
    low_by_path = []
    
    for i in range(len(flight_paths)):
        high, medium, low = stats.get_risk_counts_by_path(i)
        high_by_path.append(high)
        medium_by_path.append(medium)
        low_by_path.append(low)
    
    path_indices = range(len(flight_paths))
    ax2.bar(path_indices, high_by_path, label='HIGH', color='red', width=0.25, align='center')
    ax2.bar([i + 0.25 for i in path_indices], medium_by_path, label='MEDIUM', color='orange', width=0.25, align='center')
    ax2.bar([i + 0.5 for i in path_indices], low_by_path, label='LOW', color='green', width=0.25, align='center')
    
    ax2.set_xlabel('Flight Path')
    ax2.set_ylabel('Number of Incidents')
    ax2.set_title('Risk Level by Flight Path')
    ax2.set_xticks(path_indices)
    ax2.set_xticklabels([f"Path {i+1}" for i in path_indices], rotation=45)
    ax2.legend()
    ax2.grid(axis='y')
    
    # Plot 3: Position estimation errors over time
    ax3 = fig.add_subplot(2, 2, 3)
    error_by_time = stats.get_position_error_by_time()
    time_steps = sorted(error_by_time.keys())
    errors = [error_by_time[t] for t in time_steps]
    
    ax3.plot(time_steps, errors, 'bo-', linewidth=2, markersize=8)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Average 3D Error (km)')
    ax3.set_title('Position Estimation Error Over Time')
    ax3.set_xticks([t for t in time_steps])
    ax3.set_xticklabels([f"{t+1}" for t in time_steps])
    ax3.grid(True)
    
    # Plot 4: Flock movement visualization
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    
    for flock_idx in range(stats.num_flocks):
        positions = [(key[0], val) for key, val in stats.flock_positions.items() 
                   if key[1] == flock_idx]
        positions.sort(key=lambda x: x[0])
        
        if positions:
            x_coords = [pos[1][0] for pos in positions]
            y_coords = [pos[1][1] for pos in positions]
            z_coords = [pos[1][2] for pos in positions]
            
            ax4.plot(x_coords, y_coords, z_coords, 
                   color=colors[flock_idx % len(colors)], 
                   marker='o', linestyle='-', linewidth=2,
                   label=f'Flock {flock_idx+1}')
            
            # Add time step indicators
            for i, (t, pos) in enumerate(positions):
                ax4.text(pos[0], pos[1], pos[2], f'{t}', color=colors[flock_idx % len(colors)])
    
    # Add airport reference
    ax4.scatter([0], [0], [0], c='k', marker='s', s=100, label='Airport')
    
    ax4.set_xlabel('x (km)')
    ax4.set_ylabel('y (km)')
    ax4.set_zlabel('z (km)')
    ax4.set_title('Bird Flock Trajectories')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
