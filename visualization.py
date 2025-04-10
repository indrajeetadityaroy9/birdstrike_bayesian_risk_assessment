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
