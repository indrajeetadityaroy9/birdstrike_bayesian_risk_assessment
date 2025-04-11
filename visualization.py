# visualization.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import time

def plot_risk_assessment(grid_x, grid_y, objective_values, true_position, estimated_position,
                        sensors_x, sensors_y, sensors_z, runways, flight_paths, risk_levels,
                        time_step, flock_idx, species_type=None, save_path=None):
    """
    Enhanced visualization of bird strike risk assessment with improved clarity
    """
    fig = plt.figure(figsize=(20, 10))
    plt.subplots_adjust(wspace=0.3)
    
    # Create custom risk colormap: green -> yellow -> orange -> red
    risk_colors = [(0.0, (0.0, 0.6, 0.0)),  # Dark green
                   (0.3, (0.0, 0.8, 0.0)),  # Green
                   (0.5, (1.0, 1.0, 0.0)),  # Yellow
                   (0.7, (1.0, 0.7, 0.0)),  # Orange
                   (0.9, (0.9, 0.0, 0.0)),  # Bright red
                   (1.0, (0.6, 0.0, 0.0))]  # Dark red
    risk_cmap = LinearSegmentedColormap.from_list("risk_colormap", risk_colors)
    
    # ======== 2D TOP-DOWN VIEW (MAP OBJECTIVE CONTOUR) ========
    ax1 = fig.add_subplot(1, 2, 1)
    
    # Normalize objective values for better visualization
    obj_min = np.min(objective_values)
    obj_max = np.max(objective_values)
    norm_obj = (objective_values - obj_min) / (obj_max - obj_min + 1e-10)
    
    # Plot MAP objective function as contour with fewer levels for clarity
    contour_levels = 15  # Reduced number of levels
    contour = ax1.contourf(grid_x, grid_y, norm_obj, levels=contour_levels, 
                         cmap='Blues_r', alpha=0.7)
    cb = fig.colorbar(contour, ax=ax1)
    cb.set_label('Bird Position Probability (Higher = More Likely)', fontsize=10)
    
    # Highlight area of highest probability
    high_prob_threshold = 0.8
    high_prob_areas = norm_obj > high_prob_threshold
    
    if np.any(high_prob_areas):
        ax1.contour(grid_x, grid_y, norm_obj, levels=[high_prob_threshold], 
                   colors=['white'], linestyles='dashed', linewidths=2, 
                   alpha=0.8)
    
    # Plot airport and runways with higher visibility
    ax1.plot(0, 0, 'ko', markersize=12, markeredgecolor='white', label='Airport Reference')
    
    for runway in runways:
        ax1.plot([runway["start"][0], runway["end"][0]], 
                [runway["start"][1], runway["end"][1]], 
                'k-', linewidth=6, solid_capstyle='round')
    
    # Plot flight paths with distinct colors and styles based on risk level
    for i, path in enumerate(flight_paths):
        # Find risk level for this path
        path_risk = "LOW"
        for path_name, risk in risk_levels:
            if path["name"] == path_name:
                path_risk = risk
                break
                
        # Color and style based on risk and path type
        if "Approach" in path["name"]:
            linestyle = '--'
            marker = '>'
        else:  # Departure
            linestyle = '-'
            marker = '<'
            
        if path_risk == "HIGH":
            color = 'red'
            linewidth = 3.5
            zorder = 10
        elif path_risk == "MEDIUM":
            color = 'orange'
            linewidth = 3
            zorder = 9
        else:
            color = 'green'
            linewidth = 2.5
            zorder = 8
            
        # Plot path with risk-based styling
        ax1.plot([path["start"][0], path["end"][0]], 
                [path["start"][1], path["end"][1]], 
                color=color, linestyle=linestyle, linewidth=linewidth, 
                marker=marker, markersize=8, zorder=zorder,
                label=f"{path['name']} ({path_risk})" if i < len(flight_paths) else "")
                
        # Show path width with subtle background
        width = path.get("width", 0.3)
        dx = path["end"][0] - path["start"][0]
        dy = path["end"][1] - path["start"][1]
        path_length = np.sqrt(dx**2 + dy**2)
        
        if path_length > 0:
            # Unit perpendicular vector
            perpx, perpy = -dy/path_length, dx/path_length
            
            # Create width envelope points
            width_points_x = []
            width_points_y = []
            
            for t in np.linspace(0, 1, 10):
                # Point along center line
                cx = path["start"][0] + t * dx
                cy = path["start"][1] + t * dy
                
                # Add width in perpendicular direction
                width_points_x.extend([cx + perpx * width/2, cx - perpx * width/2])
                width_points_y.extend([cy + perpy * width/2, cy - perpy * width/2])
            
            ax1.fill(width_points_x, width_points_y, color=color, alpha=0.1, zorder=zorder-1)
    
    # Plot sensors with increased visibility
    ax1.scatter(sensors_x, sensors_y, marker='^', s=100, color='cyan', 
               edgecolor='black', zorder=11, label='Radar Sensors')
    
    # Plot bird positions with emphasis
    ax1.scatter(true_position[0], true_position[1], marker='o', s=150, color='red', 
               edgecolor='black', zorder=12, 
               label=f'True Flock Position (alt: {true_position[2]:.2f} km)')
    ax1.scatter(estimated_position[0], estimated_position[1], marker='x', s=150, 
               color='black', edgecolor='black', linewidth=2, zorder=13,
               label=f'Estimated Position (alt: {estimated_position[2]:.2f} km)')
    
    # Connect true and estimated positions with a line
    ax1.plot([true_position[0], estimated_position[0]],
            [true_position[1], estimated_position[1]],
            'k--', alpha=0.5, zorder=11)
    
    # Error distance annotation
    error_2d = np.sqrt((true_position[0] - estimated_position[0])**2 + 
                      (true_position[1] - estimated_position[1])**2)
    error_3d = np.sqrt(error_2d**2 + (true_position[2] - estimated_position[2])**2)
    
    mid_x = (true_position[0] + estimated_position[0]) / 2
    mid_y = (true_position[1] + estimated_position[1]) / 2
    ax1.annotate(f'Error: {error_3d:.2f} km',
                xy=(mid_x, mid_y), xytext=(mid_x + 0.3, mid_y + 0.3),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='black'),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                zorder=14)
    
    # Add risk assessment summary box with improved styling
    risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for _, risk in risk_levels:
        risk_counts[risk] += 1
        
    risk_text = f"RISK ASSESSMENT SUMMARY:\n"
    risk_text += f"Species: {species_type if species_type else 'Unknown'}\n"
    risk_text += f"HIGH Risk Paths: {risk_counts['HIGH']}\n"
    risk_text += f"MEDIUM Risk Paths: {risk_counts['MEDIUM']}\n"
    risk_text += f"LOW Risk Paths: {risk_counts['LOW']}\n"
    
    # Detailed risk by path
    risk_text += "\nPath Assessments:\n"
    for path_name, risk in risk_levels:
        path_name_short = path_name.replace("Runway", "RW")
        color = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'}[risk]
        risk_text += f"• {path_name_short}: {risk}\n"
    
    # Position the risk text box in the bottom left with improved styling
    ax1.text(0.02, 0.02, risk_text, transform=ax1.transAxes, 
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                    alpha=0.9, edgecolor='gray'))
    
    ax1.set_xlabel('x (km)')
    ax1.set_ylabel('y (km)')
    title = f'Bird Strike Risk Assessment - Time Step {time_step+1}, Flock {flock_idx+1}'
    if species_type:
        title += f' ({species_type})'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Create legend with categories for better organization
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Manually create legend with categories
    legend_elements = []
    
    # Add reference points
    ref_elements = [
        by_label['Airport Reference'] if 'Airport Reference' in by_label else None,
        by_label['Radar Sensors'] if 'Radar Sensors' in by_label else None,
        by_label['True Flock Position (alt: {:.2f} km)'.format(true_position[2])] if 'True Flock Position (alt: {:.2f} km)'.format(true_position[2]) in by_label else None,
        by_label['Estimated Position (alt: {:.2f} km)'.format(estimated_position[2])] if 'Estimated Position (alt: {:.2f} km)'.format(estimated_position[2]) in by_label else None
    ]
    
    # Filter out None values
    ref_elements = [elem for elem in ref_elements if elem is not None]
    
    if ref_elements:
        legend_elements.append(('Reference Points', ref_elements))
    
    # Flight paths by risk level
    high_risk_paths = []
    medium_risk_paths = []
    low_risk_paths = []
    
    for path in flight_paths:
        path_risk = "LOW"
        for path_name, risk in risk_levels:
            if path["name"] == path_name:
                path_risk = risk
                break
                
        # Create custom legend entries
        if path_risk == "HIGH":
            color = 'red'
            if "Approach" in path["name"]:
                high_risk_paths.append(Line2D([0], [0], color=color, linestyle='--', 
                                           marker='>', markersize=8, lw=3))
            else:
                high_risk_paths.append(Line2D([0], [0], color=color, linestyle='-', 
                                           marker='<', markersize=8, lw=3))
        elif path_risk == "MEDIUM":
            color = 'orange'
            if "Approach" in path["name"]:
                medium_risk_paths.append(Line2D([0], [0], color=color, linestyle='--', 
                                              marker='>', markersize=8, lw=2.5))
            else:
                medium_risk_paths.append(Line2D([0], [0], color=color, linestyle='-', 
                                              marker='<', markersize=8, lw=2.5))
        else:
            color = 'green'
            if "Approach" in path["name"]:
                low_risk_paths.append(Line2D([0], [0], color=color, linestyle='--', 
                                          marker='>', markersize=8, lw=2))
            else:
                low_risk_paths.append(Line2D([0], [0], color=color, linestyle='-', 
                                          marker='<', markersize=8, lw=2))
    
    if high_risk_paths:
        legend_elements.append(('HIGH Risk Paths', high_risk_paths))
    if medium_risk_paths:
        legend_elements.append(('MEDIUM Risk Paths', medium_risk_paths))
    if low_risk_paths:
        legend_elements.append(('LOW Risk Paths', low_risk_paths))
    
    # Create grouped legend
    if legend_elements:
        for title, elements in legend_elements:
            ax1.legend(elements, [f"{title}"] + [""] * (len(elements) - 1), 
                     loc='upper right', fontsize=9, framealpha=0.9)
    
    # ======== 3D VIEW ========
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Set up a cleaner background
    ax2.w_xaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
    ax2.w_yaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
    ax2.w_zaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
    
    # Add horizontal plane at ground level for better context
    x_range = ax2.get_xlim()
    y_range = ax2.get_ylim()
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], 2), 
                         np.linspace(y_range[0], y_range[1], 2))
    zz = np.zeros_like(xx)
    ax2.plot_surface(xx, yy, zz, color='gray', alpha=0.1)
    
    # Plot airport and sensors with improved visibility
    ax2.scatter(0, 0, 0, c='k', marker='o', s=150, label='Airport Reference')
    ax2.scatter(sensors_x, sensors_y, sensors_z, c='cyan', marker='^', s=80, 
               edgecolor='black', label='Radar Sensors')
    
    # Plot runways
    for runway in runways:
        ax2.plot([runway["start"][0], runway["end"][0]], 
                [runway["start"][1], runway["end"][1]], 
                [runway["start"][2], runway["end"][2]], 
                'k-', linewidth=5, solid_capstyle='round')
    
    # Plot flight paths with risk zones
    for path in flight_paths:
        # Find risk level for this path
        path_risk = "LOW"
        for path_name, risk in risk_levels:
            if path["name"] == path_name:
                path_risk = risk
                break
        
        # Color based on risk
        if path_risk == "HIGH":
            color = 'red'
            alpha = 0.5
        elif path_risk == "MEDIUM":
            color = 'orange'
            alpha = 0.4
        else:
            color = 'green'
            alpha = 0.3
        
        # Plot path centerline
        ax2.plot([path["start"][0], path["end"][0]], 
                [path["start"][1], path["end"][1]], 
                [path["start"][2], path["end"][2]], 
                color=color, linewidth=3)
        
        # Create tube representing the flight path with risk-based coloring
        width = path["width"]
        t_values = np.linspace(0, 1, 10)
        theta_values = np.linspace(0, 2*np.pi, 16)
        
        # Create mesh representing the flight path tube
        xx = []
        yy = []
        zz = []
        
        for t in t_values:
            # Position along flight path
            px = path["start"][0] + t * (path["end"][0] - path["start"][0])
            py = path["start"][1] + t * (path["end"][1] - path["start"][1])
            pz = path["start"][2] + t * (path["end"][2] - path["start"][2])
            
            # Generate circle around this point
            for theta in theta_values:
                # Direction vector along path
                vx = path["end"][0] - path["start"][0]
                vy = path["end"][1] - path["start"][1]
                vz = path["end"][2] - path["start"][2]
                
                # Normalized perpendicular vectors
                # This is a simplified approach that works well for horizontal or vertical paths
                mag = np.sqrt(vx**2 + vy**2 + vz**2)
                if mag > 0:
                    vx, vy, vz = vx/mag, vy/mag, vz/mag
                
                # Create two perpendicular vectors
                if abs(vz) < 0.9:  # Not close to vertical
                    px1, py1, pz1 = 0, 0, 1  # Up vector
                else:
                    px1, py1, pz1 = 0, 1, 0  # Forward vector
                
                # First perpendicular vector using cross product
                nx = vy * pz1 - vz * py1
                ny = vz * px1 - vx * pz1
                nz = vx * py1 - vy * px1
                
                # Normalize
                nmag = np.sqrt(nx**2 + ny**2 + nz**2)
                if nmag > 0:
                    nx, ny, nz = nx/nmag, ny/nmag, nz/nmag
                
                # Second perpendicular vector using cross product
                mx = vy * nz - vz * ny
                my = vz * nx - vx * nz
                mz = vx * ny - vy * nx
                
                # Normalize
                mmag = np.sqrt(mx**2 + my**2 + mz**2)
                if mmag > 0:
                    mx, my, mz = mx/mmag, my/mmag, mz/mmag
                
                # Point on circle
                xx.append(px + width/2 * (nx * np.cos(theta) + mx * np.sin(theta)))
                yy.append(py + width/2 * (ny * np.cos(theta) + my * np.sin(theta)))
                zz.append(pz + width/2 * (nz * np.cos(theta) + mz * np.sin(theta)))
        
        # Plot tube as scattered points
        ax2.scatter(xx, yy, zz, c=color, alpha=alpha, s=10, edgecolor='none')
        
        # Add labels to flight paths
        t_mid = 0.5
        mid_x = path["start"][0] + t_mid * (path["end"][0] - path["start"][0])
        mid_y = path["start"][1] + t_mid * (path["end"][1] - path["start"][1])
        mid_z = path["start"][2] + t_mid * (path["end"][2] - path["start"][2])
        
        # Create shortened name
        path_name_short = path["name"].replace("Runway", "RW")
        
        ax2.text(mid_x, mid_y, mid_z + 0.05, f"{path_name_short}\n({path_risk})", 
                color=color, fontweight='bold', horizontalalignment='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor=color))
    
    # Plot bird positions with emphasis
    ax2.scatter(true_position[0], true_position[1], true_position[2], 
               c='red', marker='o', s=200, label='True Flock Position',
               edgecolor='black')
    ax2.scatter(estimated_position[0], estimated_position[1], estimated_position[2], 
               c='black', marker='x', s=200, label='Estimated Position',
               edgecolor='black', linewidth=2)
    
    # Connect true and estimated positions with a line
    ax2.plot([true_position[0], estimated_position[0]],
            [true_position[1], estimated_position[1]],
            [true_position[2], estimated_position[2]],
            'k--', alpha=0.7)
    
    # Add altitude indicator lines
    for z in [0.0, 0.5, 1.0, 1.5]:
        xmin, xmax = ax2.get_xlim()
        ymin, ymax = ax2.get_ylim()
        
        # Skip if z is outside z-axis range
        if z < ax2.get_zlim()[0] or z > ax2.get_zlim()[1]:
            continue
            
        # Add subtle grid at specific heights
        ax2.plot([xmin, xmin, xmax, xmax, xmin], 
                [ymin, ymax, ymax, ymin, ymin], 
                [z, z, z, z, z], 'k-', alpha=0.1)
        
        # Add altitude labels at edges
        ax2.text(xmin, ymin, z, f"{z:.1f} km", fontsize=8, ha='left', va='bottom')
    
    # Add distance to nearest flight path
    min_dist = float('inf')
    nearest_path = None
    
    for path in flight_paths:
        # Simple distance calculation to path centerline
        start = np.array(path["start"])
        end = np.array(path["end"])
        point = np.array(true_position)
        
        # Vector from start to end
        v = end - start
        # Vector from start to point
        w = point - start
        
        # Projection coefficient
        c1 = np.dot(w, v) / np.dot(v, v)
        
        if c1 < 0:
            # Point is before start
            dist = np.linalg.norm(point - start)
        elif c1 > 1:
            # Point is after end
            dist = np.linalg.norm(point - end)
        else:
            # Point projects onto line segment
            projection = start + c1 * v
            dist = np.linalg.norm(point - projection)
        
        if dist < min_dist:
            min_dist = dist
            nearest_path = path["name"]
    
    if nearest_path:
        # Find position for annotation
        ax2.text(true_position[0], true_position[1], true_position[2] + 0.1,
                f"Distance to nearest path\n({nearest_path}):\n{min_dist:.2f} km",
                fontsize=8, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
    
    ax2.set_xlabel('x (km)')
    ax2.set_ylabel('y (km)')
    ax2.set_zlabel('z (km)')
    ax2.set_title('3D View of Bird Positions and Flight Paths', fontsize=14, fontweight='bold')
    
    ax2.set_xlim([-3, 3])
    ax2.set_ylim([-3, 3])
    ax2.set_zlim([0, 2])
    
    # Set optimized view angle
    ax2.view_init(elev=25, azim=-60)
    
    # Create a better 3D legend with clear categories
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Create custom legend entries for risk levels
    risk_legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markeredgecolor='black', markersize=10, label='HIGH Risk Areas'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
               markeredgecolor='black', markersize=10, label='MEDIUM Risk Areas'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markeredgecolor='black', markersize=10, label='LOW Risk Areas')
    ]
    
    ax2.legend(handles=list(by_label.values()) + risk_legend_elements, 
              labels=list(by_label.keys()) + [r.get_label() for r in risk_legend_elements],
              loc='upper right', fontsize=9, framealpha=0.9)
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    return fig

def plot_migration_patterns(grid_x, grid_y, migration_data, flock_species):
    """Plot historical migration patterns from GPS data"""
    plt.figure(figsize=(10, 8))
    
    # Plot airport and runways
    plt.plot(0, 0, 'ko', markersize=10, label='Airport')
    
    # Plot migration paths for relevant species
    for species in flock_species:
        if species == "Gull":
            bird_name = "Eric"  # Map to tracked bird
        elif species == "Goose":
            bird_name = "Sanne"
        else:
            bird_name = "Nico"
            
        bird_data = migration_data.data[migration_data.data['bird_name'] == bird_name]
        
        # Convert GPS to local coordinates (simplified)
        x_coords = (bird_data['longitude'] - bird_data['longitude'].iloc[0]) * 0.1
        y_coords = (bird_data['latitude'] - bird_data['latitude'].iloc[0]) * 0.1
        
        plt.plot(x_coords, y_coords, 'o-', alpha=0.5, 
                 label=f'Historical {species} Path')
    
    plt.title('Bird Migration Patterns from GPS Tracking Data')
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    plt.legend()
    plt.grid(True)
    plt.show()

# risk_mapping.py - Improved plot_2d_risk_map function
def plot_2d_risk_map(self, z_level=0.2, ax=None, show_uncertainty=True, 
                    runways=None, flight_paths=None, bird_positions=None,
                    save_path=None):
    """
    Enhanced 2D risk map visualization with improved clarity and intuitive risk patterns
    """
    # Compute risk map and uncertainty at the specified altitude
    risk_map, uncertainty_map = self.compute_risk_map(z_level)
    
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 12))
    else:
        fig = ax.figure
    
    # Create an improved colormap for risk levels with clear transitions
    risk_colors = [
        (0.0, (0.0, 0.6, 0.0)),    # Dark green (LOW)
        (0.3, (0.2, 0.8, 0.2)),    # Green (LOW)
        (0.5, (1.0, 1.0, 0.0)),    # Yellow (LOW-MEDIUM boundary)
        (0.7, (1.0, 0.6, 0.0)),    # Orange (MEDIUM)
        (0.85, (0.9, 0.2, 0.0)),   # Red-orange (MEDIUM-HIGH boundary)
        (1.0, (0.7, 0.0, 0.0))     # Dark red (HIGH)
    ]
    risk_cmap = LinearSegmentedColormap.from_list("enhanced_risk_cmap", risk_colors)
    
    # Set up coordinate system
    extent = [self.x_range[0], self.x_range[1], self.y_range[0], self.y_range[1]]
    
    # IMPORTANT: Instead of using the pre-computed regular grid risk map,
    # we'll generate a more intuitive risk representation based on flight paths
    
    # First, create a clean base grid
    x_res = (self.x_range[1] - self.x_range[0]) / 100
    y_res = (self.y_range[1] - self.y_range[0]) / 100
    x_grid = np.arange(self.x_range[0], self.x_range[1] + x_res, x_res)
    y_grid = np.arange(self.y_range[0], self.y_range[1] + y_res, y_res)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Initialize risk field with low base values
    intuitive_risk = np.ones_like(X) * 0.8  # Base risk level
    
    # Generate a more intuitive risk field based on proximity to flight paths and runways
    if flight_paths:
        for path in flight_paths:
            # Get path parameters
            start = np.array(path["start"][:2])  # Just x,y coordinates
            end = np.array(path["end"][:2])
            width = path.get("width", 0.3)
            
            # Calculate path direction vector
            path_vec = end - start
            path_length = np.linalg.norm(path_vec)
            
            if path_length > 0:
                # For each point in our grid, calculate distance to this path
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        point = np.array([X[i, j], Y[i, j]])
                        
                        # Vector from start to point
                        start_to_point = point - start
                        
                        # Calculate projection onto path
                        projection = np.dot(start_to_point, path_vec) / path_length
                        
                        # Normalize to 0-1 along path length
                        t = projection / path_length
                        
                        # If projection is on the path
                        if 0 <= t <= 1:
                            # Calculate perpendicular distance to path
                            path_point = start + t * path_vec
                            distance = np.linalg.norm(point - path_point)
                            
                            # Apply a risk function that increases with proximity to path
                            # with maximum near the centerline
                            if distance < width * 3:  # Consider area up to 3x path width
                                # Calculate risk based on distance (inverse relationship)
                                distance_factor = 1.0 - min(1.0, distance / (width * 3))
                                path_risk = 1.0 + 2.0 * distance_factor**2  # Up to risk level 3.0
                                
                                # Apply higher risk near takeoff and landing points
                                endpoint_factor = max(0, 1.0 - min(t, 1-t) * 5)  # Higher risk at endpoints
                                path_risk += endpoint_factor * 0.5
                                
                                # Update overall risk (take maximum of current and new risk)
                                intuitive_risk[i, j] = max(intuitive_risk[i, j], path_risk)
    
    # Add increased risk around runway intersections
    if runways and len(runways) > 1:
        # Find all runway intersections and mark them as high risk zones
        for i, runway1 in enumerate(runways):
            for j, runway2 in enumerate(runways[i+1:], i+1):
                # Get runway endpoints
                r1_start = np.array(runway1["start"][:2])
                r1_end = np.array(runway1["end"][:2])
                r2_start = np.array(runway2["start"][:2])
                r2_end = np.array(runway2["end"][:2])
                
                # Check for intersection
                # This is a simplified approach - in a real system you'd use proper line intersection
                if np.linalg.norm(r1_start - r2_start) < 0.1 or np.linalg.norm(r1_start - r2_end) < 0.1 or \
                   np.linalg.norm(r1_end - r2_start) < 0.1 or np.linalg.norm(r1_end - r2_end) < 0.1:
                    # Runways intersect or have common endpoint
                    intersection_point = None
                    
                    # Find approximate intersection point
                    if np.linalg.norm(r1_start - r2_start) < 0.1:
                        intersection_point = r1_start
                    elif np.linalg.norm(r1_start - r2_end) < 0.1:
                        intersection_point = r1_start
                    elif np.linalg.norm(r1_end - r2_start) < 0.1:
                        intersection_point = r1_end
                    elif np.linalg.norm(r1_end - r2_end) < 0.1:
                        intersection_point = r1_end
                    
                    if intersection_point is not None:
                        # Add high risk zone around intersection
                        for i in range(X.shape[0]):
                            for j in range(X.shape[1]):
                                point = np.array([X[i, j], Y[i, j]])
                                dist_to_intersection = np.linalg.norm(point - intersection_point)
                                
                                if dist_to_intersection < 0.5:  # 500m radius around intersection
                                    # Reduce risk with distance from intersection
                                    intersection_risk = 3.0 * (1.0 - min(1.0, dist_to_intersection / 0.5))
                                    intuitive_risk[i, j] = max(intuitive_risk[i, j], intersection_risk)
    
    # Add risk around bird positions
    if bird_positions:
        # Convert to array of positions if needed
        if isinstance(bird_positions[0], tuple):
            bird_pos_array = np.array([(pos[0], pos[1]) for pos in bird_positions])
        
            # Add risk around each bird position
            for pos in bird_pos_array:
                # Create a risk "halo" around each bird
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        point = np.array([X[i, j], Y[i, j]])
                        dist = np.linalg.norm(point - pos)
                        
                        if dist < 0.7:  # 700m radius around bird
                            # Risk decreases with distance
                            bird_risk = 2.5 * (1.0 - min(1.0, dist / 0.7))
                            intuitive_risk[i, j] = max(intuitive_risk[i, j], bird_risk)
    
    # Plot the more intuitive risk map
    risk_contour = ax.contourf(
        X, Y, intuitive_risk,
        levels=15,  # Fewer levels for clearer visualization
        cmap=risk_cmap, 
        alpha=0.8,
        extent=extent
    )
    
    # Add colorbar with clear labels
    cbar = plt.colorbar(risk_contour, ax=ax, pad=0.02)
    cbar.set_label('Risk Level', fontsize=14, fontweight='bold')
    
    # Add specific tick marks for risk thresholds with text labels
    cbar.set_ticks([1.0, 1.5, 2.0, 2.5, 3.0])
    cbar.set_ticklabels(['LOW', 'LOW/MED', 'MEDIUM', 'MED/HIGH', 'HIGH'])
    
    # Add clear risk zone boundaries with smooth contours
    # HIGH risk boundary (2.5) with distinctive styling
    high_risk = ax.contour(
        X, Y, intuitive_risk,
        levels=[2.5], 
        colors=['darkred'], 
        linewidths=3, 
        linestyles='solid',
        extent=extent
    )
    
    # MEDIUM risk boundary (1.5) with distinctive styling
    medium_risk = ax.contour(
        X, Y, intuitive_risk,
        levels=[1.5], 
        colors=['darkorange'], 
        linewidths=2.5, 
        linestyles='solid',
        extent=extent
    )
    
    # Label risk boundaries in key areas only (not on every line segment)
    fmt = {}
    for l in high_risk.levels:
        fmt[l] = 'HIGH RISK'
    for l in medium_risk.levels:
        fmt[l] = 'MEDIUM RISK'
        
    # Add manual labels instead of using clabel (which can be cluttered)
    # Find a good spot for HIGH risk label
    high_paths = high_risk.collections[0].get_paths()
    if len(high_paths) > 0:
        for path in high_paths:
            if len(path.vertices) > 10:  # Only label substantial contours
                # Get a point ~30% along the contour
                idx = int(len(path.vertices) * 0.3)
                x, y = path.vertices[idx]
                ax.text(x, y, 'HIGH RISK ZONE', color='white', fontweight='bold', fontsize=10,
                       bbox=dict(facecolor='darkred', alpha=0.9, boxstyle='round'),
                       ha='center', va='center', zorder=20)
                break
    
    # Find a good spot for MEDIUM risk label
    medium_paths = medium_risk.collections[0].get_paths()
    if len(medium_paths) > 0:
        for path in medium_paths:
            if len(path.vertices) > 10:  # Only label substantial contours
                # Get a point ~30% along the contour
                idx = int(len(path.vertices) * 0.3)
                x, y = path.vertices[idx]
                ax.text(x, y, 'MEDIUM RISK ZONE', color='black', fontweight='bold', fontsize=10,
                       bbox=dict(facecolor='gold', alpha=0.9, boxstyle='round'),
                       ha='center', va='center', zorder=20)
                break
    
    # Show uncertainty if requested with improved styling
    if show_uncertainty and uncertainty_map is not None:
        # Re-sample uncertainty to match our new grid
        uncertainty_interp = np.zeros_like(X)
        
        # Simple nearest-neighbor interpolation for uncertainty
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Find closest point in original grid
                x_idx = int((X[i, j] - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * 
                           (uncertainty_map.shape[1] - 1))
                y_idx = int((Y[i, j] - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * 
                           (uncertainty_map.shape[0] - 1))
                
                # Bound indices to valid range
                x_idx = max(0, min(uncertainty_map.shape[1] - 1, x_idx))
                y_idx = max(0, min(uncertainty_map.shape[0] - 1, y_idx))
                
                uncertainty_interp[i, j] = uncertainty_map[y_idx, x_idx]
        
        # Calculate uncertainty levels based on percentiles for more meaningful contours
        uncertainty_levels = np.percentile(uncertainty_interp.flatten(), [75, 90])
        
        # Plot only high uncertainty contours for clarity
        uncertainty_contour = ax.contour(
            X, Y, uncertainty_interp,
            levels=uncertainty_levels, 
            colors=['blue'], 
            linestyles='dashed', 
            linewidths=[1.5, 2.5], 
            alpha=0.7, 
            extent=extent
        )
        
        # Label high uncertainty areas
        high_uncertainty_paths = uncertainty_contour.collections[-1].get_paths()
        if len(high_uncertainty_paths) > 0:
            for path in high_uncertainty_paths:
                if len(path.vertices) > 10:  # Only label substantial contours
                    # Get a point ~30% along the contour
                    idx = int(len(path.vertices) * 0.3)
                    x, y = path.vertices[idx]
                    ax.text(x, y, 'High Uncertainty', color='blue', fontsize=9, 
                           fontweight='bold', style='italic',
                           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                           ha='center', va='center', zorder=19)
                    break
    
    # Plot airport reference point with improved styling
    ax.plot(0, 0, 'ko', markersize=14, markeredgecolor='white', markeredgewidth=2, 
           label='Airport Reference')
    ax.text(0.05, 0.05, 'Airport', fontsize=12, fontweight='bold', 
           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    # Plot runways with improved styling
    if runways:
        for runway_idx, runway in enumerate(runways):
            ax.plot(
                [runway["start"][0], runway["end"][0]],
                [runway["start"][1], runway["end"][1]],
                'k-', linewidth=8, solid_capstyle='round',
                label=f"Runway {runway_idx+1}" if runway_idx == 0 else ""
            )
            
            # Add runway labels
            mid_x = (runway["start"][0] + runway["end"][0]) / 2
            mid_y = (runway["start"][1] + runway["end"][1]) / 2
            
            # Add runway number text
            ax.text(mid_x, mid_y + 0.1, f"Runway {runway_idx+1}", 
                   fontsize=12, fontweight='bold', ha='center',
                   bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'))
    
    # Plot flight paths with improved styling
    if flight_paths:
        for path_idx, path in enumerate(flight_paths):
            # Determine path style based on type
            if "Approach" in path["name"]:
                linestyle = '--'
                marker = '>'
                path_type = "Approach"
                color = 'blue'
                alpha_fill = 0.15
            else:  # Departure
                linestyle = '-'
                marker = '<'
                path_type = "Departure"
                color = 'green'
                alpha_fill = 0.15
            
            # Determine path width with clearer visibility
            width = path.get("width", 0.3)
            
            # Plot the path with improved styling
            ax.plot(
                [path["start"][0], path["end"][0]],
                [path["start"][1], path["end"][1]],
                color=color, linestyle=linestyle, linewidth=4, marker=marker,
                markersize=10, markevery=[1], zorder=10,
            )
            
            # Create an improved path visualization
            start = np.array(path["start"][:2])
            end = np.array(path["end"][:2])
            path_vec = end - start
            path_length = np.linalg.norm(path_vec)
            
            if path_length > 0:
                # Unit perpendicular vector
                perp_vec = np.array([-path_vec[1], path_vec[0]]) / path_length
                
                # Create width envelope points with more detail
                width_points_x = []
                width_points_y = []
                
                for t in np.linspace(0, 1, 30):  # More points for smoother outline
                    # Point along center line
                    cx = start[0] + t * path_vec[0]
                    cy = start[1] + t * path_vec[1]
                    
                    # Add width in perpendicular direction
                    width_points_x.extend([cx + perp_vec[0] * width/2, cx - perp_vec[0] * width/2])
                    width_points_y.extend([cy + perp_vec[1] * width/2, cy - perp_vec[1] * width/2])
                
                # Fill path width with appropriate styling
                ax.fill(width_points_x, width_points_y, color=color, alpha=alpha_fill, zorder=5)
                
                # Add a border to the path width for better visibility
                ax.plot(width_points_x[:30], width_points_y[:30], color=color, linestyle=':', 
                       linewidth=1.5, alpha=0.5, zorder=6)
                ax.plot(width_points_x[30:], width_points_y[30:], color=color, linestyle=':', 
                       linewidth=1.5, alpha=0.5, zorder=6)
            
                # Sample intuitive risk along the path to determine risk level
                path_risks = []
                for t in np.linspace(0, 1, 10):
                    # Point along center line
                    cx = start[0] + t * path_vec[0]
                    cy = start[1] + t * path_vec[1]
                    
                    # Find closest point in grid
                    x_idx = np.argmin(np.abs(X[0, :] - cx))
                    y_idx = np.argmin(np.abs(Y[:, 0] - cy))
                    
                    if 0 <= y_idx < intuitive_risk.shape[0] and 0 <= x_idx < intuitive_risk.shape[1]:
                        path_risks.append(intuitive_risk[y_idx, x_idx])
                
                # Determine overall path risk level
                if path_risks:
                    avg_risk = np.mean(path_risks)
                    max_risk = np.max(path_risks)
                    
                    # Get risk label
                    if max_risk >= 2.5:
                        risk_label = "HIGH"
                        box_color = 'red'
                    elif max_risk >= 1.5:
                        risk_label = "MEDIUM"
                        box_color = 'orange'
                    else:
                        risk_label = "LOW"
                        box_color = 'green'
                    
                    # Add path label with risk information
                    label_x = (start[0] + end[0]) / 2 + perp_vec[0] * (width/2 + 0.15)
                    label_y = (start[1] + end[1]) / 2 + perp_vec[1] * (width/2 + 0.15)
                    
                    # Create path name
                    path_name_short = path["name"].replace("Runway", "RW")
                    
                    # Add enhanced label with path info and risk level
                    ax.text(label_x, label_y, f"{path_name_short}\n{risk_label} RISK",
                           color='black', fontweight='bold', fontsize=10,
                           bbox=dict(facecolor=box_color, alpha=0.7, edgecolor='black', 
                                    boxstyle='round', pad=0.5),
                           ha='center', va='center', zorder=15)
    
    # Plot bird positions with improved styling
    if bird_positions:
        # Handle both list of tuples and dictionary formats
        if isinstance(bird_positions[0], tuple):
            # Extract positions
            bird_xs = [pos[0] for pos in bird_positions]
            bird_ys = [pos[1] for pos in bird_positions]
            bird_zs = [pos[2] for pos in bird_positions]
            
            # Plot birds with larger markers and better visibility
            birds = ax.scatter(
                bird_xs, bird_ys, 
                c='red', marker='o', s=200, edgecolor='white', linewidth=2,
                label='Bird Positions', zorder=20
            )
            
            # Add bird information labels
            for i, pos in enumerate(bird_positions):
                # Calculate distance to slice plane
                z_distance = abs(pos[2] - z_level)
                
                # Sample risk at bird position
                x_idx = np.argmin(np.abs(X[0, :] - pos[0]))
                y_idx = np.argmin(np.abs(Y[:, 0] - pos[1]))
                
                if 0 <= y_idx < intuitive_risk.shape[0] and 0 <= x_idx < intuitive_risk.shape[1]:
                    bird_risk_level = intuitive_risk[y_idx, x_idx]
                    
                    # Get risk label
                    if bird_risk_level >= 2.5:
                        risk_label = "HIGH RISK"
                        box_color = 'red'
                        box_edge = 'darkred'
                    elif bird_risk_level >= 1.5:
                        risk_label = "MEDIUM RISK"
                        box_color = 'orange'
                        box_edge = 'darkorange'
                    else:
                        risk_label = "LOW RISK"
                        box_color = 'lightgreen'
                        box_edge = 'green'
                else:
                    risk_label = "UNKNOWN RISK"
                    box_color = 'lightgray'
                    box_edge = 'gray'
                
                # Adjust label based on proximity to current slice
                if z_distance < 0.05:  # Within 50m of slice plane
                    # Birds at this altitude level - highlight
                    text_color = 'black'
                    box_alpha = 0.9
                    z_info = f"AT THIS ALTITUDE\n({pos[2]:.2f} km)"
                elif z_distance < 0.2:  # Within 200m
                    # Birds close to this altitude - less emphasis
                    text_color = 'black'
                    box_alpha = 0.7
                    z_info = f"Near this altitude\n({pos[2]:.2f} km)"
                else:
                    # Birds far from this altitude - minimal emphasis
                    text_color = 'gray'
                    box_alpha = 0.5
                    z_info = f"Different altitude\n({pos[2]:.2f} km)"
                
                # Create informative label with risk assessment
                bird_info = f"Flock {i+1}\n{z_info}\n{risk_label}"
                
                # Add the enhanced bird position label
                ax.text(pos[0] + 0.15, pos[1] + 0.15, bird_info,
                       fontsize=9, color=text_color, fontweight='bold',
                       bbox=dict(facecolor=box_color, alpha=box_alpha, 
                               edgecolor=box_edge, boxstyle='round,pad=0.5'),
                       zorder=21)
    
    # Add annotations for axis references
    ax.set_xlabel('x (km)', fontsize=14, fontweight='bold')
    ax.set_ylabel('y (km)', fontsize=14, fontweight='bold')
    
    # Generate informative title 
    title = f'Bird Strike Risk Map (Altitude: {z_level:.2f} km)'
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Add grid and customize appearance
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add map context and legend in a clean info box
    context_text = (
        f"Altitude: {z_level:.2f} km\n"
        f"Grid Resolution: {x_res:.2f} km\n"
        f"Risk increases near flight paths and intersections"
    )
    
    # Add context text in a clear box
    ax.text(0.02, 0.98, context_text,
           transform=ax.transAxes, fontsize=12, fontweight='bold',
           bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'),
           va='top', ha='left', zorder=30)
    
    # Add timestamp for reference
    import time
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    ax.text(0.98, 0.02,
           f"Generated: {timestamp}\nBird Strike Risk Assessment System",
           transform=ax.transAxes, fontsize=9, color='black',
           ha='right', va='bottom', zorder=30,
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Create improved legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        # Reference elements
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
               markeredgecolor='white', markersize=10, label='Airport Reference'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markeredgecolor='white', markersize=10, label='Bird Position'),
        
        # Risk zones
        Patch(facecolor='darkred', edgecolor='white', alpha=0.7, label='HIGH Risk Zone (>2.5)'),
        Patch(facecolor='orange', edgecolor='black', alpha=0.7, label='MEDIUM Risk Zone (1.5-2.5)'),
        Patch(facecolor='green', edgecolor='black', alpha=0.7, label='LOW Risk Zone (<1.5)'),
        
        # Path types
        Line2D([0], [0], color='blue', linestyle='--', marker='>', 
              markersize=8, label='Approach Path'),
        Line2D([0], [0], color='green', linestyle='-', marker='<', 
              markersize=8, label='Departure Path')
    ]
    
    # Add uncertainty to legend if shown
    if show_uncertainty:
        legend_elements.append(
            Line2D([0], [0], color='blue', linestyle='dashed', 
                  alpha=0.7, label='Uncertainty Contour')
        )
    
    # Create enhanced legend with grouping
    ax.legend(handles=legend_elements, loc='upper right', 
             fontsize=10, framealpha=0.9, title='Legend',
             title_fontsize=12, bbox_to_anchor=(1.0, 0.98))
    
    # Set axis limits to match extent
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Enhanced risk map saved to {save_path}")
    
    # Return the axis for further customization if needed
    return ax