# statistics_module.py
import numpy as np

class RiskStatistics:
    """Class to track and analyze simulation statistics"""
    
    def __init__(self, num_flocks, num_paths):
        self.num_flocks = num_flocks
        self.num_paths = num_paths
        
        # Risk level counts by time step, flock, path
        self.risk_levels = {}  # (time_step, flock_idx, path_idx) -> risk level
        
        # Position estimation errors
        self.position_errors = {}  # (time_step, flock_idx) -> (error_3d, error_2d, error_z)
        
        # Flock position history
        self.flock_positions = {}  # (time_step, flock_idx) -> (x, y, z)
    
    def record_risk(self, time_step, flock_idx, path_idx, risk_level):
        """Record risk level for a specific time step, flock, and flight path"""
        self.risk_levels[(time_step, flock_idx, path_idx)] = risk_level
    
    def record_position_error(self, time_step, flock_idx, error_3d, error_2d, error_z):
        """Record position estimation errors"""
        self.position_errors[(time_step, flock_idx)] = (error_3d, error_2d, error_z)
    
    def record_flock_position(self, time_step, flock_idx, x, y, z):
        """Record actual flock position for movement analysis"""
        self.flock_positions[(time_step, flock_idx)] = (x, y, z)
    
    def get_risk_counts_by_time(self, time_step):
        """Get counts of each risk level for a specific time step"""
        high = 0
        medium = 0
        low = 0
        
        for key, risk in self.risk_levels.items():
            if key[0] == time_step:
                if risk == "HIGH":
                    high += 1
                elif risk == "MEDIUM":
                    medium += 1
                else:
                    low += 1
        
        return high, medium, low
    
    def get_risk_counts_by_path(self, path_idx):
        """Get counts of each risk level for a specific flight path"""
        high = 0
        medium = 0
        low = 0
        
        for key, risk in self.risk_levels.items():
            if key[2] == path_idx:
                if risk == "HIGH":
                    high += 1
                elif risk == "MEDIUM":
                    medium += 1
                else:
                    low += 1
        
        return high, medium, low
    
    def get_risk_counts_by_flock(self, flock_idx):
        """Get counts of each risk level for a specific flock"""
        high = 0
        medium = 0
        low = 0
        
        for key, risk in self.risk_levels.items():
            if key[1] == flock_idx:
                if risk == "HIGH":
                    high += 1
                elif risk == "MEDIUM":
                    medium += 1
                else:
                    low += 1
        
        return high, medium, low
    
    def get_most_dangerous_path(self):
        """Get the flight path with the highest risk counts"""
        path_risks = {}
        for i in range(self.num_paths):
            high, medium, _ = self.get_risk_counts_by_path(i)
            path_risks[i] = (high * 10 + medium * 3)  # Weighted risk score
        
        most_dangerous = max(path_risks.items(), key=lambda x: x[1])
        return most_dangerous[0], most_dangerous[1]
    
    def get_average_position_error(self):
        """Get average position estimation errors across all time steps and flocks"""
        errors_3d = [e[0] for e in self.position_errors.values()]
        errors_2d = [e[1] for e in self.position_errors.values()]
        errors_z = [e[2] for e in self.position_errors.values()]
        
        return {
            "avg_3d": np.mean(errors_3d),
            "std_3d": np.std(errors_3d),
            "max_3d": np.max(errors_3d),
            "avg_2d": np.mean(errors_2d),
            "std_2d": np.std(errors_2d),
            "avg_z": np.mean(errors_z),
            "std_z": np.std(errors_z)
        }
    
    def get_position_error_by_time(self):
        """Get average position errors for each time step"""
        time_steps = set(key[0] for key in self.position_errors.keys())
        result = {}
        
        for t in time_steps:
            errors_3d = [e[0] for k, e in self.position_errors.items() if k[0] == t]
            result[t] = np.mean(errors_3d)
        
        return result
    
    def get_altitude_statistics(self):
        """Get statistics about flock altitudes over time"""
        altitudes = [pos[2] for pos in self.flock_positions.values()]
        
        return {
            "min": np.min(altitudes),
            "max": np.max(altitudes),
            "avg": np.mean(altitudes),
            "std": np.std(altitudes)
        }
    
    def get_movement_statistics(self):
        """Calculate movement patterns of flocks"""
        result = {}
        
        for flock_idx in range(self.num_flocks):
            positions = [(key[0], val) for key, val in self.flock_positions.items() 
                       if key[1] == flock_idx]
            positions.sort(key=lambda x: x[0])
            
            # Calculate velocities between consecutive time steps
            velocities = []
            for i in range(1, len(positions)):
                prev_t, prev_pos = positions[i-1]
                curr_t, curr_pos = positions[i]
                
                dx = curr_pos[0] - prev_pos[0]
                dy = curr_pos[1] - prev_pos[1]
                dz = curr_pos[2] - prev_pos[2]
                
                velocity = np.sqrt(dx**2 + dy**2 + dz**2) / (curr_t - prev_t)
                velocities.append(velocity)
            
            if velocities:
                result[flock_idx] = {
                    "avg_velocity": np.mean(velocities),
                    "max_velocity": np.max(velocities),
                    "min_velocity": np.min(velocities)
                }
            else:
                result[flock_idx] = {
                    "avg_velocity": 0,
                    "max_velocity": 0,
                    "min_velocity": 0
                }
        
        return result
    
    def print_risk_summary(self):
        """Print summary of risk levels across the simulation"""
        print("\nRISK ASSESSMENT SUMMARY")
        print("----------------------")
        
        # Overall risk counts
        total_high = sum(1 for risk in self.risk_levels.values() if risk == "HIGH")
        total_medium = sum(1 for risk in self.risk_levels.values() if risk == "MEDIUM")
        total_low = sum(1 for risk in self.risk_levels.values() if risk == "LOW")
        
        print(f"Total Risk Incidents:")
        print(f"  HIGH: {total_high}")
        print(f"  MEDIUM: {total_medium}")
        print(f"  LOW: {total_low}")
        
        # Risk by flock
        print("\nRisk by Flock:")
        for i in range(self.num_flocks):
            high, medium, low = self.get_risk_counts_by_flock(i)
            print(f"  Flock #{i+1}: {high} HIGH, {medium} MEDIUM, {low} LOW")
        
        # Most dangerous flight path
        dangerous_path_idx, risk_score = self.get_most_dangerous_path()
        print(f"\nMost Dangerous Flight Path: #{dangerous_path_idx+1} (Risk Score: {risk_score})")
        
        # Risk trend over time
        print("\nRisk Trend Over Time:")
        time_steps = sorted(set(key[0] for key in self.risk_levels.keys()))
        for t in time_steps:
            high, medium, low = self.get_risk_counts_by_time(t)
            print(f"  Time Step {t+1}: {high} HIGH, {medium} MEDIUM, {low} LOW")
    
    def print_estimation_accuracy(self):
        """Print statistics about position estimation accuracy"""
        print("\nPOSITION ESTIMATION ACCURACY")
        print("--------------------------")
        
        error_stats = self.get_average_position_error()
        
        print(f"Average 3D Position Error: {error_stats['avg_3d']:.2f} km (±{error_stats['std_3d']:.2f})")
        print(f"Max 3D Position Error: {error_stats['max_3d']:.2f} km")
        print(f"Average 2D Position Error: {error_stats['avg_2d']:.2f} km (±{error_stats['std_2d']:.2f})")
        print(f"Average Altitude Error: {error_stats['avg_z']:.2f} km (±{error_stats['std_z']:.2f})")
        
        # Error trend over time
        print("\nError Trend Over Time:")
        error_by_time = self.get_position_error_by_time()
        for t, error in sorted(error_by_time.items()):
            print(f"  Time Step {t+1}: {error:.2f} km")
    
    def print_movement_patterns(self):
        """Print statistics about bird movement patterns"""
        print("\nBIRD MOVEMENT PATTERNS")
        print("-------------------")
        
        # Altitude statistics
        alt_stats = self.get_altitude_statistics()
        print(f"Altitude Range: {alt_stats['min']:.2f} - {alt_stats['max']:.2f} km")
        print(f"Average Altitude: {alt_stats['avg']:.2f} km (±{alt_stats['std']:.2f})")
        
        # Movement velocity
        move_stats = self.get_movement_statistics()
        print("\nFlock Movement Velocities:")
        for flock_idx, stats in move_stats.items():
            print(f"  Flock #{flock_idx+1}: {stats['avg_velocity']:.2f} km/step (range: {stats['min_velocity']:.2f} - {stats['max_velocity']:.2f})")