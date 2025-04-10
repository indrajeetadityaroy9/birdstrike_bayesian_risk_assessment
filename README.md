# Bird Strike Risk Assessment System

An advanced system for predicting and assessing bird strike risks to aircraft using Bayesian estimation and historical FAA wildlife strike data.

## Overview

This project implements a comprehensive bird strike risk assessment system for aviation safety. It combines:

1. **Bayesian position estimation** for real-time bird flock tracking using radar sensors
2. **Historical FAA wildlife strike data** for species-specific risk profiling
3. **Spatio-temporal analysis** of strike patterns and environmental factors

The system provides accurate bird position estimates and risk assessments for different flight paths, incorporating factors such as bird species, time of day, and season.

## System Architecture

The system consists of several interconnected modules:

```
                  ┌────────────────────┐
                  │  Data Sources      │
                  │  - Radar sensors   │
                  │  - FAA strike data │
                  │  - Weather data    │
                  └─────────┬──────────┘
                            │
                            ▼
┌──────────────────┐    ┌───────────────────┐    ┌──────────────────┐
│ Bayesian Module  │    │ Integration Layer │    │  FAA Analysis    │
│ - MAP estimation │◄──►│ - Risk prediction │◄──►│  - Species risk  │
│ - Position track │    │ - Visualization   │    │  - Temporal risk │
└──────────────────┘    └───────────────────┘    └──────────────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │  Outputs           │
                  │  - Risk assessment │
                  │  - Visualizations  │
                  │  - Statistics      │
                  └────────────────────┘
```

## Files and Components

### Core Files

- **main.py**: Main execution script that orchestrates the entire system
- **utility.py**: Core mathematical functions for Bayesian calculations
- **bird_strike_system.py**: FAA wildlife strike data analysis and integration
- **bird_behavior.py**: Bird movement and behavior modeling functions
- **visualization.py**: Plotting and visualization functions
- **statistics_module.py**: Tracking and analyzing simulation statistics

### Data Files

- **wildlife_strikes.csv**: FAA wildlife strike database (1990-2015)

## Installation and Setup

### Dependencies

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn
```

### Basic Setup

1. Clone the repository
2. Download the [FAA Wildlife Strike Database](https://www.kaggle.com/datasets/faa/wildlife-strikes) from Kaggle
3. Place the wildlife_strikes.csv file in the project directory
4. Run the main script:

```bash
python main.py
```

## File Descriptions

### main.py

The main execution script that coordinates the system components:

- Initializes the FAA risk assessment system
- Sets up the airport and runway configuration
- Places radar sensors around the airport
- Simulates bird flocks with realistic movement patterns
- Implements the Bayesian tracking and risk assessment loop
- Generates visualizations and statistical reports

### utility.py

Contains core mathematical functions:

- `calculate_distance_3d`: 3D Euclidean distance calculation
- `compute_map_objective`: Computes the MAP objective function for position estimation
- `evaluate_map_objective_grid`: Evaluates the MAP function across a grid
- `calculate_distance_to_path`: Calculates shortest distance from bird position to flight path

### bird_strike_system.py

FAA wildlife strike data analysis and integration:

- `BirdStrikeRiskSystem`: Main class for FAA data integration
- `load_faa_data`: Loads and preprocesses the FAA wildlife strike dataset
- `analyze_species_risk`: Analyzes risk profiles of different bird species
- `analyze_temporal_patterns`: Extracts temporal patterns in bird strikes
- `analyze_spatial_patterns`: Analyzes spatial patterns in bird strikes
- `calculate_risk`: Enhanced risk calculation using historical data
- `_get_species_risk_factor`: Helper function for species risk assessment
- `_get_seasonal_risk_factor`: Helper function for seasonal risk assessment

### bird_behavior.py

Models bird movement and behavior:

- `get_bird_priors`: Generates prior distribution parameters for bird positions
- `simulate_bird_movement`: Simulates realistic bird movement patterns

### visualization.py

Plotting and visualization functions:

- `plot_risk_assessment`: Creates visualizations of risk assessment results
- `plot_statistics`: Generates statistical plots for analysis

### statistics_module.py

Tracking and analyzing simulation statistics:

- `RiskStatistics`: Class for tracking bird strike risk statistics
- `record_risk`: Records risk levels for specific flight paths
- `record_position_error`: Tracks position estimation errors
- `print_risk_summary`: Generates risk assessment summaries
- `print_estimation_accuracy`: Reports on estimation accuracy

## How It Works

1. **Sensor Setup**: Radar sensors are positioned around the airport to detect bird movements.

2. **Bird Detection**: The system simulates multiple bird flocks with realistic movement patterns.

3. **Bayesian Position Estimation**:
   - Radar sensors provide noisy distance measurements to each bird flock
   - The system applies Maximum A Posteriori (MAP) estimation to determine bird positions
   - A grid-based approach visualizes the probability distribution of bird locations

4. **FAA Data Integration**:
   - Historical FAA wildlife strike data is analyzed to create species-specific risk profiles
   - Temporal patterns (seasonal, monthly) are extracted from historical data
   - Spatial patterns identify high-risk airports and regions

5. **Risk Assessment**:
   - For each detected bird flock, risk is calculated for different flight paths
   - Risk factors include distance to flight path, bird species, time of day, and season
   - The system categorizes risk as HIGH, MEDIUM, or LOW

6. **Visualization and Reporting**:
   - 2D and 3D visualizations show bird positions and flight paths
   - Risk statistics are generated for analysis
   - Species-specific risk summaries provide context from historical data

## Output Interpretation

### Risk Assessment

The system produces risk assessments for each flight path at each time step:
- **HIGH**: Bird flock is in immediate proximity to flight path
- **MEDIUM**: Bird flock is close to flight path
- **LOW**: Bird flock is at a safe distance from flight path

Risk is modified by species-specific factors from FAA data (e.g., geese pose higher risks than sparrows).

### Statistics

The system generates several statistical outputs:
- **Risk summary**: Counts of HIGH/MEDIUM/LOW risk incidents
- **Position accuracy**: Error metrics for the Bayesian estimation
- **Species-specific risk**: Historical strike data and damage rates by species

## Example Output

```
==================================================
FINAL SIMULATION STATISTICS WITH FAA CONTEXT
==================================================

RISK ASSESSMENT SUMMARY
----------------------
Total Risk Incidents:
  HIGH: 3
  MEDIUM: 57
  LOW: 0

Risk by Flock:
  Flock #1: 0 HIGH, 20 MEDIUM, 0 LOW
  Flock #2: 3 HIGH, 17 MEDIUM, 0 LOW
  Flock #3: 0 HIGH, 20 MEDIUM, 0 LOW

Most Dangerous Flight Path: #4 (Risk Score: 66)

POSITION ESTIMATION ACCURACY
--------------------------
Average 3D Position Error: 0.15 km (±0.05)
Max 3D Position Error: 0.23 km
Average 2D Position Error: 0.14 km (±0.04)
Average Altitude Error: 0.04 km (±0.03)

Species-specific risk for Flock #2 (Goose):
Species: Goose
  Historical Strikes: 2290
  Damage Rate: 52.2%
  Risk Score: 0.6053
```

## Future Enhancements

1. **Live Radar Integration**: Connect to actual radar systems rather than simulated data
2. **Weather Data Integration**: Incorporate real-time weather data to improve bird behavior modeling
3. **Machine Learning**: Add predictive models for anticipating bird movements
4. **Web Interface**: Create a user-friendly dashboard for airport operations

## Credits

This system integrates FAA Wildlife Strike Database (1990-2015) with Bayesian estimation techniques for bird position tracking and risk assessment.

## License

[MIT License](LICENSE)
