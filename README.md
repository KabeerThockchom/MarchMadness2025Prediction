# March Madness 2025 Prediction Model

## Project Overview
This project creates a machine learning model to predict the outcomes of the 2025 NCAA March Madness basketball tournaments (both men's and women's) for the [Kaggle March Machine Learning Mania 2025 competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/).

The competition requires predicting the probability of victory for all possible team matchups, with submissions evaluated using the Brier score (mean squared error between predicted probabilities and actual outcomes).

## Data Description
The model uses a rich set of historical NCAA basketball data including:

- Regular season game results (1985-2025 for men, 1998-2025 for women)
- Tournament results
- Team statistics (free throws, rebounds, assists, etc.)
- Team rankings from various systems (Pomeroy, Sagarin, RPI, etc.)
- Conference data
- Tournament seeding information
- Game locations

## Methodology

### 1. Feature Engineering
Our approach focuses on creating comprehensive team and matchup features:

- **Team-level features**: Season performance metrics including win percentage, scoring margins, shooting percentages, rebounding, etc.
- **Historical metrics**: Performance trends, strength of schedule
- **Conference context**: Performance relative to conference averages
- **Tournament experience**: Historical seed performance
- **Matchup-specific features**: Team comparisons and differential statistics

### 2. Model Architecture
We implemented a machine learning pipeline using:

- **Primary model**: XGBoost with GPU acceleration
- **Cross-validation**: 5-fold stratified cross-validation for robust generalization
- **Model calibration**: Probability calibration to optimize Brier score
- **Attempted ensembling**: Initially planned to combine XGBoost and LightGBM, but encountered compatibility issues with LightGBM

### 3. Training Process
The training process involved:

1. Computing team-level statistics for all seasons
2. Generating all possible matchup combinations
3. Creating differential and comparative features for each potential matchup
4. Training models on historical game outcomes
5. Calibrating probability predictions
6. Generating predictions for all 2025 matchups

### 4. Technical Optimizations
The pipeline includes several optimizations:

- **Memory efficiency**: Proper data types, garbage collection, and batch processing
- **GPU acceleration**: Utilizing NVIDIA GPU for model training
- **Checkpointing**: Saving intermediate results to allow for resuming processing
- **Error handling**: Robust failure recovery and fallback options

## Results
The final model achieved:

- Mean Brier Score (cross-validation): 0.160506
- Most confident predictions typically showed ~93.4% win probability

Key insights:
- Several teams showed consistently high win probabilities against various opponents
- Team IDs 1397, 1120, and 1196 were predicted to be particularly strong

## Usage Instructions

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support
- 32GB+ RAM recommended for processing large matchup matrices

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/march-madness-2025.git
cd march-madness-2025

# Install dependencies
pip install numpy pandas xgboost lightgbm scikit-learn
```

### Running the Pipeline
```bash
# Full pipeline from raw data to predictions
python march_madness_model.py

# For restarting from saved team statistics
python restart_script.py
```

### Data Directory Structure
```
data/
├── MTeams.csv                     # Men's team information
├── WTeams.csv                     # Women's team information
├── MRegularSeasonResults.csv      # Men's regular season games
├── WRegularSeasonResults.csv      # Women's regular season games
├── MNCAATourneyResults.csv        # Men's tournament games
├── WNCAATourneyResults.csv        # Women's tournament games
└── [additional data files]
```

## Dependencies
- numpy
- pandas
- xgboost (with GPU support)
- lightgbm
- scikit-learn
- matplotlib (for visualization, optional)

## Challenges and Lessons Learned
- Processing large matchup matrices requires careful memory management
- GPU acceleration significantly improves training speed
- Probability calibration is critical for optimizing Brier score
- Managing library dependencies and version compatibility is important

## Future Improvements
- Develop custom models for women's tournament (currently using baseline)
- Add advanced tournament simulation capabilities
- Implement more sophisticated feature engineering
- Explore deep learning approaches
- Improve GPU memory utilization for larger batch sizes

## Acknowledgments
- Kenneth Massey for providing much of the historical data
- Jeff Sonas of Sonas Consulting for support in assembling the dataset
- The Kaggle competition organizers
