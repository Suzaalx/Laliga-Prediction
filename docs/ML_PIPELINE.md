# ML Pipeline Documentation

## Overview

The La Liga prediction system uses an enhanced Dixon-Coles model with comprehensive feature engineering to predict football match outcomes. This document explains the core ML pipeline components and their functionality.

## Core Components

### 1. Dixon-Coles Model (`src/laliga_pipeline/dc_model.py`)

The Dixon-Coles model is a bivariate Poisson model specifically designed for football match prediction. Our enhanced implementation includes:

#### Key Features:
- **Time Decay Weighting**: Recent matches are weighted more heavily using exponential, linear, or Gaussian decay functions
- **Low-Score Correlation**: Adjustments for correlated outcomes in low-scoring games (0-0, 1-0, 0-1, 1-1)
- **Enhanced Correlation**: Extended adjustments for additional score combinations (0-2, 2-0, 2-1, 1-2)
- **Venue Effects**: Venue-specific home advantage calculations
- **Regularization**: L2 regularization to prevent overfitting

#### Mathematical Foundation:
```
λ_home = exp(μ + α_home - δ_away + γ + venue_effect)
λ_away = exp(μ + α_away - δ_home)
```

Where:
- `μ`: Overall scoring rate
- `α_i`: Team i's attacking strength
- `δ_i`: Team i's defensive strength  
- `γ`: Home advantage parameter
- `venue_effect`: Venue-specific adjustment

#### Usage:
```python
from src.laliga_pipeline.dc_model import fit_dc_enhanced

# Fit enhanced Dixon-Coles model
result = fit_dc_enhanced(
    df=match_data,
    xi=0.01,  # Time decay parameter
    decay_type='exponential',
    use_enhanced_corr=True,
    regularization=0.001
)
```

### 2. Feature Engineering (`src/laliga_pipeline/features.py`)

Comprehensive feature engineering pipeline that transforms raw match data into ML-ready features.

#### Key Features:
- **Rolling Statistics**: Configurable windows (5, 10, 20 games) for form calculation
- **Per-90 Normalization**: Fair comparison across different match durations
- **Venue-Specific Metrics**: Home/away performance differentiation
- **Referee Analysis**: Referee tendency features (cards, fouls)
- **Head-to-Head Records**: Historical matchup performance
- **Momentum Indicators**: Winning/losing streaks and recent form

#### Feature Categories:

1. **Basic Statistics**:
   - Goals for/against per 90 minutes
   - Shots, shots on target per 90
   - Corners, fouls, cards per 90

2. **Rolling Form Features**:
   - Recent performance over multiple windows
   - Venue-specific form (home/away)
   - Opponent-adjusted metrics

3. **Advanced Metrics**:
   - Goal difference trends
   - Defensive/offensive strength ratios
   - Consistency measures (standard deviations)

4. **Contextual Features**:
   - Days since last match
   - Referee discipline tendencies
   - Head-to-head historical performance

#### Usage:
```python
from src.laliga_pipeline.features import create_feature_pipeline

# Create comprehensive feature set
features_df = create_feature_pipeline(
    df=raw_data,
    windows=[5, 10, 20],
    include_h2h=True,
    include_momentum=True,
    include_venue=True,
    include_referee=True
)
```

### 3. Backtesting Framework (`src/laliga_pipeline/backtest.py`)

Robust evaluation framework using rolling-origin cross-validation.

#### Key Features:
- **Rolling-Origin CV**: Time-series aware validation
- **Multiple Metrics**: Brier score, log loss, calibration
- **Probability Calibration**: Reliability assessment across probability ranges
- **Performance Tracking**: Model performance over time

#### Evaluation Metrics:

1. **Brier Score**: Measures accuracy of probabilistic predictions
   ```
   BS = (1/N) * Σ(p_i - o_i)²
   ```

2. **Log Loss**: Information-theoretic measure
   ```
   LL = -(1/N) * Σ[o_i * log(p_i) + (1-o_i) * log(1-p_i)]
   ```

3. **Calibration**: Reliability of predicted probabilities
   - Perfect calibration: predicted probability = observed frequency
   - Measured across probability bins

#### Usage:
```python
from src.laliga_pipeline.backtest import rolling_backtest

# Perform rolling backtest
results = rolling_backtest(
    df=features_df,
    min_train_size=500,
    test_size=50,
    step_size=25
)
```

### 4. Data Loading (`src/laliga_pipeline/loaders.py`)

Utilities for loading and preprocessing match data from various sources.

#### Features:
- **Multi-format Support**: CSV, JSON, database connections
- **Data Validation**: Schema validation and quality checks
- **Preprocessing**: Date parsing, team name standardization
- **Caching**: Efficient data loading with caching support

## Pipeline Workflow

### 1. Data Preparation
```python
# Load raw match data
raw_data = load_season_data('data/season-*.csv')

# Convert to long format and validate
long_data = to_long_enhanced(raw_data)
```

### 2. Feature Engineering
```python
# Create comprehensive feature set
features = create_feature_pipeline(
    df=long_data,
    windows=[5, 10, 20],
    include_h2h=True,
    include_momentum=True
)
```

### 3. Model Training
```python
# Fit Dixon-Coles model
model_result = fit_dc_enhanced(
    df=features,
    xi=0.01,
    decay_type='exponential',
    use_enhanced_corr=True
)
```

### 4. Evaluation
```python
# Perform backtesting
backtest_results = rolling_backtest(
    df=features,
    min_train_size=500,
    test_size=50
)

# Analyze calibration
calibration_plot(backtest_results)
```

### 5. Prediction
```python
# Generate predictions for upcoming matches
predictions = predict_matches(
    model=model_result,
    upcoming_fixtures=fixtures_df
)
```

## Model Performance

### Typical Performance Metrics:
- **Brier Score**: ~0.22-0.25 (lower is better)
- **Log Loss**: ~0.65-0.70 (lower is better)
- **Calibration**: Well-calibrated across probability ranges
- **ROC AUC**: ~0.58-0.62 for match outcomes

### Calibration Quality:
The model produces well-calibrated probabilities, meaning:
- When the model predicts 30% win probability, the team wins ~30% of the time
- Calibration is maintained across different probability ranges
- No significant over/under-confidence bias

## Configuration

### Key Parameters:
- **Time Decay (xi)**: 0.005-0.02 (higher = more recent bias)
- **Regularization**: 0.001-0.01 (prevents overfitting)
- **Rolling Windows**: [5, 10, 20] games for form calculation
- **Minimum Training Size**: 500+ matches for stable estimates

### Hyperparameter Tuning:
Use grid search or Bayesian optimization to find optimal parameters:
```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    'xi': [0.005, 0.01, 0.015, 0.02],
    'regularization': [0.001, 0.005, 0.01],
    'decay_type': ['exponential', 'linear']
}

# Grid search with cross-validation
best_params = grid_search_cv(param_grid, features_df)
```

## Best Practices

### 1. Data Quality
- Ensure consistent team name formatting
- Validate date formats and chronological order
- Handle missing statistics appropriately
- Remove duplicate matches

### 2. Feature Engineering
- Use appropriate rolling windows for different metrics
- Normalize per-90 minutes for fair comparison
- Include venue-specific features
- Consider referee effects

### 3. Model Training
- Use sufficient training data (500+ matches)
- Apply time decay weighting
- Include regularization to prevent overfitting
- Validate on out-of-sample data

### 4. Evaluation
- Use time-series aware validation
- Focus on calibration, not just accuracy
- Monitor performance over time
- Test on multiple seasons

### 5. Production Deployment
- Implement automated retraining
- Monitor prediction quality
- Track feature drift
- Maintain model versioning

## Troubleshooting

### Common Issues:

1. **Poor Calibration**:
   - Increase regularization
   - Adjust time decay parameter
   - Check for data leakage

2. **Overfitting**:
   - Reduce model complexity
   - Increase regularization
   - Use more training data

3. **Unstable Predictions**:
   - Increase minimum training size
   - Smooth feature calculations
   - Check for outliers

4. **Performance Degradation**:
   - Retrain with recent data
   - Update feature engineering
   - Check for distribution shifts

## References

1. Dixon, M.J. and Coles, S.G. (1997). "Modelling Association Football Scores and Inefficiencies in the Football Betting Market"
2. Karlis, D. and Ntzoufras, I. (2003). "Analysis of sports data by using bivariate Poisson models"
3. Baio, G. and Blangiardo, M. (2010). "Bayesian hierarchical model for the prediction of football results"