#!/usr/bin/env python3
"""
La Liga Prediction Model Training Script

This script trains machine learning models to predict Over 2.5 goals in La Liga matches
based on the reference implementation from train_models.py.

Usage:
    python train_laliga_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, cross_val_score, KFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_laliga_data(data_dir: str) -> pd.DataFrame:
    """
    Load and combine all La Liga historical data files.
    
    Args:
        data_dir (str): Directory containing season CSV files
        
    Returns:
        pd.DataFrame: Combined historical data
    """
    print("Loading La Liga historical data...")
    
    all_data = []
    
    # Load all season files except the current incomplete season and upcoming fixtures
    for filename in os.listdir(data_dir):
        if filename.startswith('season-') and filename.endswith('.csv'):
            # Skip incomplete current season data
            if filename in ['season-2526.csv']:
                continue
                
            filepath = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(filepath)
                if not df.empty:
                    all_data.append(df)
                    print(f"Loaded {filename}: {len(df)} matches")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    if not all_data:
        raise ValueError("No valid data files found")
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Total matches loaded: {len(combined_data)}")
    
    return combined_data


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data for model training.
    
    Args:
        df (pd.DataFrame): Raw match data
        
    Returns:
        pd.DataFrame: Preprocessed data ready for training
    """
    print("Preprocessing data...")
    
    # Create a copy to avoid modifying original
    data = df.copy()
    
    # Drop rows with missing essential data
    essential_cols = ['FTHG', 'FTAG']
    data = data.dropna(subset=essential_cols)
    
    # Create Over2.5 target variable
    data['Over2.5'] = ((data['FTHG'] + data['FTAG']) > 2.5).astype(int)
    
    # Select only numeric columns for training
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target from features
    feature_cols = [col for col in numeric_cols if col != 'Over2.5']
    
    # Keep only relevant columns
    processed_data = data[feature_cols + ['Over2.5']].copy()
    
    # Handle missing values more robustly
    print(f"Missing values before cleaning: {processed_data.isnull().sum().sum()}")
    
    # Fill NaN values with appropriate defaults
    for col in feature_cols:
        if processed_data[col].isnull().any():
            if processed_data[col].dtype in ['float64', 'int64']:
                # Use median for numeric columns, fallback to 0 if all NaN
                median_val = processed_data[col].median()
                fill_val = median_val if not pd.isna(median_val) else 0.0
                processed_data[col] = processed_data[col].fillna(fill_val)
            else:
                processed_data[col] = processed_data[col].fillna(0)
    
    # Final check and cleanup
    processed_data = processed_data.replace([np.inf, -np.inf], 0)
    
    # Drop any remaining rows with NaN values
    processed_data = processed_data.dropna()
    
    print(f"Missing values after cleaning: {processed_data.isnull().sum().sum()}")
    print(f"Features selected: {len(feature_cols)}")
    print(f"Training samples: {len(processed_data)}")
    print(f"Over 2.5 goals rate: {processed_data['Over2.5'].mean():.3f}")
    
    return processed_data


def train_models(X: np.ndarray, y: np.ndarray) -> VotingClassifier:
    """
    Train multiple models and create a voting classifier ensemble.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target variable
        
    Returns:
        VotingClassifier: Trained ensemble model
    """
    print("Training models...")
    
    # Define models and hyperparameters (simplified for faster training)
    models_config = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.1, 0.2]
            }
        }
    }
    
    # Cross-validation setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(accuracy_score)
    
    best_models = []
    
    for name, config in models_config.items():
        print(f"Training {name}...")
        
        # Hyperparameter tuning
        grid_search = HalvingGridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=cv,
            scoring=scorer,
            n_jobs=-1,
            random_state=42
        )
        
        grid_search.fit(X, y)
        
        # Evaluate best model
        cv_scores = cross_val_score(grid_search.best_estimator_, X, y, cv=cv, scoring=scorer)
        print(f"{name} - Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        best_models.append((name.lower().replace(' ', '_'), grid_search.best_estimator_))
    
    # Create voting classifier
    print("Creating ensemble model...")
    voting_clf = VotingClassifier(estimators=best_models, voting='soft')
    voting_clf.fit(X, y)
    
    # Evaluate ensemble
    cv_scores = cross_val_score(voting_clf, X, y, cv=cv, scoring=scorer)
    print(f"Ensemble - Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    return voting_clf


def save_model(model: VotingClassifier, model_dir: str) -> str:
    """
    Save the trained model to disk.
    
    Args:
        model (VotingClassifier): Trained model
        model_dir (str): Directory to save the model
        
    Returns:
        str: Path to saved model file
    """
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"laliga_model_{timestamp}.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {model_path}")
    return model_path


def main():
    """
    Main training pipeline.
    """
    try:
        # Configuration
        data_dir = "data"
        model_dir = "models"
        
        print("=" * 50)
        print("La Liga Prediction Model Training")
        print("=" * 50)
        
        # Load and preprocess data
        raw_data = load_laliga_data(data_dir)
        processed_data = preprocess_data(raw_data)
        
        # Prepare features and target
        y = processed_data['Over2.5'].values
        X = processed_data.drop('Over2.5', axis=1).values
        
        print(f"Training data shape: {X.shape}")
        print(f"Target distribution: {np.bincount(y)}")
        
        # Train models
        model = train_models(X, y)
        
        # Save model
        model_path = save_model(model, model_dir)
        
        print("\n" + "=" * 50)
        print("Training completed successfully!")
        print(f"Model saved to: {model_path}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()