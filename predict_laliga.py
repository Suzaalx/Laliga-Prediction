#!/usr/bin/env python3
"""
La Liga Match Predictions Script

This script loads a trained model and generates predictions for upcoming La Liga matches.
Based on the reference implementation from make_predictions.py.

Usage:
    python predict_laliga.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Tuple


def load_model(model_path: str):
    """
    Load the trained model from pickle file.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        model: Loaded machine learning model
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from: {model_path}")
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {e}")


def load_upcoming_fixtures(data_dir: str) -> pd.DataFrame:
    """
    Load upcoming fixtures data.
    
    Args:
        data_dir (str): Directory containing data files
        
    Returns:
        pd.DataFrame: Upcoming fixtures
    """
    fixtures_path = os.path.join(data_dir, "upcoming-fixtures-2526.csv")
    
    if not os.path.exists(fixtures_path):
        raise FileNotFoundError(f"Upcoming fixtures file not found: {fixtures_path}")
    
    fixtures = pd.read_csv(fixtures_path)
    print(f"Loaded {len(fixtures)} upcoming fixtures")
    
    return fixtures


def load_historical_data(data_dir: str) -> pd.DataFrame:
    """
    Load historical data for team statistics calculation.
    
    Args:
        data_dir (str): Directory containing data files
        
    Returns:
        pd.DataFrame: Historical match data
    """
    print("Loading historical data for team statistics...")
    
    all_data = []
    
    # Load recent seasons for team form calculation
    recent_seasons = ['season-2425.csv', 'season-2324.csv', 'season-2122.csv']
    
    for filename in recent_seasons:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                if not df.empty:
                    all_data.append(df)
                    print(f"Loaded {filename}: {len(df)} matches")
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
    
    if not all_data:
        raise ValueError("No historical data found")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Clean data
    combined_data = combined_data.dropna(subset=['FTHG', 'FTAG'])
    
    print(f"Total historical matches: {len(combined_data)}")
    return combined_data


def calculate_team_stats(historical_data: pd.DataFrame, team: str, is_home: bool = True) -> Dict[str, float]:
    """
    Calculate team statistics from historical data.
    
    Args:
        historical_data (pd.DataFrame): Historical match data
        team (str): Team name
        is_home (bool): Whether calculating home or away stats
        
    Returns:
        Dict[str, float]: Team statistics
    """
    if is_home:
        team_matches = historical_data[historical_data['HomeTeam'] == team]
        goals_scored = team_matches['FTHG']
        goals_conceded = team_matches['FTAG']
    else:
        team_matches = historical_data[historical_data['AwayTeam'] == team]
        goals_scored = team_matches['FTAG']
        goals_conceded = team_matches['FTHG']
    
    if len(team_matches) == 0:
        return {
            'avg_goals_scored': 1.5,
            'avg_goals_conceded': 1.5,
            'over_2_5_rate': 0.5,
            'matches_played': 0
        }
    
    # Calculate recent form (last 5 matches)
    recent_matches = team_matches.tail(5)
    if is_home:
        recent_goals_scored = recent_matches['FTHG']
        recent_goals_conceded = recent_matches['FTAG']
    else:
        recent_goals_scored = recent_matches['FTAG']
        recent_goals_conceded = recent_matches['FTHG']
    
    total_goals = goals_scored + goals_conceded
    over_2_5 = (total_goals > 2.5).sum()
    
    return {
        'avg_goals_scored': goals_scored.mean(),
        'avg_goals_conceded': goals_conceded.mean(),
        'over_2_5_rate': over_2_5 / len(team_matches) if len(team_matches) > 0 else 0.5,
        'recent_goals_scored': recent_goals_scored.mean() if len(recent_matches) > 0 else goals_scored.mean(),
        'recent_goals_conceded': recent_goals_conceded.mean() if len(recent_matches) > 0 else goals_conceded.mean(),
        'matches_played': len(team_matches)
    }


def prepare_prediction_features(home_team: str, away_team: str, historical_data: pd.DataFrame, n_features: int = 17) -> np.ndarray:
    """
    Prepare features for prediction based on team statistics.
    
    Args:
        home_team (str): Home team name
        away_team (str): Away team name
        historical_data (pd.DataFrame): Historical data
        n_features (int): Number of features expected by the model
        
    Returns:
        np.ndarray: Feature vector for prediction
    """
    home_stats = calculate_team_stats(historical_data, home_team, is_home=True)
    away_stats = calculate_team_stats(historical_data, away_team, is_home=False)
    
    # Create feature vector based on common patterns in football data
    features = [
        home_stats['avg_goals_scored'],
        home_stats['avg_goals_conceded'],
        away_stats['avg_goals_scored'],
        away_stats['avg_goals_conceded'],
        home_stats['over_2_5_rate'],
        away_stats['over_2_5_rate'],
        home_stats['recent_goals_scored'],
        home_stats['recent_goals_conceded'],
        away_stats['recent_goals_scored'],
        away_stats['recent_goals_conceded'],
        # Combined metrics
        (home_stats['avg_goals_scored'] + away_stats['avg_goals_scored']) / 2,
        (home_stats['avg_goals_conceded'] + away_stats['avg_goals_conceded']) / 2,
        (home_stats['over_2_5_rate'] + away_stats['over_2_5_rate']) / 2,
        # Additional features to match training data
        home_stats['matches_played'],
        away_stats['matches_played'],
        abs(home_stats['avg_goals_scored'] - away_stats['avg_goals_scored']),
        abs(home_stats['avg_goals_conceded'] - away_stats['avg_goals_conceded'])
    ]
    
    # Ensure we have exactly the right number of features
    if len(features) > n_features:
        features = features[:n_features]
    elif len(features) < n_features:
        # Pad with zeros if we have fewer features
        features.extend([0.0] * (n_features - len(features)))
    
    return np.array(features).reshape(1, -1)


def make_predictions(model, fixtures: pd.DataFrame, historical_data: pd.DataFrame) -> List[Dict]:
    """
    Generate predictions for all fixtures.
    
    Args:
        model: Trained machine learning model
        fixtures (pd.DataFrame): Upcoming fixtures
        historical_data (pd.DataFrame): Historical data
        
    Returns:
        List[Dict]: Predictions for each match
    """
    predictions = []
    
    for _, fixture in fixtures.iterrows():
        home_team = fixture['HomeTeam']
        away_team = fixture['AwayTeam']
        match_date = fixture.get('Date', 'TBD')
        
        try:
            # Prepare features
            features = prepare_prediction_features(home_team, away_team, historical_data)
            
            # Make prediction
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            # Get confidence
            confidence = max(probabilities)
            over_2_5_prob = probabilities[1] if len(probabilities) > 1 else 0.5
            
            prediction_result = {
                'home_team': home_team,
                'away_team': away_team,
                'date': match_date,
                'prediction': 'Over 2.5' if prediction == 1 else 'Under 2.5',
                'over_2_5_probability': over_2_5_prob,
                'confidence': confidence,
                'recommendation': 'Strong' if confidence > 0.7 else 'Moderate' if confidence > 0.6 else 'Weak'
            }
            
            predictions.append(prediction_result)
            
        except Exception as e:
            print(f"Error predicting {home_team} vs {away_team}: {e}")
            continue
    
    return predictions


def format_predictions_output(predictions: List[Dict]) -> str:
    """
    Format predictions into a readable output.
    
    Args:
        predictions (List[Dict]): List of predictions
        
    Returns:
        str: Formatted output string
    """
    output = []
    output.append("🎯 La Liga Over 2.5 Goals Predictions 🎯")
    output.append("=" * 50)
    output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"Total matches: {len(predictions)}")
    output.append("")
    
    # Sort by confidence
    sorted_predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    for pred in sorted_predictions:
        output.append(f"⚽ {pred['home_team']} vs {pred['away_team']}")
        output.append(f"   📅 Date: {pred['date']}")
        output.append(f"   🎯 Prediction: {pred['prediction']}")
        output.append(f"   📊 Over 2.5 Probability: {pred['over_2_5_probability']:.1%}")
        output.append(f"   💪 Confidence: {pred['recommendation']} ({pred['confidence']:.1%})")
        output.append("")
    
    return "\n".join(output)


def find_latest_model(model_dir: str) -> str:
    """
    Find the latest trained model file.
    
    Args:
        model_dir (str): Directory containing model files
        
    Returns:
        str: Path to the latest model file
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    model_files = [f for f in os.listdir(model_dir) if f.startswith('laliga_model_') and f.endswith('.pkl')]
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    
    # Sort by filename (which includes timestamp)
    latest_model = sorted(model_files)[-1]
    return os.path.join(model_dir, latest_model)


def main():
    """
    Main prediction pipeline.
    """
    try:
        # Configuration
        data_dir = "data"
        model_dir = "models"
        output_file = "predictions_output.txt"
        
        print("=" * 50)
        print("La Liga Match Predictions")
        print("=" * 50)
        
        # Load model
        model_path = find_latest_model(model_dir)
        model = load_model(model_path)
        
        # Load data
        fixtures = load_upcoming_fixtures(data_dir)
        historical_data = load_historical_data(data_dir)
        
        # Generate predictions
        print("\nGenerating predictions...")
        predictions = make_predictions(model, fixtures, historical_data)
        
        # Format and save output
        output_text = format_predictions_output(predictions)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        
        print(f"\nPredictions saved to: {output_file}")
        print("\n" + "=" * 50)
        print("Preview of predictions:")
        print("=" * 50)
        print(output_text[:1000] + "..." if len(output_text) > 1000 else output_text)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise


if __name__ == "__main__":
    main()