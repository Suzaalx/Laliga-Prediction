"""Smart model management with caching and incremental updates."""

import pandas as pd
from typing import Dict, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

from .model_cache import ModelCache
from .dc_model import fit_dc_enhanced, predict_match
from .loaders import load_all
from .features import to_long_enhanced

logger = logging.getLogger(__name__)

class SmartDixonColesModel:
    """Smart Dixon-Coles model with caching and incremental updates."""
    
    def __init__(self, cache_dir: str = "./model_cache", data_dir: str = "./data"):
        self.cache = ModelCache(cache_dir)
        self.data_dir = Path(data_dir)
        self.current_model = None
        self.current_params = None
        self.model_config = {
            'xi': 0.003,  # Increased for better recent match weighting
            'decay_type': 'exponential',
            'venue_effects': None,
            'use_enhanced_corr': True,
            'regularization': 0.005,  # Reduced for less aggressive regularization
            'min_matches_per_team': 10,  # Minimum matches required per team
            'form_window': 6,  # Recent form consideration window
            'home_advantage_boost': 1.15  # Enhanced home advantage factor
        }
        
    def get_or_train_model(self, force_retrain: bool = False, 
                          update_threshold_days: int = 30) -> Dict:
        """Get cached model or train new one if needed."""
        
        # Load current data
        logger.info("Loading match data...")
        matches = load_all(self.data_dir)
        logger.info(f"Loaded {len(matches)} matches from {matches['Date'].min()} to {matches['Date'].max()}")
        
        if not force_retrain:
            # Try to find compatible cached model
            model_key = self.cache.find_compatible_model(matches, self.model_config)
            
            if model_key:
                # Check if model needs updating
                if not self.cache.needs_update(model_key, matches, update_threshold_days):
                    logger.info(f"Using cached model: {model_key}")
                    cached_model = self.cache.load_model(model_key)
                    self.current_model = cached_model
                    self.current_params = cached_model['params']
                    return cached_model
                else:
                    logger.info(f"Cached model {model_key} needs updating with new data")
            else:
                logger.info("No compatible cached model found")
        
        # Train new model
        logger.info("Training new Dixon-Coles model...")
        start_time = datetime.now()
        
        params = fit_dc_enhanced(
            matches,
            xi=self.model_config['xi'],
            decay_type=self.model_config['decay_type'],
            venue_effects=self.model_config['venue_effects'],
            use_enhanced_corr=self.model_config['use_enhanced_corr'],
            regularization=self.model_config['regularization'],
            min_matches_per_team=self.model_config['min_matches_per_team'],
            form_window=self.model_config['form_window'],
            home_advantage_boost=self.model_config['home_advantage_boost']
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        # Quick performance evaluation on recent data
        performance = self._quick_evaluation(matches, params)
        performance['training_time_seconds'] = training_time
        
        # Cache the model
        model_key = self.cache.save_model(
            params, matches, self.model_config, performance
        )
        
        self.current_params = params
        self.current_model = {
            'params': params,
            'config': self.model_config,
            'performance': performance,
            'model_key': model_key
        }
        
        logger.info(f"Model cached with key: {model_key}")
        logger.info(f"Quick evaluation - Accuracy: {performance.get('accuracy', 'N/A'):.3f}, "
                   f"Log Loss: {performance.get('log_loss', 'N/A'):.3f}")
        
        return self.current_model
    
    def _quick_evaluation(self, matches: pd.DataFrame, params: Dict, 
                         test_size: int = 100) -> Dict:
        """Quick model evaluation on recent matches."""
        if len(matches) < test_size + 50:
            return {}
            
        # Use last test_size matches for evaluation
        train_data = matches.iloc[:-test_size].copy()
        test_data = matches.iloc[-test_size:].copy()
        
        correct_predictions = 0
        total_log_loss = 0
        
        for _, row in test_data.iterrows():
            try:
                # Get prediction probabilities
                home_goals, away_goals = predict_match(
                    params, row['HomeTeam'], row['AwayTeam']
                )
                
                # Convert to match outcome probabilities
                prob_home = sum(home_goals[i] * away_goals[j] 
                               for i in range(len(home_goals)) 
                               for j in range(len(away_goals)) if i > j)
                prob_draw = sum(home_goals[i] * away_goals[i] 
                               for i in range(min(len(home_goals), len(away_goals))))
                prob_away = sum(home_goals[i] * away_goals[j] 
                               for i in range(len(home_goals)) 
                               for j in range(len(away_goals)) if i < j)
                
                # Normalize probabilities
                total_prob = prob_home + prob_draw + prob_away
                if total_prob > 0:
                    prob_home /= total_prob
                    prob_draw /= total_prob
                    prob_away /= total_prob
                else:
                    prob_home = prob_draw = prob_away = 1/3
                
                # Check prediction accuracy
                predicted_outcome = max([(prob_home, 'H'), (prob_draw, 'D'), (prob_away, 'A')])[1]
                if predicted_outcome == row['FTR']:
                    correct_predictions += 1
                
                # Calculate log loss contribution
                actual_probs = {'H': prob_home, 'D': prob_draw, 'A': prob_away}
                actual_prob = actual_probs[row['FTR']]
                total_log_loss -= np.log(max(actual_prob, 1e-12))
                
            except Exception as e:
                logger.warning(f"Error in quick evaluation for match {row['HomeTeam']} vs {row['AwayTeam']}: {e}")
                continue
        
        accuracy = correct_predictions / len(test_data) if len(test_data) > 0 else 0
        avg_log_loss = total_log_loss / len(test_data) if len(test_data) > 0 else float('inf')
        
        return {
            'accuracy': accuracy,
            'log_loss': avg_log_loss,
            'test_matches': len(test_data),
            'evaluation_date': datetime.now().isoformat()
        }
    
    def predict(self, home_team: str, away_team: str) -> Dict:
        """Make prediction for a match."""
        if self.current_params is None:
            raise ValueError("No model loaded. Call get_or_train_model() first.")
        
        home_goals, away_goals = predict_match(self.current_params, home_team, away_team)
        
        # Convert to match outcome probabilities
        prob_home = sum(home_goals[i] * away_goals[j] 
                       for i in range(len(home_goals)) 
                       for j in range(len(away_goals)) if i > j)
        prob_draw = sum(home_goals[i] * away_goals[i] 
                       for i in range(min(len(home_goals), len(away_goals))))
        prob_away = sum(home_goals[i] * away_goals[j] 
                       for i in range(len(home_goals)) 
                       for j in range(len(away_goals)) if i < j)
        
        # Normalize probabilities
        total_prob = prob_home + prob_draw + prob_away
        if total_prob > 0:
            prob_home /= total_prob
            prob_draw /= total_prob
            prob_away /= total_prob
        else:
            prob_home = prob_draw = prob_away = 1/3
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'probabilities': {
                'home_win': prob_home,
                'draw': prob_draw,
                'away_win': prob_away
            },
            'predicted_outcome': max([(prob_home, 'Home Win'), (prob_draw, 'Draw'), (prob_away, 'Away Win')])[1],
            'confidence': max(prob_home, prob_draw, prob_away)
        }
    
    def get_model_info(self) -> Dict:
        """Get information about current model."""
        if self.current_model is None:
            return {'status': 'No model loaded'}
        
        return {
            'status': 'Model loaded',
            'model_key': self.current_model.get('model_key', 'Unknown'),
            'config': self.current_model.get('config', {}),
            'performance': self.current_model.get('performance', {}),
            'cached_models': len(self.cache.list_models())
        }
    
    def list_cached_models(self) -> list:
        """List all cached models."""
        return self.cache.list_models()
    
    def cleanup_cache(self, keep_latest: int = 5):
        """Clean up old cached models."""
        self.cache.cleanup_old_models(keep_latest)
        logger.info(f"Cache cleanup completed, kept {keep_latest} latest models")