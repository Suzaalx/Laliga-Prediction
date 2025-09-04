import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.stats import poisson
import warnings
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger("dixon_coles")

@dataclass
class DCParameters:
    """Dixon-Coles model parameters"""
    home_advantage: float
    attack_strengths: Dict[str, float]
    defense_strengths: Dict[str, float]
    rho: float  # Low-score correlation parameter
    mu: float   # Overall scoring rate
    time_decay: float
    log_likelihood: float
    convergence_info: Dict[str, Any]

class DixonColesModel:
    """Dixon-Coles Poisson model with time decay and low-score corrections"""
    
    def __init__(
        self,
        time_decay_factor: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        regularization: float = 0.001
    ):
        self.time_decay_factor = time_decay_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        
        self.parameters: Optional[DCParameters] = None
        self.teams: List[str] = []
        self.is_fitted = False
        
        # Low-score correction matrix (Dixon-Coles tau function)
        self.tau_matrix = {
            (0, 0): -0.1,  # 0-0
            (0, 1): 0.1,   # 0-1
            (1, 0): 0.1,   # 1-0
            (1, 1): -0.1   # 1-1
        }
    
    def fit(
        self,
        matches_df: pd.DataFrame,
        reference_date: Optional[datetime] = None
    ) -> DCParameters:
        """Fit the Dixon-Coles model to match data
        
        Args:
            matches_df: DataFrame with columns ['date', 'home_team', 'away_team', 'home_goals', 'away_goals']
            reference_date: Reference date for time decay (defaults to most recent match)
        
        Returns:
            Fitted model parameters
        """
        
        logger.info("Starting Dixon-Coles model fitting")
        
        # Validate input data
        self._validate_input_data(matches_df)
        
        # Prepare data
        df = matches_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Set reference date
        if reference_date is None:
            reference_date = df['date'].max()
        
        # Get unique teams
        self.teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))
        n_teams = len(self.teams)
        
        logger.info(f"Fitting model with {len(df)} matches and {n_teams} teams")
        
        # Calculate time weights
        df['days_ago'] = (reference_date - df['date']).dt.days
        df['weight'] = np.exp(-self.time_decay_factor * df['days_ago'])
        
        # Create team indices
        team_to_idx = {team: i for i, team in enumerate(self.teams)}
        df['home_idx'] = df['home_team'].map(team_to_idx)
        df['away_idx'] = df['away_team'].map(team_to_idx)
        
        # Initial parameter estimates
        initial_params = self._get_initial_parameters(df, n_teams)
        
        # Optimize parameters
        result = minimize(
            fun=self._negative_log_likelihood,
            x0=initial_params,
            args=(df,),
            method='L-BFGS-B',
            options={
                'maxiter': self.max_iterations,
                'ftol': self.tolerance,
                'disp': False
            }
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
        # Extract fitted parameters
        self.parameters = self._extract_parameters(result.x, n_teams)
        self.parameters.log_likelihood = -result.fun
        self.parameters.convergence_info = {
            'success': result.success,
            'message': result.message,
            'iterations': result.nit,
            'function_evaluations': result.nfev
        }
        
        self.is_fitted = True
        
        logger.info(
            f"Model fitted successfully. Log-likelihood: {self.parameters.log_likelihood:.4f}, "
            f"Home advantage: {self.parameters.home_advantage:.4f}, "
            f"Rho: {self.parameters.rho:.4f}"
        )
        
        return self.parameters
    
    def predict_match(
        self,
        home_team: str,
        away_team: str,
        max_goals: int = 10
    ) -> Dict[str, Any]:
        """Predict probabilities for a single match
        
        Args:
            home_team: Home team name
            away_team: Away team name
            max_goals: Maximum goals to consider for scoreline probabilities
        
        Returns:
            Dictionary with match probabilities and expected goals
        """
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if home_team not in self.teams or away_team not in self.teams:
            raise ValueError(f"Unknown team(s): {home_team}, {away_team}")
        
        # Calculate expected goals
        lambda_home, lambda_away = self._calculate_expected_goals(home_team, away_team)
        
        # Calculate scoreline probabilities
        scoreline_probs = self._calculate_scoreline_probabilities(
            lambda_home, lambda_away, max_goals
        )
        
        # Calculate match outcome probabilities
        home_win_prob = sum(
            scoreline_probs[(h, a)] 
            for h in range(max_goals + 1) 
            for a in range(max_goals + 1) 
            if h > a
        )
        
        draw_prob = sum(
            scoreline_probs[(h, h)] 
            for h in range(max_goals + 1)
        )
        
        away_win_prob = sum(
            scoreline_probs[(h, a)] 
            for h in range(max_goals + 1) 
            for a in range(max_goals + 1) 
            if a > h
        )
        
        # Most likely scoreline
        most_likely_score = max(scoreline_probs.items(), key=lambda x: x[1])
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'expected_goals': {
                'home': lambda_home,
                'away': lambda_away
            },
            'match_probabilities': {
                'home_win': home_win_prob,
                'draw': draw_prob,
                'away_win': away_win_prob
            },
            'most_likely_score': {
                'home_goals': most_likely_score[0][0],
                'away_goals': most_likely_score[0][1],
                'probability': most_likely_score[1]
            },
            'scoreline_probabilities': {
                f"{h}-{a}": prob 
                for (h, a), prob in scoreline_probs.items() 
                if prob > 0.01  # Only include likely scorelines
            }
        }
    
    def predict_matches(
        self,
        matches: List[Tuple[str, str]],
        max_goals: int = 10
    ) -> List[Dict[str, Any]]:
        """Predict probabilities for multiple matches"""
        
        return [self.predict_match(home, away, max_goals) for home, away in matches]
    
    def get_team_strengths(self) -> Dict[str, Dict[str, float]]:
        """Get team attack and defense strengths"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing team strengths")
        
        return {
            team: {
                'attack_strength': self.parameters.attack_strengths[team],
                'defense_strength': self.parameters.defense_strengths[team]
            }
            for team in self.teams
        }
    
    def _validate_input_data(self, df: pd.DataFrame) -> None:
        """Validate input data format"""
        
        required_columns = ['date', 'home_team', 'away_team', 'home_goals', 'away_goals']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        # Check for missing values
        if df[required_columns].isnull().any().any():
            raise ValueError("Input data contains missing values")
        
        # Check goal values are non-negative integers
        if not all(df['home_goals'] >= 0) or not all(df['away_goals'] >= 0):
            raise ValueError("Goal values must be non-negative")
    
    def _get_initial_parameters(self, df: pd.DataFrame, n_teams: int) -> np.ndarray:
        """Get initial parameter estimates"""
        
        # Overall scoring rate
        total_goals = df['home_goals'].sum() + df['away_goals'].sum()
        total_matches = len(df)
        mu_init = np.log(total_goals / (2 * total_matches))
        
        # Home advantage
        home_goals_per_match = df['home_goals'].mean()
        away_goals_per_match = df['away_goals'].mean()
        gamma_init = np.log(home_goals_per_match / away_goals_per_match) / 2
        
        # Team strengths (start with small random values)
        np.random.seed(42)  # For reproducibility
        attack_init = np.random.normal(0, 0.1, n_teams)
        defense_init = np.random.normal(0, 0.1, n_teams)
        
        # Ensure sum constraints
        attack_init = attack_init - attack_init.mean()
        defense_init = defense_init - defense_init.mean()
        
        # Low-score correlation parameter
        rho_init = -0.1
        
        # Combine all parameters
        return np.concatenate([
            [mu_init, gamma_init, rho_init],
            attack_init[:-1],  # Remove last element for constraint
            defense_init[:-1]  # Remove last element for constraint
        ])
    
    def _extract_parameters(self, params: np.ndarray, n_teams: int) -> DCParameters:
        """Extract parameters from optimization result"""
        
        mu = params[0]
        gamma = params[1]
        rho = params[2]
        
        # Extract team parameters
        attack_params = params[3:3 + n_teams - 1]
        defense_params = params[3 + n_teams - 1:]
        
        # Add constraint: sum of parameters = 0
        attack_strengths = np.append(attack_params, -attack_params.sum())
        defense_strengths = np.append(defense_params, -defense_params.sum())
        
        # Create dictionaries
        attack_dict = {team: attack_strengths[i] for i, team in enumerate(self.teams)}
        defense_dict = {team: defense_strengths[i] for i, team in enumerate(self.teams)}
        
        return DCParameters(
            home_advantage=gamma,
            attack_strengths=attack_dict,
            defense_strengths=defense_dict,
            rho=rho,
            mu=mu,
            time_decay=self.time_decay_factor,
            log_likelihood=0.0,  # Will be set later
            convergence_info={}
        )
    
    def _negative_log_likelihood(self, params: np.ndarray, df: pd.DataFrame) -> float:
        """Calculate negative log-likelihood for optimization"""
        
        try:
            n_teams = len(self.teams)
            
            # Extract parameters
            mu = params[0]
            gamma = params[1]
            rho = params[2]
            
            attack_params = params[3:3 + n_teams - 1]
            defense_params = params[3 + n_teams - 1:]
            
            # Apply sum constraints
            attack_strengths = np.append(attack_params, -attack_params.sum())
            defense_strengths = np.append(defense_params, -defense_params.sum())
            
            # Calculate log-likelihood
            log_likelihood = 0.0
            
            for _, match in df.iterrows():
                home_idx = match['home_idx']
                away_idx = match['away_idx']
                home_goals = int(match['home_goals'])
                away_goals = int(match['away_goals'])
                weight = match['weight']
                
                # Expected goals
                lambda_home = np.exp(
                    mu + attack_strengths[home_idx] - defense_strengths[away_idx] + gamma
                )
                lambda_away = np.exp(
                    mu + attack_strengths[away_idx] - defense_strengths[home_idx]
                )
                
                # Basic Poisson probabilities
                prob_home = poisson.pmf(home_goals, lambda_home)
                prob_away = poisson.pmf(away_goals, lambda_away)
                
                # Apply Dixon-Coles correction for low scores
                tau = self._get_tau_correction(home_goals, away_goals, rho)
                
                # Combined probability
                match_prob = prob_home * prob_away * tau
                
                # Add to weighted log-likelihood
                if match_prob > 0:
                    log_likelihood += weight * np.log(match_prob)
                else:
                    log_likelihood += weight * (-1000)  # Penalty for zero probability
            
            # Add regularization
            regularization_penalty = self.regularization * (
                np.sum(attack_strengths**2) + np.sum(defense_strengths**2)
            )
            
            return -(log_likelihood - regularization_penalty)
            
        except Exception as e:
            logger.warning(f"Error in likelihood calculation: {e}")
            return 1e10  # Return large value on error
    
    def _get_tau_correction(
        self,
        home_goals: int,
        away_goals: int,
        rho: float
    ) -> float:
        """Calculate Dixon-Coles tau correction for low scores"""
        
        if (home_goals, away_goals) in self.tau_matrix:
            return 1 + rho * self.tau_matrix[(home_goals, away_goals)]
        else:
            return 1.0
    
    def _calculate_expected_goals(
        self,
        home_team: str,
        away_team: str
    ) -> Tuple[float, float]:
        """Calculate expected goals for both teams"""
        
        lambda_home = np.exp(
            self.parameters.mu + 
            self.parameters.attack_strengths[home_team] - 
            self.parameters.defense_strengths[away_team] + 
            self.parameters.home_advantage
        )
        
        lambda_away = np.exp(
            self.parameters.mu + 
            self.parameters.attack_strengths[away_team] - 
            self.parameters.defense_strengths[home_team]
        )
        
        return lambda_home, lambda_away
    
    def _calculate_scoreline_probabilities(
        self,
        lambda_home: float,
        lambda_away: float,
        max_goals: int
    ) -> Dict[Tuple[int, int], float]:
        """Calculate probabilities for all possible scorelines"""
        
        scoreline_probs = {}
        
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                # Basic Poisson probabilities
                prob_home = poisson.pmf(home_goals, lambda_home)
                prob_away = poisson.pmf(away_goals, lambda_away)
                
                # Apply Dixon-Coles correction
                tau = self._get_tau_correction(home_goals, away_goals, self.parameters.rho)
                
                # Combined probability
                scoreline_probs[(home_goals, away_goals)] = prob_home * prob_away * tau
        
        # Normalize probabilities
        total_prob = sum(scoreline_probs.values())
        if total_prob > 0:
            scoreline_probs = {
                score: prob / total_prob 
                for score, prob in scoreline_probs.items()
            }
        
        return scoreline_probs
    
    def save_parameters(self, filepath: str) -> None:
        """Save model parameters to file"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        import pickle
        
        model_data = {
            'parameters': self.parameters,
            'teams': self.teams,
            'time_decay_factor': self.time_decay_factor,
            'tau_matrix': self.tau_matrix
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model parameters saved to {filepath}")
    
    def load_parameters(self, filepath: str) -> None:
        """Load model parameters from file"""
        
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.parameters = model_data['parameters']
        self.teams = model_data['teams']
        self.time_decay_factor = model_data['time_decay_factor']
        self.tau_matrix = model_data['tau_matrix']
        self.is_fitted = True
        
        logger.info(f"Model parameters loaded from {filepath}")