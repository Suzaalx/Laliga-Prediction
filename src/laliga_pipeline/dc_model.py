"""Enhanced Dixon-Coles Model Implementation

This module implements an enhanced version of the Dixon-Coles model for football match prediction.
The Dixon-Coles model is a bivariate Poisson model that accounts for the correlation between
home and away goals, particularly in low-scoring games.

Key Features:
- Time decay weighting for recent match importance
- Enhanced low-score correlation adjustments
- Venue-specific home advantage effects
- Multiple decay functions (exponential, linear, gaussian)
- Regularization for overfitting prevention

References:
- Dixon, M.J. and Coles, S.G. (1997). Modelling Association Football Scores and Inefficiencies in the Football Betting Market
"""

import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from typing import Dict, Optional, Tuple
import warnings

def _weights(xi: float, dates: pd.Series, decay_type: str = 'exponential') -> np.ndarray:
    """Enhanced time decay weights with multiple decay functions."""
    maxd = dates.max()
    days_diff = (maxd - dates).dt.days.values
    
    if decay_type == 'exponential':
        return np.exp(-xi * days_diff)
    elif decay_type == 'linear':
        max_days = days_diff.max()
        return np.maximum(0, 1 - xi * days_diff / max_days)
    elif decay_type == 'gaussian':
        return np.exp(-0.5 * (xi * days_diff) ** 2)
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")

def _enhanced_correlation(g1: int, g2: int, lh: float, la: float, rho: float) -> float:
    """Enhanced low-score correlation adjustment with additional score combinations."""
    # Original Dixon-Coles adjustments
    if g1 == 0 and g2 == 0:
        return 1 - lh * la * rho
    elif g1 == 0 and g2 == 1:
        return 1 + lh * rho
    elif g1 == 1 and g2 == 0:
        return 1 + la * rho
    elif g1 == 1 and g2 == 1:
        return 1 - rho
    # Enhanced adjustments for additional low-score combinations
    elif g1 == 0 and g2 == 2:
        return 1 + 0.5 * lh * rho
    elif g1 == 2 and g2 == 0:
        return 1 + 0.5 * la * rho
    elif g1 == 2 and g2 == 1:
        return 1 - 0.3 * rho
    elif g1 == 1 and g2 == 2:
        return 1 - 0.3 * rho
    else:
        return 1.0

def _enhanced_home_advantage(home_team: str, away_team: str, venue_effects: Optional[Dict] = None) -> float:
    """Enhanced home advantage with venue-specific effects."""
    base_advantage = 0.25  # Base home advantage
    
    if venue_effects is None:
        return base_advantage
    
    # Add venue-specific adjustments
    venue_boost = venue_effects.get(home_team, 0.0)
    away_penalty = venue_effects.get(f"away_{away_team}", 0.0)
    
    return base_advantage + venue_boost - away_penalty

def _ll_enhanced(params, home_i, away_i, gh, ga, w, n, teams, venue_effects=None, use_enhanced_corr=True):
    """Enhanced log-likelihood function with improved correlation and home advantage."""
    atk = params[:n] - params[:n].mean()
    dfn = params[n:2*n] - params[n:2*n].mean()
    home_adv = params[-2]
    rho = params[-1]
    
    ll = 0.0
    for h, a, g1, g2, wt in zip(home_i, away_i, gh, ga, w):
        # Enhanced home advantage calculation
        if venue_effects is not None:
            home_team = teams[h]
            away_team = teams[a]
            effective_home_adv = _enhanced_home_advantage(home_team, away_team, venue_effects)
        else:
            effective_home_adv = home_adv
        
        # Calculate expected goals
        lh = np.exp(atk[h] - dfn[a] + effective_home_adv)
        la = np.exp(atk[a] - dfn[h])
        
        # Base Poisson probability
        p = poisson.pmf(g1, lh) * poisson.pmf(g2, la)
        
        # Apply correlation adjustment
        if use_enhanced_corr:
            corr_factor = _enhanced_correlation(g1, g2, lh, la, rho)
        else:
            # Original Dixon-Coles correlation
            if g1 == 0 and g2 == 0:
                corr_factor = 1 - lh * la * rho
            elif g1 == 0 and g2 == 1:
                corr_factor = 1 + lh * rho
            elif g1 == 1 and g2 == 0:
                corr_factor = 1 + la * rho
            elif g1 == 1 and g2 == 1:
                corr_factor = 1 - rho
            else:
                corr_factor = 1.0
        
        p *= corr_factor
        ll += wt * np.log(max(p, 1e-12))
    
    return -ll

def _calculate_team_form(df: pd.DataFrame, form_window: int = 6) -> Dict:
    """Calculate recent form metrics for each team."""
    teams = np.unique(np.r_[df["HomeTeam"].values, df["AwayTeam"].values])
    form_metrics = {}
    
    for team in teams:
        # Get recent matches for this team
        home_matches = df[df["HomeTeam"] == team].tail(form_window)
        away_matches = df[df["AwayTeam"] == team].tail(form_window)
        
        # Calculate form metrics
        home_goals_for = home_matches["FTHG"].sum()
        home_goals_against = home_matches["FTAG"].sum()
        away_goals_for = away_matches["FTAG"].sum()
        away_goals_against = away_matches["FTHG"].sum()
        
        total_matches = len(home_matches) + len(away_matches)
        if total_matches > 0:
            avg_goals_for = (home_goals_for + away_goals_for) / total_matches
            avg_goals_against = (home_goals_against + away_goals_against) / total_matches
            form_metrics[team] = {
                'goals_for_avg': avg_goals_for,
                'goals_against_avg': avg_goals_against,
                'goal_difference': avg_goals_for - avg_goals_against
            }
        else:
            form_metrics[team] = {'goals_for_avg': 1.0, 'goals_against_avg': 1.0, 'goal_difference': 0.0}
    
    return form_metrics

def fit_dc_enhanced(df: pd.DataFrame, 
                   xi: float = 0.01,
                   decay_type: str = 'exponential',
                   venue_effects: Optional[Dict] = None,
                   use_enhanced_corr: bool = True,
                   regularization: float = 0.0,
                   min_matches_per_team: int = 10,
                   form_window: int = 6,
                   home_advantage_boost: float = 1.0) -> Dict:
    """Enhanced Dixon-Coles model with improved features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Match data with columns: HomeTeam, AwayTeam, FTHG, FTAG, Date
    xi : float
        Time decay parameter (higher = more recent matches weighted more)
    decay_type : str
        Type of time decay ('exponential', 'linear', 'gaussian')
    venue_effects : dict, optional
        Venue-specific home advantage adjustments
    use_enhanced_corr : bool
        Whether to use enhanced correlation adjustments
    regularization : float
        L2 regularization strength for parameters
    
    Returns:
    --------
    dict : Model parameters including attack/defense strengths, home advantage, and rho
    """
    # Filter teams with minimum matches
    team_counts = pd.concat([df["HomeTeam"], df["AwayTeam"]]).value_counts()
    valid_teams = team_counts[team_counts >= min_matches_per_team].index
    df_filtered = df[(df["HomeTeam"].isin(valid_teams)) & (df["AwayTeam"].isin(valid_teams))]
    
    teams = np.sort(np.unique(np.r_[df_filtered["HomeTeam"].values, df_filtered["AwayTeam"].values]))
    idx = {t: i for i, t in enumerate(teams)}
    
    home_i = df_filtered["HomeTeam"].map(idx).values
    away_i = df_filtered["AwayTeam"].map(idx).values
    gh = df_filtered["FTHG"].values
    ga = df_filtered["FTAG"].values
    
    # Calculate team form metrics
    form_metrics = _calculate_team_form(df_filtered, form_window)
    
    # Enhanced time weights
    w = _weights(xi, df_filtered["Date"], decay_type)
    
    n = len(teams)
    
    # Initialize parameters with form-based starting values
    attack_init = np.zeros(n)
    defense_init = np.zeros(n)
    
    for i, team in enumerate(teams):
        if team in form_metrics:
            # Initialize based on recent form
            attack_init[i] = np.log(max(form_metrics[team]['goals_for_avg'], 0.1))
            defense_init[i] = -np.log(max(form_metrics[team]['goals_against_avg'], 0.1))
        else:
            attack_init[i] = np.random.normal(0, 0.1)
            defense_init[i] = np.random.normal(0, 0.1)
    
    home_adv_init = 0.25 * home_advantage_boost
    rho_init = -0.1
    
    x0 = np.r_[attack_init, defense_init, np.array([home_adv_init, rho_init])]
    
    # Enhanced optimization with regularization
    def objective(params):
        ll = _ll_enhanced(params, home_i, away_i, gh, ga, w, n, teams, 
                         venue_effects, use_enhanced_corr)
        # Add L2 regularization
        if regularization > 0:
            reg_penalty = regularization * np.sum(params[:-2] ** 2)
            ll += reg_penalty
        return ll
    
    # Parameter bounds
    bounds = [(None, None)] * (2 * n) + [(0.0, 1.0), (-0.5, 0.5)]  # home_adv, rho bounds
    
    try:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 1000, "ftol": 1e-9})
        
        if not res.success:
            warnings.warn(f"Optimization did not converge: {res.message}")
        
        x = res.x
        # Ensure zero-sum constraints
        x[:n] -= x[:n].mean()
        x[n:2*n] -= x[n:2*n].mean()
        
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}. Using fallback method.")
        # Fallback to original method
        return fit_dc(df, xi)
    
    # Build output dictionary
    out = {f"attack_{t}": x[i] for i, t in enumerate(teams)}
    out.update({f"defense_{t}": x[n+i] for i, t in enumerate(teams)})
    out["home_adv"] = x[-2]
    out["rho"] = x[-1]
    out["teams"] = teams.tolist()
    out["xi"] = xi
    out["decay_type"] = decay_type
    out["log_likelihood"] = -res.fun if 'res' in locals() else None
    
    return out

def predict_match(model: Dict, home_team: str, away_team: str, 
                 max_goals: int = 10) -> Tuple[np.ndarray, Dict]:
    """Predict match outcome probabilities using the enhanced model.
    
    Parameters:
    -----------
    model : dict
        Fitted model parameters
    home_team : str
        Home team name
    away_team : str
        Away team name
    max_goals : int
        Maximum goals to consider for probability matrix
    
    Returns:
    --------
    tuple : (probability_matrix, summary_probabilities)
    """
    try:
        home_attack = model[f"attack_{home_team}"]
        home_defense = model[f"defense_{home_team}"]
        away_attack = model[f"attack_{away_team}"]
        away_defense = model[f"defense_{away_team}"]
        home_adv = model["home_adv"]
        rho = model["rho"]
    except KeyError as e:
        raise ValueError(f"Team not found in model: {e}")
    
    # Expected goals
    lh = np.exp(home_attack - away_defense + home_adv)
    la = np.exp(away_attack - home_defense)
    
    # Probability matrix
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob = poisson.pmf(i, lh) * poisson.pmf(j, la)
            # Apply correlation adjustment
            corr_factor = _enhanced_correlation(i, j, lh, la, rho)
            prob_matrix[i, j] = prob * corr_factor
    
    # Normalize probabilities
    prob_matrix /= prob_matrix.sum()
    
    # Summary probabilities
    home_win = np.sum(np.triu(prob_matrix, k=1))
    draw = np.sum(np.diag(prob_matrix))
    away_win = np.sum(np.tril(prob_matrix, k=-1))
    
    summary = {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "expected_home_goals": lh,
        "expected_away_goals": la,
        "most_likely_score": np.unravel_index(prob_matrix.argmax(), prob_matrix.shape)
    }
    
    return prob_matrix, summary

# Backward compatibility
def fit_dc(df: pd.DataFrame, xi: float = 0.001):
    """Original Dixon-Coles implementation for backward compatibility."""
    return fit_dc_enhanced(df, xi, decay_type='exponential', 
                          use_enhanced_corr=False, regularization=0.0)
