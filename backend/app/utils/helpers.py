import re
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from difflib import SequenceMatcher, get_close_matches
from collections import defaultdict

from app.core.logging import get_logger

logger = get_logger("helpers")

def fuzzy_match_team(
    team_name: str,
    known_teams: List[str],
    threshold: float = 0.8
) -> Optional[str]:
    """Find the best fuzzy match for a team name
    
    Args:
        team_name: Team name to match
        known_teams: List of known team names
        threshold: Minimum similarity threshold (0-1)
    
    Returns:
        Best matching team name or None if no good match found
    """
    
    if not team_name or not known_teams:
        return None
    
    # Clean the input team name
    clean_name = clean_team_name(team_name)
    
    # First try exact match
    for known_team in known_teams:
        if clean_team_name(known_team) == clean_name:
            return known_team
    
    # Try fuzzy matching
    best_match = None
    best_score = 0
    
    for known_team in known_teams:
        # Calculate similarity using multiple methods
        scores = [
            SequenceMatcher(None, clean_name.lower(), clean_team_name(known_team).lower()).ratio(),
            _jaro_winkler_similarity(clean_name.lower(), clean_team_name(known_team).lower()),
            _token_similarity(clean_name.lower(), clean_team_name(known_team).lower())
        ]
        
        # Use weighted average of similarity scores
        score = np.mean(scores)
        
        if score > best_score and score >= threshold:
            best_score = score
            best_match = known_team
    
    if best_match:
        logger.debug(f"Fuzzy matched '{team_name}' to '{best_match}' (score: {best_score:.3f})")
    
    return best_match

def clean_team_name(team_name: str) -> str:
    """Clean and standardize team name
    
    Args:
        team_name: Raw team name
    
    Returns:
        Cleaned team name
    """
    
    if not team_name:
        return ""
    
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', team_name.strip())
    
    # Common replacements
    replacements = {
        'CF': 'Club de Fútbol',
        'FC': 'Fútbol Club',
        'UD': 'Unión Deportiva',
        'CD': 'Club Deportivo',
        'RCD': 'Real Club Deportivo',
        'RC': 'Real Club',
        'SD': 'Sociedad Deportiva',
        'AD': 'Agrupación Deportiva'
    }
    
    # Apply replacements (case insensitive)
    for abbr, full in replacements.items():
        pattern = r'\b' + re.escape(abbr) + r'\b'
        cleaned = re.sub(pattern, full, cleaned, flags=re.IGNORECASE)
    
    return cleaned

def calculate_form_points(
    results: List[str],
    weights: Optional[List[float]] = None
) -> float:
    """Calculate form points from recent results
    
    Args:
        results: List of results ('H', 'D', 'A' for home team perspective)
        weights: Optional weights for each result (most recent first)
    
    Returns:
        Weighted form points
    """
    
    if not results:
        return 0.0
    
    # Default weights (exponential decay)
    if weights is None:
        weights = [0.9 ** i for i in range(len(results))]
    
    # Ensure weights match results length
    if len(weights) != len(results):
        weights = weights[:len(results)] + [weights[-1]] * (len(results) - len(weights))
    
    # Points mapping
    points_map = {'H': 3, 'D': 1, 'A': 0}  # From home team perspective
    
    total_points = 0.0
    total_weight = 0.0
    
    for result, weight in zip(results, weights):
        if result in points_map:
            total_points += points_map[result] * weight
            total_weight += weight
    
    return total_points / total_weight if total_weight > 0 else 0.0

def get_rolling_average(
    values: List[float],
    window: int,
    min_periods: int = 1
) -> List[float]:
    """Calculate rolling average with minimum periods
    
    Args:
        values: List of values
        window: Rolling window size
        min_periods: Minimum number of periods required
    
    Returns:
        List of rolling averages
    """
    
    if not values:
        return []
    
    # Convert to pandas Series for easier calculation
    series = pd.Series(values)
    rolling_avg = series.rolling(window=window, min_periods=min_periods).mean()
    
    return rolling_avg.fillna(0).tolist()

def calculate_elo_rating(
    team_rating: float,
    opponent_rating: float,
    actual_score: float,
    k_factor: float = 32.0,
    home_advantage: float = 100.0,
    is_home: bool = True
) -> float:
    """Calculate new Elo rating after a match
    
    Args:
        team_rating: Current team Elo rating
        opponent_rating: Current opponent Elo rating
        actual_score: Actual match result (1 for win, 0.5 for draw, 0 for loss)
        k_factor: K-factor for rating adjustment
        home_advantage: Home advantage in Elo points
        is_home: Whether the team played at home
    
    Returns:
        New Elo rating
    """
    
    # Adjust for home advantage
    adjusted_team_rating = team_rating + (home_advantage if is_home else 0)
    adjusted_opponent_rating = opponent_rating + (0 if is_home else home_advantage)
    
    # Calculate expected score
    rating_diff = adjusted_opponent_rating - adjusted_team_rating
    expected_score = 1 / (1 + 10 ** (rating_diff / 400))
    
    # Calculate new rating
    new_rating = team_rating + k_factor * (actual_score - expected_score)
    
    return new_rating

def calculate_poisson_probability(
    lambda_param: float,
    k: int
) -> float:
    """Calculate Poisson probability
    
    Args:
        lambda_param: Poisson parameter (expected value)
        k: Number of events
    
    Returns:
        Probability of exactly k events
    """
    
    if lambda_param <= 0 or k < 0:
        return 0.0
    
    # Use scipy for numerical stability
    from scipy.stats import poisson
    return poisson.pmf(k, lambda_param)

def calculate_match_importance(
    match_date: datetime,
    season_start: datetime,
    season_end: datetime,
    importance_curve: str = 'linear'
) -> float:
    """Calculate match importance based on timing in season
    
    Args:
        match_date: Date of the match
        season_start: Start date of the season
        season_end: End date of the season
        importance_curve: Type of importance curve ('linear', 'exponential', 'sigmoid')
    
    Returns:
        Importance factor (0-1)
    """
    
    # Calculate season progress (0-1)
    total_days = (season_end - season_start).days
    days_elapsed = (match_date - season_start).days
    
    if total_days <= 0:
        return 1.0
    
    progress = max(0, min(1, days_elapsed / total_days))
    
    # Apply importance curve
    if importance_curve == 'linear':
        return 0.5 + 0.5 * progress  # 0.5 to 1.0
    elif importance_curve == 'exponential':
        return np.exp(progress) / np.exp(1)  # Normalized exponential
    elif importance_curve == 'sigmoid':
        # Sigmoid curve centered at mid-season
        x = (progress - 0.5) * 10  # Scale to -5 to 5
        return 1 / (1 + np.exp(-x))
    else:
        return 1.0

def normalize_features(
    features: Dict[str, float],
    feature_stats: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, float]:
    """Normalize features using z-score or min-max scaling
    
    Args:
        features: Dictionary of feature values
        feature_stats: Dictionary of feature statistics (mean, std, min, max)
    
    Returns:
        Dictionary of normalized features
    """
    
    if not feature_stats:
        return features
    
    normalized = {}
    
    for feature, value in features.items():
        if feature in feature_stats:
            stats = feature_stats[feature]
            
            # Use z-score normalization if std is available
            if 'mean' in stats and 'std' in stats and stats['std'] > 0:
                normalized[feature] = (value - stats['mean']) / stats['std']
            # Use min-max normalization if min/max available
            elif 'min' in stats and 'max' in stats and stats['max'] > stats['min']:
                normalized[feature] = (value - stats['min']) / (stats['max'] - stats['min'])
            else:
                normalized[feature] = value
        else:
            normalized[feature] = value
    
    return normalized

def calculate_team_strength(
    goals_scored: List[int],
    goals_conceded: List[int],
    venue: List[str],
    decay_factor: float = 0.95
) -> Dict[str, float]:
    """Calculate team strength metrics with time decay
    
    Args:
        goals_scored: List of goals scored (most recent first)
        goals_conceded: List of goals conceded (most recent first)
        venue: List of venues ('H' for home, 'A' for away)
        decay_factor: Time decay factor
    
    Returns:
        Dictionary of strength metrics
    """
    
    if not goals_scored or len(goals_scored) != len(goals_conceded) or len(goals_scored) != len(venue):
        return {
            'attack_strength': 1.0,
            'defense_strength': 1.0,
            'home_attack_strength': 1.0,
            'home_defense_strength': 1.0,
            'away_attack_strength': 1.0,
            'away_defense_strength': 1.0
        }
    
    # Calculate weighted averages
    weights = [decay_factor ** i for i in range(len(goals_scored))]
    total_weight = sum(weights)
    
    # Overall strength
    weighted_goals_scored = sum(g * w for g, w in zip(goals_scored, weights)) / total_weight
    weighted_goals_conceded = sum(g * w for g, w in zip(goals_conceded, weights)) / total_weight
    
    # Venue-specific strength
    home_goals_scored = [g for g, v in zip(goals_scored, venue) if v == 'H']
    home_goals_conceded = [g for g, v in zip(goals_conceded, venue) if v == 'H']
    away_goals_scored = [g for g, v in zip(goals_scored, venue) if v == 'A']
    away_goals_conceded = [g for g, v in zip(goals_conceded, venue) if v == 'A']
    
    home_weights = weights[:len(home_goals_scored)]
    away_weights = weights[:len(away_goals_scored)]
    
    home_attack = (sum(g * w for g, w in zip(home_goals_scored, home_weights)) / sum(home_weights)) if home_weights else weighted_goals_scored
    home_defense = (sum(g * w for g, w in zip(home_goals_conceded, home_weights)) / sum(home_weights)) if home_weights else weighted_goals_conceded
    away_attack = (sum(g * w for g, w in zip(away_goals_scored, away_weights)) / sum(away_weights)) if away_weights else weighted_goals_scored
    away_defense = (sum(g * w for g, w in zip(away_goals_conceded, away_weights)) / sum(away_weights)) if away_weights else weighted_goals_conceded
    
    # Convert to strength ratios (relative to league average)
    league_avg_goals = 1.5  # Typical league average
    
    return {
        'attack_strength': max(0.1, weighted_goals_scored / league_avg_goals),
        'defense_strength': max(0.1, league_avg_goals / max(0.1, weighted_goals_conceded)),
        'home_attack_strength': max(0.1, home_attack / league_avg_goals),
        'home_defense_strength': max(0.1, league_avg_goals / max(0.1, home_defense)),
        'away_attack_strength': max(0.1, away_attack / league_avg_goals),
        'away_defense_strength': max(0.1, league_avg_goals / max(0.1, away_defense))
    }

def calculate_head_to_head_stats(
    matches: List[Dict[str, Any]],
    team1: str,
    team2: str,
    max_matches: int = 10
) -> Dict[str, Any]:
    """Calculate head-to-head statistics between two teams
    
    Args:
        matches: List of match dictionaries
        team1: First team name
        team2: Second team name
        max_matches: Maximum number of recent matches to consider
    
    Returns:
        Dictionary of head-to-head statistics
    """
    
    # Filter head-to-head matches
    h2h_matches = []
    for match in matches:
        home_team = match.get('home_team', '').strip().lower()
        away_team = match.get('away_team', '').strip().lower()
        team1_lower = team1.strip().lower()
        team2_lower = team2.strip().lower()
        
        if ((home_team == team1_lower and away_team == team2_lower) or
            (home_team == team2_lower and away_team == team1_lower)):
            h2h_matches.append(match)
    
    # Sort by date (most recent first) and limit
    h2h_matches = sorted(h2h_matches, key=lambda x: x.get('date', datetime.min), reverse=True)[:max_matches]
    
    if not h2h_matches:
        return {
            'total_matches': 0,
            'team1_wins': 0,
            'team2_wins': 0,
            'draws': 0,
            'team1_goals': 0,
            'team2_goals': 0,
            'avg_goals_per_match': 0,
            'team1_win_rate': 0,
            'recent_form': []
        }
    
    # Calculate statistics
    team1_wins = 0
    team2_wins = 0
    draws = 0
    team1_goals = 0
    team2_goals = 0
    recent_form = []
    
    team1_lower = team1.strip().lower()
    
    for match in h2h_matches:
        home_team = match.get('home_team', '').strip().lower()
        away_team = match.get('away_team', '').strip().lower()
        home_goals = match.get('home_goals', 0) or 0
        away_goals = match.get('away_goals', 0) or 0
        
        # Determine team1's perspective
        if home_team == team1_lower:
            team1_goals += home_goals
            team2_goals += away_goals
            if home_goals > away_goals:
                team1_wins += 1
                recent_form.append('W')
            elif home_goals < away_goals:
                team2_wins += 1
                recent_form.append('L')
            else:
                draws += 1
                recent_form.append('D')
        else:
            team1_goals += away_goals
            team2_goals += home_goals
            if away_goals > home_goals:
                team1_wins += 1
                recent_form.append('W')
            elif away_goals < home_goals:
                team2_wins += 1
                recent_form.append('L')
            else:
                draws += 1
                recent_form.append('D')
    
    total_matches = len(h2h_matches)
    
    return {
        'total_matches': total_matches,
        'team1_wins': team1_wins,
        'team2_wins': team2_wins,
        'draws': draws,
        'team1_goals': team1_goals,
        'team2_goals': team2_goals,
        'avg_goals_per_match': (team1_goals + team2_goals) / total_matches,
        'team1_win_rate': team1_wins / total_matches,
        'recent_form': recent_form[:5]  # Last 5 matches
    }

def _jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Calculate Jaro-Winkler similarity between two strings"""
    
    if s1 == s2:
        return 1.0
    
    len1, len2 = len(s1), len(s2)
    
    if len1 == 0 or len2 == 0:
        return 0.0
    
    # Calculate match window
    match_window = max(len1, len2) // 2 - 1
    match_window = max(0, match_window)
    
    # Initialize match arrays
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    
    matches = 0
    transpositions = 0
    
    # Find matches
    for i in range(len1):
        start = max(0, i - match_window)
        end = min(i + match_window + 1, len2)
        
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    
    if matches == 0:
        return 0.0
    
    # Count transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    
    # Calculate Jaro similarity
    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3
    
    # Calculate Jaro-Winkler similarity
    prefix = 0
    for i in range(min(len1, len2, 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    
    return jaro + 0.1 * prefix * (1 - jaro)

def _token_similarity(s1: str, s2: str) -> float:
    """Calculate token-based similarity between two strings"""
    
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    
    if not tokens1 and not tokens2:
        return 1.0
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    return len(intersection) / len(union)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    return numerator / denominator if denominator != 0 else default

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max"""
    return max(min_val, min(max_val, value))

def exponential_decay_weights(n: int, decay_factor: float = 0.95) -> List[float]:
    """Generate exponential decay weights
    
    Args:
        n: Number of weights to generate
        decay_factor: Decay factor (0-1)
    
    Returns:
        List of weights (most recent first)
    """
    return [decay_factor ** i for i in range(n)]

def linear_decay_weights(n: int) -> List[float]:
    """Generate linear decay weights
    
    Args:
        n: Number of weights to generate
    
    Returns:
        List of weights (most recent first)
    """
    if n <= 0:
        return []
    
    return [(n - i) / n for i in range(n)]