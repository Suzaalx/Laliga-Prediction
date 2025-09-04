import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger("feature_engineering")

@dataclass
class TeamFeatures:
    """Team features for a specific match"""
    team_name: str
    venue: str  # 'home' or 'away'
    
    # Rolling averages (last N matches)
    avg_goals_scored: float
    avg_goals_conceded: float
    avg_shots: float
    avg_shots_target: float
    avg_shots_conceded: float
    avg_shots_target_conceded: float
    avg_corners: float
    avg_corners_conceded: float
    avg_fouls: float
    avg_fouls_conceded: float
    avg_yellow_cards: float
    avg_red_cards: float
    
    # Form metrics
    form_points: float
    win_percentage: float
    recent_form: str  # e.g., "WWDLW"
    
    # Venue-specific metrics
    home_advantage: float
    venue_form_points: float
    
    # Head-to-head metrics
    h2h_goals_for: float
    h2h_goals_against: float
    h2h_win_percentage: float
    
    # Advanced metrics
    attack_strength: float
    defense_strength: float
    shot_conversion_rate: float
    shot_accuracy: float
    discipline_index: float
    
    # Referee-adjusted metrics
    referee_adjusted_cards: float
    referee_adjusted_fouls: float

class FeatureEngineer:
    """Feature engineering for football match prediction"""
    
    def __init__(
        self,
        rolling_window: int = 10,
        min_matches: int = 5,
        form_window: int = 5,
        h2h_window: int = 10
    ):
        self.rolling_window = rolling_window
        self.min_matches = min_matches
        self.form_window = form_window
        self.h2h_window = h2h_window
        
        # Cache for computed features
        self._feature_cache = {}
        self._referee_stats_cache = {}
    
    def engineer_match_features(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
        referee: Optional[str],
        matches_df: pd.DataFrame,
        referee_stats_df: Optional[pd.DataFrame] = None
    ) -> Tuple[TeamFeatures, TeamFeatures]:
        """Engineer features for both teams in a match
        
        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Date of the match
            referee: Referee name (optional)
            matches_df: Historical match data
            referee_stats_df: Referee statistics (optional)
        
        Returns:
            Tuple of (home_team_features, away_team_features)
        """
        
        logger.debug(f"Engineering features for {home_team} vs {away_team} on {match_date}")
        
        # Filter historical data (before match date)
        historical_matches = matches_df[matches_df['date'] < match_date].copy()
        
        if historical_matches.empty:
            logger.warning(f"No historical data available for match on {match_date}")
            return self._get_default_features(home_team, away_team)
        
        # Get referee statistics
        referee_stats = self._get_referee_stats(referee, referee_stats_df) if referee else {}
        
        # Engineer features for both teams
        home_features = self._engineer_team_features(
            home_team, 'home', match_date, away_team, historical_matches, referee_stats
        )
        
        away_features = self._engineer_team_features(
            away_team, 'away', match_date, home_team, historical_matches, referee_stats
        )
        
        return home_features, away_features
    
    def _engineer_team_features(
        self,
        team: str,
        venue: str,
        match_date: datetime,
        opponent: str,
        historical_matches: pd.DataFrame,
        referee_stats: Dict[str, float]
    ) -> TeamFeatures:
        """Engineer features for a single team"""
        
        # Get team's recent matches
        team_matches = self._get_team_matches(team, historical_matches, self.rolling_window)
        
        if len(team_matches) < self.min_matches:
            logger.warning(f"Insufficient data for {team}: {len(team_matches)} matches")
            return self._get_default_team_features(team, venue)
        
        # Calculate basic rolling averages
        basic_stats = self._calculate_basic_rolling_stats(team, team_matches)
        
        # Calculate form metrics
        form_stats = self._calculate_form_metrics(team, team_matches, self.form_window)
        
        # Calculate venue-specific metrics
        venue_stats = self._calculate_venue_metrics(team, venue, historical_matches)
        
        # Calculate head-to-head metrics
        h2h_stats = self._calculate_h2h_metrics(team, opponent, historical_matches, self.h2h_window)
        
        # Calculate advanced metrics
        advanced_stats = self._calculate_advanced_metrics(team, team_matches, historical_matches)
        
        # Apply referee adjustments
        referee_adjusted = self._apply_referee_adjustments(
            basic_stats, referee_stats
        )
        
        return TeamFeatures(
            team_name=team,
            venue=venue,
            **basic_stats,
            **form_stats,
            **venue_stats,
            **h2h_stats,
            **advanced_stats,
            **referee_adjusted
        )
    
    def _get_team_matches(
        self,
        team: str,
        matches_df: pd.DataFrame,
        limit: int
    ) -> pd.DataFrame:
        """Get recent matches for a team"""
        
        team_matches = matches_df[
            (matches_df['home_team'] == team) | (matches_df['away_team'] == team)
        ].copy()
        
        # Sort by date (most recent first) and limit
        team_matches = team_matches.sort_values('date', ascending=False).head(limit)
        
        return team_matches
    
    def _calculate_basic_rolling_stats(
        self,
        team: str,
        team_matches: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate basic rolling statistics"""
        
        stats = {
            'goals_for': [],
            'goals_against': [],
            'shots_for': [],
            'shots_against': [],
            'shots_target_for': [],
            'shots_target_against': [],
            'corners_for': [],
            'corners_against': [],
            'fouls_for': [],
            'fouls_against': [],
            'yellow_cards': [],
            'red_cards': []
        }
        
        for _, match in team_matches.iterrows():
            is_home = match['home_team'] == team
            
            if is_home:
                stats['goals_for'].append(match.get('home_goals', 0) or 0)
                stats['goals_against'].append(match.get('away_goals', 0) or 0)
                stats['shots_for'].append(match.get('home_shots', 0) or 0)
                stats['shots_against'].append(match.get('away_shots', 0) or 0)
                stats['shots_target_for'].append(match.get('home_shots_target', 0) or 0)
                stats['shots_target_against'].append(match.get('away_shots_target', 0) or 0)
                stats['corners_for'].append(match.get('home_corners', 0) or 0)
                stats['corners_against'].append(match.get('away_corners', 0) or 0)
                stats['fouls_for'].append(match.get('home_fouls', 0) or 0)
                stats['fouls_against'].append(match.get('away_fouls', 0) or 0)
                stats['yellow_cards'].append(match.get('home_yellow_cards', 0) or 0)
                stats['red_cards'].append(match.get('home_red_cards', 0) or 0)
            else:
                stats['goals_for'].append(match.get('away_goals', 0) or 0)
                stats['goals_against'].append(match.get('home_goals', 0) or 0)
                stats['shots_for'].append(match.get('away_shots', 0) or 0)
                stats['shots_against'].append(match.get('home_shots', 0) or 0)
                stats['shots_target_for'].append(match.get('away_shots_target', 0) or 0)
                stats['shots_target_against'].append(match.get('home_shots_target', 0) or 0)
                stats['corners_for'].append(match.get('away_corners', 0) or 0)
                stats['corners_against'].append(match.get('home_corners', 0) or 0)
                stats['fouls_for'].append(match.get('away_fouls', 0) or 0)
                stats['fouls_against'].append(match.get('home_fouls', 0) or 0)
                stats['yellow_cards'].append(match.get('away_yellow_cards', 0) or 0)
                stats['red_cards'].append(match.get('away_red_cards', 0) or 0)
        
        return {
            'avg_goals_scored': np.mean(stats['goals_for']),
            'avg_goals_conceded': np.mean(stats['goals_against']),
            'avg_shots': np.mean(stats['shots_for']),
            'avg_shots_target': np.mean(stats['shots_target_for']),
            'avg_shots_conceded': np.mean(stats['shots_against']),
            'avg_shots_target_conceded': np.mean(stats['shots_target_against']),
            'avg_corners': np.mean(stats['corners_for']),
            'avg_corners_conceded': np.mean(stats['corners_against']),
            'avg_fouls': np.mean(stats['fouls_for']),
            'avg_fouls_conceded': np.mean(stats['fouls_against']),
            'avg_yellow_cards': np.mean(stats['yellow_cards']),
            'avg_red_cards': np.mean(stats['red_cards'])
        }
    
    def _calculate_form_metrics(
        self,
        team: str,
        team_matches: pd.DataFrame,
        form_window: int
    ) -> Dict[str, Any]:
        """Calculate form-based metrics"""
        
        recent_matches = team_matches.head(form_window)
        
        points = []
        results = []
        
        for _, match in recent_matches.iterrows():
            is_home = match['home_team'] == team
            home_goals = match.get('home_goals', 0) or 0
            away_goals = match.get('away_goals', 0) or 0
            
            if is_home:
                team_goals = home_goals
                opponent_goals = away_goals
            else:
                team_goals = away_goals
                opponent_goals = home_goals
            
            if team_goals > opponent_goals:
                points.append(3)
                results.append('W')
            elif team_goals == opponent_goals:
                points.append(1)
                results.append('D')
            else:
                points.append(0)
                results.append('L')
        
        total_points = sum(points)
        max_points = len(points) * 3
        
        return {
            'form_points': total_points,
            'win_percentage': (results.count('W') / len(results)) * 100 if results else 0,
            'recent_form': ''.join(results[:5])  # Last 5 matches
        }
    
    def _calculate_venue_metrics(
        self,
        team: str,
        venue: str,
        historical_matches: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate venue-specific metrics"""
        
        if venue == 'home':
            venue_matches = historical_matches[historical_matches['home_team'] == team]
        else:
            venue_matches = historical_matches[historical_matches['away_team'] == team]
        
        if venue_matches.empty:
            return {
                'home_advantage': 0.0,
                'venue_form_points': 0.0
            }
        
        # Calculate venue form
        venue_points = []
        for _, match in venue_matches.head(self.form_window).iterrows():
            if venue == 'home':
                team_goals = match.get('home_goals', 0) or 0
                opponent_goals = match.get('away_goals', 0) or 0
            else:
                team_goals = match.get('away_goals', 0) or 0
                opponent_goals = match.get('home_goals', 0) or 0
            
            if team_goals > opponent_goals:
                venue_points.append(3)
            elif team_goals == opponent_goals:
                venue_points.append(1)
            else:
                venue_points.append(0)
        
        # Calculate home advantage (difference between home and away performance)
        home_matches = historical_matches[historical_matches['home_team'] == team]
        away_matches = historical_matches[historical_matches['away_team'] == team]
        
        home_ppg = self._calculate_points_per_game(team, home_matches, 'home')
        away_ppg = self._calculate_points_per_game(team, away_matches, 'away')
        
        return {
            'home_advantage': home_ppg - away_ppg,
            'venue_form_points': sum(venue_points)
        }
    
    def _calculate_h2h_metrics(
        self,
        team: str,
        opponent: str,
        historical_matches: pd.DataFrame,
        h2h_window: int
    ) -> Dict[str, float]:
        """Calculate head-to-head metrics"""
        
        h2h_matches = historical_matches[
            ((historical_matches['home_team'] == team) & (historical_matches['away_team'] == opponent)) |
            ((historical_matches['home_team'] == opponent) & (historical_matches['away_team'] == team))
        ].head(h2h_window)
        
        if h2h_matches.empty:
            return {
                'h2h_goals_for': 0.0,
                'h2h_goals_against': 0.0,
                'h2h_win_percentage': 0.0
            }
        
        goals_for = []
        goals_against = []
        wins = 0
        
        for _, match in h2h_matches.iterrows():
            is_home = match['home_team'] == team
            
            if is_home:
                team_goals = match.get('home_goals', 0) or 0
                opponent_goals = match.get('away_goals', 0) or 0
            else:
                team_goals = match.get('away_goals', 0) or 0
                opponent_goals = match.get('home_goals', 0) or 0
            
            goals_for.append(team_goals)
            goals_against.append(opponent_goals)
            
            if team_goals > opponent_goals:
                wins += 1
        
        return {
            'h2h_goals_for': np.mean(goals_for),
            'h2h_goals_against': np.mean(goals_against),
            'h2h_win_percentage': (wins / len(h2h_matches)) * 100
        }
    
    def _calculate_advanced_metrics(
        self,
        team: str,
        team_matches: pd.DataFrame,
        all_matches: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate advanced performance metrics"""
        
        # Calculate league averages for normalization
        league_avg_goals = self._calculate_league_average_goals(all_matches)
        
        # Team's attacking and defensive performance
        team_goals_for = []
        team_goals_against = []
        team_shots = []
        team_shots_target = []
        
        for _, match in team_matches.iterrows():
            is_home = match['home_team'] == team
            
            if is_home:
                team_goals_for.append(match.get('home_goals', 0) or 0)
                team_goals_against.append(match.get('away_goals', 0) or 0)
                team_shots.append(match.get('home_shots', 0) or 0)
                team_shots_target.append(match.get('home_shots_target', 0) or 0)
            else:
                team_goals_for.append(match.get('away_goals', 0) or 0)
                team_goals_against.append(match.get('home_goals', 0) or 0)
                team_shots.append(match.get('away_shots', 0) or 0)
                team_shots_target.append(match.get('away_shots_target', 0) or 0)
        
        avg_goals_for = np.mean(team_goals_for)
        avg_goals_against = np.mean(team_goals_against)
        avg_shots = np.mean(team_shots)
        avg_shots_target = np.mean(team_shots_target)
        
        # Calculate strengths relative to league average
        attack_strength = avg_goals_for / league_avg_goals if league_avg_goals > 0 else 1.0
        defense_strength = league_avg_goals / avg_goals_against if avg_goals_against > 0 else 1.0
        
        # Shot conversion and accuracy
        shot_conversion_rate = avg_goals_for / avg_shots if avg_shots > 0 else 0.0
        shot_accuracy = avg_shots_target / avg_shots if avg_shots > 0 else 0.0
        
        # Discipline index (lower is better)
        avg_yellow_cards = np.mean([match.get('home_yellow_cards', 0) if match['home_team'] == team 
                                   else match.get('away_yellow_cards', 0) for _, match in team_matches.iterrows()])
        avg_red_cards = np.mean([match.get('home_red_cards', 0) if match['home_team'] == team 
                                else match.get('away_red_cards', 0) for _, match in team_matches.iterrows()])
        
        discipline_index = avg_yellow_cards + (avg_red_cards * 3)  # Weight red cards more heavily
        
        return {
            'attack_strength': attack_strength,
            'defense_strength': defense_strength,
            'shot_conversion_rate': shot_conversion_rate,
            'shot_accuracy': shot_accuracy,
            'discipline_index': discipline_index
        }
    
    def _apply_referee_adjustments(
        self,
        basic_stats: Dict[str, float],
        referee_stats: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply referee-specific adjustments"""
        
        if not referee_stats:
            return {
                'referee_adjusted_cards': basic_stats.get('avg_yellow_cards', 0),
                'referee_adjusted_fouls': basic_stats.get('avg_fouls', 0)
            }
        
        # Adjust based on referee tendencies
        ref_card_factor = referee_stats.get('avg_cards_per_match', 4.0) / 4.0  # Normalize to league average
        ref_foul_factor = referee_stats.get('avg_fouls_per_match', 20.0) / 20.0
        
        return {
            'referee_adjusted_cards': basic_stats.get('avg_yellow_cards', 0) * ref_card_factor,
            'referee_adjusted_fouls': basic_stats.get('avg_fouls', 0) * ref_foul_factor
        }
    
    def _calculate_points_per_game(
        self,
        team: str,
        matches: pd.DataFrame,
        venue: str
    ) -> float:
        """Calculate points per game for a team at a specific venue"""
        
        if matches.empty:
            return 0.0
        
        total_points = 0
        
        for _, match in matches.iterrows():
            if venue == 'home':
                team_goals = match.get('home_goals', 0) or 0
                opponent_goals = match.get('away_goals', 0) or 0
            else:
                team_goals = match.get('away_goals', 0) or 0
                opponent_goals = match.get('home_goals', 0) or 0
            
            if team_goals > opponent_goals:
                total_points += 3
            elif team_goals == opponent_goals:
                total_points += 1
        
        return total_points / len(matches)
    
    def _calculate_league_average_goals(
        self,
        matches: pd.DataFrame
    ) -> float:
        """Calculate league average goals per match"""
        
        if matches.empty:
            return 1.5  # Default value
        
        total_goals = matches['home_goals'].sum() + matches['away_goals'].sum()
        total_matches = len(matches)
        
        return total_goals / (2 * total_matches) if total_matches > 0 else 1.5
    
    def _get_referee_stats(
        self,
        referee: str,
        referee_stats_df: Optional[pd.DataFrame]
    ) -> Dict[str, float]:
        """Get referee statistics"""
        
        if not referee or referee_stats_df is None or referee_stats_df.empty:
            return {}
        
        referee_row = referee_stats_df[referee_stats_df['referee_name'] == referee]
        
        if referee_row.empty:
            return {}
        
        return referee_row.iloc[0].to_dict()
    
    def _get_default_features(
        self,
        home_team: str,
        away_team: str
    ) -> Tuple[TeamFeatures, TeamFeatures]:
        """Get default features when no historical data is available"""
        
        home_features = self._get_default_team_features(home_team, 'home')
        away_features = self._get_default_team_features(away_team, 'away')
        
        return home_features, away_features
    
    def _get_default_team_features(
        self,
        team: str,
        venue: str
    ) -> TeamFeatures:
        """Get default features for a team"""
        
        return TeamFeatures(
            team_name=team,
            venue=venue,
            avg_goals_scored=1.5,
            avg_goals_conceded=1.5,
            avg_shots=12.0,
            avg_shots_target=4.0,
            avg_shots_conceded=12.0,
            avg_shots_target_conceded=4.0,
            avg_corners=5.0,
            avg_corners_conceded=5.0,
            avg_fouls=12.0,
            avg_fouls_conceded=12.0,
            avg_yellow_cards=2.0,
            avg_red_cards=0.1,
            form_points=0.0,
            win_percentage=33.3,
            recent_form="",
            home_advantage=0.3 if venue == 'home' else -0.3,
            venue_form_points=0.0,
            h2h_goals_for=1.5,
            h2h_goals_against=1.5,
            h2h_win_percentage=33.3,
            attack_strength=1.0,
            defense_strength=1.0,
            shot_conversion_rate=0.125,
            shot_accuracy=0.33,
            discipline_index=2.3,
            referee_adjusted_cards=2.0,
            referee_adjusted_fouls=12.0
        )
    
    def create_feature_matrix(
        self,
        matches: List[Tuple[str, str, datetime, Optional[str]]],
        historical_data: pd.DataFrame,
        referee_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Create feature matrix for multiple matches
        
        Args:
            matches: List of (home_team, away_team, match_date, referee) tuples
            historical_data: Historical match data
            referee_data: Referee statistics data
        
        Returns:
            DataFrame with engineered features
        """
        
        features_list = []
        
        for home_team, away_team, match_date, referee in matches:
            home_features, away_features = self.engineer_match_features(
                home_team, away_team, match_date, referee, historical_data, referee_data
            )
            
            # Combine features into a single row
            feature_row = {
                'home_team': home_team,
                'away_team': away_team,
                'match_date': match_date,
                'referee': referee
            }
            
            # Add home team features with 'home_' prefix
            for field_name, value in home_features.__dict__.items():
                if field_name not in ['team_name', 'venue']:
                    feature_row[f'home_{field_name}'] = value
            
            # Add away team features with 'away_' prefix
            for field_name, value in away_features.__dict__.items():
                if field_name not in ['team_name', 'venue']:
                    feature_row[f'away_{field_name}'] = value
            
            features_list.append(feature_row)
        
        return pd.DataFrame(features_list)