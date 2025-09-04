import pandas as pd
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from app.core.logging import get_logger

logger = get_logger("team_analytics")

class TeamAnalyticsService:
    def __init__(self):
        self.artifacts_path = "/Users/boksi/Developer/laliga_pipeline/artifacts"
        self._team_form_df = None
        self._matches_features_df = None
        self._matches_clean_df = None
        
    def _load_team_form_data(self) -> pd.DataFrame:
        """Load team form data from CSV."""
        if self._team_form_df is None:
            try:
                file_path = os.path.join(self.artifacts_path, "team_form.csv")
                self._team_form_df = pd.read_csv(file_path)
                logger.info(f"Loaded team form data: {len(self._team_form_df)} records")
            except Exception as e:
                logger.error(f"Error loading team form data: {e}")
                self._team_form_df = pd.DataFrame()
        return self._team_form_df
    
    def _load_matches_features_data(self) -> pd.DataFrame:
        """Load matches features data from CSV."""
        if self._matches_features_df is None:
            try:
                file_path = os.path.join(self.artifacts_path, "matches_features.csv")
                self._matches_features_df = pd.read_csv(file_path)
                logger.info(f"Loaded matches features data: {len(self._matches_features_df)} records")
            except Exception as e:
                logger.error(f"Error loading matches features data: {e}")
                self._matches_features_df = pd.DataFrame()
        return self._matches_features_df
    
    def _load_matches_clean_data(self) -> pd.DataFrame:
        """Load clean matches data from CSV."""
        if self._matches_clean_df is None:
            try:
                file_path = os.path.join(self.artifacts_path, "matches_clean.csv")
                self._matches_clean_df = pd.read_csv(file_path)
                logger.info(f"Loaded clean matches data: {len(self._matches_clean_df)} records")
            except Exception as e:
                logger.error(f"Error loading clean matches data: {e}")
                self._matches_clean_df = pd.DataFrame()
        return self._matches_clean_df
    
    def get_team_analytics(self, team_name: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a specific team."""
        try:
            # Load data
            team_form_df = self._load_team_form_data()
            matches_features_df = self._load_matches_features_data()
            matches_clean_df = self._load_matches_clean_data()
            
            # Normalize team name for matching
            team_name_lower = team_name.lower().strip()
            
            analytics = {
                "team_name": team_name,
                "form_metrics": self._get_team_form_metrics(team_form_df, team_name_lower),
                "match_statistics": self._get_team_match_statistics(matches_clean_df, team_name_lower),
                "performance_trends": self._get_team_performance_trends(matches_features_df, team_name_lower),
                "recent_form": self._get_recent_form(team_form_df, team_name_lower),
                "head_to_head": self._get_head_to_head_summary(matches_clean_df, team_name_lower)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting team analytics for {team_name}: {e}")
            return {"error": f"Failed to get analytics for {team_name}"}
    
    def _get_team_form_metrics(self, df: pd.DataFrame, team_name: str) -> Dict[str, Any]:
        """Extract team form metrics from team_form.csv."""
        if df.empty:
            return {}
            
        team_data = df[df['team'].str.lower() == team_name]
        
        if team_data.empty:
            return {"message": "No form data available"}
        
        # Get latest form metrics
        latest_data = team_data.dropna().tail(10)  # Last 10 records with data
        
        if latest_data.empty:
            return {"message": "No recent form data available"}
        
        metrics = {
            "goals_for_avg_5": float(latest_data['gf_r5'].mean()) if 'gf_r5' in latest_data.columns else 0,
            "goals_against_avg_5": float(latest_data['ga_r5'].mean()) if 'ga_r5' in latest_data.columns else 0,
            "goals_for_avg_10": float(latest_data['gf_r10'].mean()) if 'gf_r10' in latest_data.columns else 0,
            "goals_against_avg_10": float(latest_data['ga_r10'].mean()) if 'ga_r10' in latest_data.columns else 0,
            "home_ratio_5": float(latest_data['home_r5'].mean()) if 'home_r5' in latest_data.columns else 0,
            "home_ratio_10": float(latest_data['home_r10'].mean()) if 'home_r10' in latest_data.columns else 0,
            "total_records": len(team_data),
            "data_range": {
                "from": team_data['Date'].min() if 'Date' in team_data.columns else None,
                "to": team_data['Date'].max() if 'Date' in team_data.columns else None
            }
        }
        
        return metrics
    
    def _get_team_match_statistics(self, df: pd.DataFrame, team_name: str) -> Dict[str, Any]:
        """Extract match statistics from matches_clean.csv."""
        if df.empty:
            return {}
        
        # Get matches where team played (home or away)
        home_matches = df[df['HomeTeam'].str.lower() == team_name]
        away_matches = df[df['AwayTeam'].str.lower() == team_name]
        
        total_matches = len(home_matches) + len(away_matches)
        
        if total_matches == 0:
            return {"message": "No match data available"}
        
        # Calculate wins, draws, losses
        home_wins = len(home_matches[home_matches['FTR'] == 'H'])
        home_draws = len(home_matches[home_matches['FTR'] == 'D'])
        home_losses = len(home_matches[home_matches['FTR'] == 'A'])
        
        away_wins = len(away_matches[away_matches['FTR'] == 'A'])
        away_draws = len(away_matches[away_matches['FTR'] == 'D'])
        away_losses = len(away_matches[away_matches['FTR'] == 'H'])
        
        total_wins = home_wins + away_wins
        total_draws = home_draws + away_draws
        total_losses = home_losses + away_losses
        
        # Calculate goals
        home_goals_for = home_matches['FTHG'].sum() if 'FTHG' in home_matches.columns else 0
        home_goals_against = home_matches['FTAG'].sum() if 'FTAG' in home_matches.columns else 0
        away_goals_for = away_matches['FTAG'].sum() if 'FTAG' in away_matches.columns else 0
        away_goals_against = away_matches['FTHG'].sum() if 'FTHG' in away_matches.columns else 0
        
        total_goals_for = home_goals_for + away_goals_for
        total_goals_against = home_goals_against + away_goals_against
        
        statistics = {
            "total_matches": total_matches,
            "wins": total_wins,
            "draws": total_draws,
            "losses": total_losses,
            "win_percentage": round((total_wins / total_matches) * 100, 2) if total_matches > 0 else 0,
            "goals_for": int(total_goals_for),
            "goals_against": int(total_goals_against),
            "goal_difference": int(total_goals_for - total_goals_against),
            "goals_per_match": round(total_goals_for / total_matches, 2) if total_matches > 0 else 0,
            "home_record": {
                "matches": len(home_matches),
                "wins": home_wins,
                "draws": home_draws,
                "losses": home_losses,
                "goals_for": int(home_goals_for),
                "goals_against": int(home_goals_against)
            },
            "away_record": {
                "matches": len(away_matches),
                "wins": away_wins,
                "draws": away_draws,
                "losses": away_losses,
                "goals_for": int(away_goals_for),
                "goals_against": int(away_goals_against)
            }
        }
        
        return statistics
    
    def _get_team_performance_trends(self, df: pd.DataFrame, team_name: str) -> Dict[str, Any]:
        """Extract performance trends from matches_features.csv."""
        if df.empty:
            return {}
        
        # Get matches where team played
        home_matches = df[df['HomeTeam'].str.lower() == team_name]
        away_matches = df[df['AwayTeam'].str.lower() == team_name]
        
        if home_matches.empty and away_matches.empty:
            return {"message": "No performance trend data available"}
        
        trends = {
            "home_trends": self._calculate_home_trends(home_matches),
            "away_trends": self._calculate_away_trends(away_matches)
        }
        
        return trends
    
    def _calculate_home_trends(self, matches: pd.DataFrame) -> Dict[str, Any]:
        """Calculate home performance trends."""
        if matches.empty:
            return {}
        
        # Get recent matches (last 10)
        recent_matches = matches.tail(10)
        
        trends = {
            "recent_form_5_games": {
                "avg_goals_for": float(recent_matches['H_gf_r5'].mean()) if 'H_gf_r5' in recent_matches.columns else 0,
                "avg_goals_against": float(recent_matches['H_ga_r5'].mean()) if 'H_ga_r5' in recent_matches.columns else 0
            },
            "recent_form_10_games": {
                "avg_goals_for": float(recent_matches['H_gf_r10'].mean()) if 'H_gf_r10' in recent_matches.columns else 0,
                "avg_goals_against": float(recent_matches['H_ga_r10'].mean()) if 'H_ga_r10' in recent_matches.columns else 0
            }
        }
        
        return trends
    
    def _calculate_away_trends(self, matches: pd.DataFrame) -> Dict[str, Any]:
        """Calculate away performance trends."""
        if matches.empty:
            return {}
        
        # Get recent matches (last 10)
        recent_matches = matches.tail(10)
        
        trends = {
            "recent_form_5_games": {
                "avg_goals_for": float(recent_matches['A_gf_r5'].mean()) if 'A_gf_r5' in recent_matches.columns else 0,
                "avg_goals_against": float(recent_matches['A_ga_r5'].mean()) if 'A_ga_r5' in recent_matches.columns else 0
            },
            "recent_form_10_games": {
                "avg_goals_for": float(recent_matches['A_gf_r10'].mean()) if 'A_gf_r10' in recent_matches.columns else 0,
                "avg_goals_against": float(recent_matches['A_ga_r10'].mean()) if 'A_ga_r10' in recent_matches.columns else 0
            }
        }
        
        return trends
    
    def _get_recent_form(self, df: pd.DataFrame, team_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent form data for the team."""
        if df.empty:
            return []
        
        team_data = df[df['team'].str.lower() == team_name]
        recent_data = team_data.dropna().tail(limit)
        
        form_list = []
        for _, row in recent_data.iterrows():
            form_entry = {
                "date": row.get('Date'),
                "goals_for_r5": float(row.get('gf_r5', 0)) if pd.notna(row.get('gf_r5')) else 0,
                "goals_against_r5": float(row.get('ga_r5', 0)) if pd.notna(row.get('ga_r5')) else 0,
                "home_ratio_r5": float(row.get('home_r5', 0)) if pd.notna(row.get('home_r5')) else 0
            }
            form_list.append(form_entry)
        
        return form_list
    
    def _get_head_to_head_summary(self, df: pd.DataFrame, team_name: str) -> Dict[str, Any]:
        """Get head-to-head summary against other teams."""
        if df.empty:
            return {}
        
        # Get all opponents
        home_matches = df[df['HomeTeam'].str.lower() == team_name]
        away_matches = df[df['AwayTeam'].str.lower() == team_name]
        
        opponents = set()
        if not home_matches.empty:
            opponents.update(home_matches['AwayTeam'].str.lower().unique())
        if not away_matches.empty:
            opponents.update(away_matches['HomeTeam'].str.lower().unique())
        
        h2h_summary = {
            "total_opponents": len(opponents),
            "most_played_against": self._get_most_played_opponent(df, team_name)
        }
        
        return h2h_summary
    
    def _get_most_played_opponent(self, df: pd.DataFrame, team_name: str) -> Dict[str, Any]:
        """Get the opponent played against most frequently."""
        home_matches = df[df['HomeTeam'].str.lower() == team_name]
        away_matches = df[df['AwayTeam'].str.lower() == team_name]
        
        opponent_counts = {}
        
        # Count home matches
        for opponent in home_matches['AwayTeam'].str.lower():
            opponent_counts[opponent] = opponent_counts.get(opponent, 0) + 1
        
        # Count away matches
        for opponent in away_matches['HomeTeam'].str.lower():
            opponent_counts[opponent] = opponent_counts.get(opponent, 0) + 1
        
        if not opponent_counts:
            return {"opponent": None, "matches": 0}
        
        most_played = max(opponent_counts.items(), key=lambda x: x[1])
        return {"opponent": most_played[0], "matches": most_played[1]}