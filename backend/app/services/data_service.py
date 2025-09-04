from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc
import pandas as pd
import numpy as np
from pathlib import Path
import csv
from io import StringIO

from app.models.models import Match, Team, TeamStats, RefereeStats
from app.models.schemas import MatchCreate, TeamCreate
from app.core.config import settings
from app.core.logging import get_logger
from app.core.validation import validate_csv_source, validate_artifacts_integrity, get_approved_data_source

logger = get_logger("data_service")

class DataService:
    """Service for data ingestion and processing"""
    
    def __init__(self):
        self.team_name_mappings = {
            # Common team name variations
            "real madrid": "Real Madrid",
            "real madrid cf": "Real Madrid",
            "barcelona": "Barcelona",
            "fc barcelona": "Barcelona",
            "atletico madrid": "Atletico Madrid",
            "atletico de madrid": "Atletico Madrid",
            "athletic bilbao": "Athletic Bilbao",
            "athletic club": "Athletic Bilbao",
            "real sociedad": "Real Sociedad",
            "real sociedad de futbol": "Real Sociedad",
            "valencia": "Valencia",
            "valencia cf": "Valencia",
            "sevilla": "Sevilla",
            "sevilla fc": "Sevilla",
            "villarreal": "Villarreal",
            "villarreal cf": "Villarreal",
            "real betis": "Real Betis",
            "betis": "Real Betis",
            "celta vigo": "Celta Vigo",
            "rc celta": "Celta Vigo",
            "espanyol": "Espanyol",
            "rcd espanyol": "Espanyol",
            "getafe": "Getafe",
            "getafe cf": "Getafe",
            "granada": "Granada",
            "granada cf": "Granada",
            "levante": "Levante",
            "levante ud": "Levante",
            "mallorca": "Mallorca",
            "rcd mallorca": "Mallorca",
            "osasuna": "Osasuna",
            "ca osasuna": "Osasuna",
            "alaves": "Alaves",
            "deportivo alaves": "Alaves",
            "cadiz": "Cadiz",
            "cadiz cf": "Cadiz",
            "elche": "Elche",
            "elche cf": "Elche",
            "rayo vallecano": "Rayo Vallecano",
            "rayo": "Rayo Vallecano"
        }
    
    async def ingest_csv_data(
        self,
        csv_content: str,
        season: str,
        db: Session,
        source_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Ingest match data from CSV content with source validation"""
        
        try:
            # Validate data source
            source_info = source_info or {'source_type': settings.DATA_SOURCE_TYPE}
            is_valid_source, source_errors = validate_csv_source(csv_content, source_info)
            
            if not is_valid_source:
                raise ValueError(f"Invalid data source: {source_errors}")
                
            # Validate artifacts integrity
            artifacts_valid, artifacts_errors = validate_artifacts_integrity()
            if not artifacts_valid:
                logger.warning(f"Artifacts integrity issues: {artifacts_errors}")
            
            # Parse CSV
            df = pd.read_csv(StringIO(csv_content))
            
            # Validate required columns
            required_columns = [
                'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
                'HTHG', 'HTAG', 'HTR', 'Referee'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Clean and process data
            processed_matches = await self._process_match_data(df, season)
            
            # Insert data into database
            ingestion_stats = await self._insert_matches(processed_matches, db)
            
            # Update team and referee statistics
            await self._update_team_stats(season, db)
            await self._update_referee_stats(season, db)
            
            logger.info(f"Successfully ingested {ingestion_stats['matches_inserted']} matches for season {season}")
            
            return {
                "season": season,
                "matches_processed": len(processed_matches),
                "matches_inserted": ingestion_stats['matches_inserted'],
                "matches_updated": ingestion_stats['matches_updated'],
                "teams_created": ingestion_stats['teams_created'],
                "processing_time": ingestion_stats['processing_time']
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest CSV data: {e}")
            raise
    
    async def _process_match_data(
        self,
        df: pd.DataFrame,
        season: str
    ) -> List[Dict[str, Any]]:
        """Process and clean match data from DataFrame"""
        
        processed_matches = []
        
        for _, row in df.iterrows():
            try:
                # Parse date
                match_date = pd.to_datetime(row['Date'], dayfirst=True)
                
                # Clean team names
                home_team = self._clean_team_name(row['HomeTeam'])
                away_team = self._clean_team_name(row['AwayTeam'])
                
                # Extract match data
                match_data = {
                    'date': match_date,
                    'season': season,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_goals': int(row['FTHG']) if pd.notna(row['FTHG']) else None,
                    'away_goals': int(row['FTAG']) if pd.notna(row['FTAG']) else None,
                    'ht_home_goals': int(row['HTHG']) if pd.notna(row['HTHG']) else None,
                    'ht_away_goals': int(row['HTAG']) if pd.notna(row['HTAG']) else None,
                    'result': row['FTR'] if pd.notna(row['FTR']) else None,
                    'ht_result': row['HTR'] if pd.notna(row['HTR']) else None,
                    'referee': row['Referee'] if pd.notna(row['Referee']) else None,
                    
                    # Shot statistics
                    'home_shots': int(row['HS']) if 'HS' in row and pd.notna(row['HS']) else None,
                    'away_shots': int(row['AS']) if 'AS' in row and pd.notna(row['AS']) else None,
                    'home_shots_target': int(row['HST']) if 'HST' in row and pd.notna(row['HST']) else None,
                    'away_shots_target': int(row['AST']) if 'AST' in row and pd.notna(row['AST']) else None,
                    
                    # Fouls and cards
                    'home_fouls': int(row['HF']) if 'HF' in row and pd.notna(row['HF']) else None,
                    'away_fouls': int(row['AF']) if 'AF' in row and pd.notna(row['AF']) else None,
                    'home_yellow_cards': int(row['HY']) if 'HY' in row and pd.notna(row['HY']) else None,
                    'away_yellow_cards': int(row['AY']) if 'AY' in row and pd.notna(row['AY']) else None,
                    'home_red_cards': int(row['HR']) if 'HR' in row and pd.notna(row['HR']) else None,
                    'away_red_cards': int(row['AR']) if 'AR' in row and pd.notna(row['AR']) else None,
                    
                    # Corners
                    'home_corners': int(row['HC']) if 'HC' in row and pd.notna(row['HC']) else None,
                    'away_corners': int(row['AC']) if 'AC' in row and pd.notna(row['AC']) else None
                }
                
                processed_matches.append(match_data)
                
            except Exception as e:
                logger.warning(f"Failed to process match row: {e}")
                continue
        
        return processed_matches
    
    def _clean_team_name(self, team_name: str) -> str:
        """Clean and standardize team names"""
        
        if pd.isna(team_name):
            return "Unknown"
        
        # Convert to lowercase for mapping
        clean_name = str(team_name).strip().lower()
        
        # Check mappings
        if clean_name in self.team_name_mappings:
            return self.team_name_mappings[clean_name]
        
        # Return title case if no mapping found
        return str(team_name).strip().title()
    
    async def _insert_matches(
        self,
        processed_matches: List[Dict[str, Any]],
        db: Session
    ) -> Dict[str, Any]:
        """Insert processed matches into database"""
        
        start_time = datetime.utcnow()
        matches_inserted = 0
        matches_updated = 0
        teams_created = 0
        
        # Get or create teams
        team_cache = {}
        
        for match_data in processed_matches:
            try:
                # Get or create home team
                home_team = await self._get_or_create_team(
                    match_data['home_team'], team_cache, db
                )
                if home_team['created']:
                    teams_created += 1
                
                # Get or create away team
                away_team = await self._get_or_create_team(
                    match_data['away_team'], team_cache, db
                )
                if away_team['created']:
                    teams_created += 1
                
                # Check if match already exists
                existing_match = db.query(Match).filter(
                    and_(
                        Match.date == match_data['date'],
                        Match.home_team_id == home_team['team'].id,
                        Match.away_team_id == away_team['team'].id
                    )
                ).first()
                
                if existing_match:
                    # Update existing match
                    for key, value in match_data.items():
                        if key not in ['home_team', 'away_team', 'date']:
                            setattr(existing_match, key, value)
                    
                    existing_match.updated_at = datetime.utcnow()
                    matches_updated += 1
                else:
                    # Create new match
                    new_match = Match(
                        date=match_data['date'],
                        season=match_data['season'],
                        home_team_id=home_team['team'].id,
                        away_team_id=away_team['team'].id,
                        home_goals=match_data['home_goals'],
                        away_goals=match_data['away_goals'],
                        ht_home_goals=match_data['ht_home_goals'],
                        ht_away_goals=match_data['ht_away_goals'],
                        result=match_data['result'],
                        ht_result=match_data['ht_result'],
                        referee=match_data['referee'],
                        home_shots=match_data['home_shots'],
                        away_shots=match_data['away_shots'],
                        home_shots_target=match_data['home_shots_target'],
                        away_shots_target=match_data['away_shots_target'],
                        home_fouls=match_data['home_fouls'],
                        away_fouls=match_data['away_fouls'],
                        home_yellow_cards=match_data['home_yellow_cards'],
                        away_yellow_cards=match_data['away_yellow_cards'],
                        home_red_cards=match_data['home_red_cards'],
                        away_red_cards=match_data['away_red_cards'],
                        home_corners=match_data['home_corners'],
                        away_corners=match_data['away_corners'],
                        is_completed=True
                    )
                    
                    db.add(new_match)
                    matches_inserted += 1
                
            except Exception as e:
                logger.error(f"Failed to insert match: {e}")
                continue
        
        # Commit all changes
        db.commit()
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            'matches_inserted': matches_inserted,
            'matches_updated': matches_updated,
            'teams_created': teams_created,
            'processing_time': processing_time
        }
    
    async def _get_or_create_team(
        self,
        team_name: str,
        team_cache: Dict[str, Dict],
        db: Session
    ) -> Dict[str, Any]:
        """Get existing team or create new one"""
        
        if team_name in team_cache:
            return team_cache[team_name]
        
        # Check if team exists
        existing_team = db.query(Team).filter(
            Team.name == team_name
        ).first()
        
        if existing_team:
            result = {'team': existing_team, 'created': False}
        else:
            # Create new team
            new_team = Team(
                name=team_name,
                normalized_name=team_name.lower().replace(' ', '_'),
                founded_year=None,  # Would need additional data
                stadium=None,   # Would need additional data
                city=None    # Would need additional data
            )
            
            db.add(new_team)
            db.flush()  # Get the ID without committing
            
            result = {'team': new_team, 'created': True}
        
        # Cache the result
        team_cache[team_name] = result
        
        return result
    
    async def _update_team_stats(
        self,
        season: str,
        db: Session
    ) -> None:
        """Update team statistics for the season"""
        
        try:
            # Get all teams
            teams = db.query(Team).all()
            
            for team in teams:
                # Calculate home stats
                home_matches = db.query(Match).filter(
                    and_(
                        Match.home_team_id == team.id,
                        Match.season == season,
                        Match.home_goals.isnot(None)
                    )
                ).all()
                
                # Calculate away stats
                away_matches = db.query(Match).filter(
                    and_(
                        Match.away_team_id == team.id,
                        Match.season == season,
                        Match.away_goals.isnot(None)
                    )
                ).all()
                
                # Calculate statistics
                stats = await self._calculate_team_season_stats(
                    team, home_matches, away_matches
                )
                
                # Update or create team stats
                existing_stats = db.query(TeamStats).filter(
                    and_(
                        TeamStats.team_id == team.id,
                        TeamStats.season == season
                    )
                ).first()
                
                if existing_stats:
                    # Update existing stats
                    for key, value in stats.items():
                        setattr(existing_stats, key, value)
                    existing_stats.updated_at = datetime.utcnow()
                else:
                    # Create new stats
                    new_stats = TeamStats(
                        team_id=team.id,
                        season=season,
                        **stats
                    )
                    db.add(new_stats)
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to update team stats: {e}")
            db.rollback()
    
    async def _calculate_team_season_stats(
        self,
        team: Team,
        home_matches: List[Match],
        away_matches: List[Match]
    ) -> Dict[str, Any]:
        """Calculate comprehensive team statistics for a season"""
        
        total_matches = len(home_matches) + len(away_matches)
        
        if total_matches == 0:
            return self._get_empty_stats()
        
        # Goals
        goals_for = sum(m.home_goals or 0 for m in home_matches) + sum(m.away_goals or 0 for m in away_matches)
        goals_against = sum(m.away_goals or 0 for m in home_matches) + sum(m.home_goals or 0 for m in away_matches)
        
        # Wins, draws, losses
        wins = len([m for m in home_matches if (m.home_goals or 0) > (m.away_goals or 0)]) + \
               len([m for m in away_matches if (m.away_goals or 0) > (m.home_goals or 0)])
        
        draws = len([m for m in home_matches if (m.home_goals or 0) == (m.away_goals or 0)]) + \
                len([m for m in away_matches if (m.away_goals or 0) == (m.home_goals or 0)])
        
        losses = total_matches - wins - draws
        
        # Points
        points = wins * 3 + draws
        
        # Shots
        shots_for = sum(m.home_shots or 0 for m in home_matches) + sum(m.away_shots or 0 for m in away_matches)
        shots_against = sum(m.away_shots or 0 for m in home_matches) + sum(m.home_shots or 0 for m in away_matches)
        
        shots_target_for = sum(m.home_shots_target or 0 for m in home_matches) + sum(m.away_shots_target or 0 for m in away_matches)
        shots_target_against = sum(m.away_shots_target or 0 for m in home_matches) + sum(m.home_shots_target or 0 for m in away_matches)
        
        # Cards and fouls
        yellow_cards = sum(m.home_yellow_cards or 0 for m in home_matches) + sum(m.away_yellow_cards or 0 for m in away_matches)
        red_cards = sum(m.home_red_cards or 0 for m in home_matches) + sum(m.away_red_cards or 0 for m in away_matches)
        fouls = sum(m.home_fouls or 0 for m in home_matches) + sum(m.away_fouls or 0 for m in away_matches)
        
        # Corners
        corners_for = sum(m.home_corners or 0 for m in home_matches) + sum(m.away_corners or 0 for m in away_matches)
        corners_against = sum(m.away_corners or 0 for m in home_matches) + sum(m.home_corners or 0 for m in away_matches)
        
        return {
            'matches_played': total_matches,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'points': points,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'goal_difference': goals_for - goals_against,
            'shots_for': shots_for,
            'shots_against': shots_against,
            'shots_target_for': shots_target_for,
            'shots_target_against': shots_target_against,
            'yellow_cards': yellow_cards,
            'red_cards': red_cards,
            'fouls': fouls,
            'corners_for': corners_for,
            'corners_against': corners_against,
            
            # Per-game averages
            'avg_goals_for': goals_for / total_matches,
            'avg_goals_against': goals_against / total_matches,
            'avg_shots_for': shots_for / total_matches if shots_for else 0,
            'avg_shots_against': shots_against / total_matches if shots_against else 0,
            'avg_corners_for': corners_for / total_matches if corners_for else 0,
            'avg_corners_against': corners_against / total_matches if corners_against else 0,
            
            # Home/away splits
            'home_matches': len(home_matches),
            'away_matches': len(away_matches),
            'home_wins': len([m for m in home_matches if (m.home_goals or 0) > (m.away_goals or 0)]),
            'away_wins': len([m for m in away_matches if (m.away_goals or 0) > (m.home_goals or 0)])
        }
    
    def _get_empty_stats(self) -> Dict[str, Any]:
        """Return empty statistics dictionary"""
        
        return {
            'window_size': 38,  # Full season
            'as_of_date': datetime.utcnow(),
            'venue': 'all',
            'matches_played': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_for': 0.0,
            'goals_against': 0.0,
            'goals_for_per_90': 0.0,
            'goals_against_per_90': 0.0,
            'shots_per_90': 0.0,
            'shots_target_per_90': 0.0,
            'shots_allowed_per_90': 0.0,
            'shots_target_allowed_per_90': 0.0,
            'fouls_per_90': 0.0,
            'yellow_cards_per_90': 0.0,
            'red_cards_per_90': 0.0,
            'corners_per_90': 0.0,
            'corners_allowed_per_90': 0.0,
            'points_per_game': 0.0,
            'form_score': 0.0
        }
    
    async def _update_referee_stats(
        self,
        season: str,
        db: Session
    ) -> None:
        """Update referee statistics for the season"""
        
        try:
            # Get all referees for the season
            referees = db.query(Match.referee).filter(
                and_(
                    Match.season == season,
                    Match.referee.isnot(None)
                )
            ).distinct().all()
            
            for (referee_name,) in referees:
                if not referee_name:
                    continue
                
                # Get all matches for this referee
                referee_matches = db.query(Match).filter(
                    and_(
                        Match.referee == referee_name,
                        Match.season == season,
                        Match.home_goals.isnot(None)
                    )
                ).all()
                
                if not referee_matches:
                    continue
                
                # Calculate referee stats
                stats = await self._calculate_referee_season_stats(referee_matches)
                
                # Update or create referee stats
                existing_stats = db.query(RefereeStats).filter(
                    and_(
                        RefereeStats.referee_name == referee_name,
                        RefereeStats.season == season
                    )
                ).first()
                
                if existing_stats:
                    # Update existing stats
                    for key, value in stats.items():
                        setattr(existing_stats, key, value)
                    existing_stats.updated_at = datetime.utcnow()
                else:
                    # Create new stats
                    new_stats = RefereeStats(
                        referee_name=referee_name,
                        season=season,
                        **stats
                    )
                    db.add(new_stats)
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to update referee stats: {e}")
            db.rollback()
    
    async def _calculate_referee_season_stats(
        self,
        matches: List[Match]
    ) -> Dict[str, Any]:
        """Calculate referee statistics for a season"""
        
        total_matches = len(matches)
        
        if total_matches == 0:
            return {
                'matches_officiated': 0,
                'avg_goals_per_match': 0.0,
                'avg_yellow_cards_per_match': 0.0,
                'avg_red_cards_per_match': 0.0,
                'avg_fouls_per_match': 0.0,
                'home_win_percentage': 0.0,
                'away_win_percentage': 0.0,
                'draw_percentage': 0.0
            }
        
        # Goals
        total_goals = sum((m.home_goals or 0) + (m.away_goals or 0) for m in matches)
        
        # Cards and fouls
        total_yellow_cards = sum((m.home_yellow_cards or 0) + (m.away_yellow_cards or 0) for m in matches)
        total_red_cards = sum((m.home_red_cards or 0) + (m.away_red_cards or 0) for m in matches)
        total_fouls = sum((m.home_fouls or 0) + (m.away_fouls or 0) for m in matches)
        
        # Results
        home_wins = len([m for m in matches if (m.home_goals or 0) > (m.away_goals or 0)])
        away_wins = len([m for m in matches if (m.away_goals or 0) > (m.home_goals or 0)])
        draws = len([m for m in matches if (m.home_goals or 0) == (m.away_goals or 0)])
        
        return {
            'matches_officiated': total_matches,
            'avg_goals_per_match': total_goals / total_matches,
            'avg_yellow_cards_per_match': total_yellow_cards / total_matches,
            'avg_red_cards_per_match': total_red_cards / total_matches,
            'avg_fouls_per_match': total_fouls / total_matches,
            'home_win_percentage': (home_wins / total_matches) * 100,
            'away_win_percentage': (away_wins / total_matches) * 100,
            'draw_percentage': (draws / total_matches) * 100
        }
    
    async def get_data_summary(
        self,
        db: Session
    ) -> Dict[str, Any]:
        """Get summary of available data"""
        
        try:
            # Count matches by season
            season_counts = db.query(
                Match.season,
                func.count(Match.id).label('match_count')
            ).group_by(Match.season).all()
            
            # Count teams
            total_teams = db.query(func.count(Team.id)).scalar()
            
            # Count total matches
            total_matches = db.query(func.count(Match.id)).scalar()
            
            # Get date range
            date_range = db.query(
                func.min(Match.date).label('earliest'),
                func.max(Match.date).label('latest')
            ).first()
            
            # Count referees
            total_referees = db.query(
                func.count(func.distinct(Match.referee))
            ).filter(Match.referee.isnot(None)).scalar()
            
            # Get data source information
            data_source_info = get_approved_data_source()
            artifacts_valid, artifacts_errors = validate_artifacts_integrity()
            
            return {
                'total_matches': total_matches,
                'total_teams': total_teams,
                'total_referees': total_referees,
                'seasons': {
                    season: count for season, count in season_counts
                },
                'date_range': {
                    'earliest': date_range.earliest,
                    'latest': date_range.latest
                },
                'data_source': data_source_info,
                'artifacts_status': {
                    'valid': artifacts_valid,
                    'errors': artifacts_errors if not artifacts_valid else []
                },
                'last_updated': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to get data summary: {e}")
            raise