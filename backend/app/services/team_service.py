from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc

from app.models.models import Team, Match, TeamStats
from app.models.schemas import (
    TeamResponse, TeamStatsResponse, TeamFormResponse, 
    MatchResponse, MatchResult
)
from app.core.logging import get_logger

logger = get_logger("team_service")

class TeamService:
    """Service for team-related operations"""
    
    def __init__(self):
        pass
    
    async def get_teams(
        self,
        season: Optional[str] = None,
        active_only: bool = True,
        db: Session = None
    ) -> List[TeamResponse]:
        """Get all teams with optional filtering"""
        
        query = db.query(Team)
        
        if active_only:
            query = query.filter(Team.is_active == True)
        
        if season:
            # Filter teams that played in the specified season
            query = query.join(Match, or_(
                Match.home_team_id == Team.id,
                Match.away_team_id == Team.id
            )).filter(Match.season == season).distinct()
        
        teams = query.order_by(Team.name).all()
        
        result = []
        for team in teams:
            result.append(TeamResponse(
                id=team.id,
                name=team.name,
                normalized_name=team.normalized_name,
                founded_year=team.founded_year,
                stadium=team.stadium,
                city=team.city,
                is_active=team.is_active,
                created_at=team.created_at,
                updated_at=team.updated_at
            ))
        
        return result
    
    async def get_team_by_id(self, team_id: int, db: Session) -> Optional[TeamResponse]:
        """Get team by ID"""
        
        team = db.query(Team).filter(Team.id == team_id).first()
        
        if not team:
            return None
        
        return TeamResponse(
            id=team.id,
            name=team.name,
            normalized_name=team.normalized_name,
            founded_year=team.founded_year,
            stadium=team.stadium,
            city=team.city,
            is_active=team.is_active,
            created_at=team.created_at,
            updated_at=team.updated_at
        )
    
    async def get_team_by_name(self, team_name: str, db: Session) -> Optional[TeamResponse]:
        """Get team by name with fuzzy matching"""
        
        # Try exact match first
        team = db.query(Team).filter(
            func.lower(Team.name) == func.lower(team_name)
        ).first()
        
        if not team:
            # Try partial match
            team = db.query(Team).filter(
                Team.name.ilike(f"%{team_name}%")
            ).first()
        
        if not team:
            # Try normalized name match
            team = db.query(Team).filter(
                func.lower(Team.normalized_name) == func.lower(team_name)
            ).first()
        
        if not team:
            return None
        
        return TeamResponse(
            id=team.id,
            name=team.name,
            short_name=team.short_name,
            founded=team.founded,
            stadium=team.stadium,
            city=team.city,
            country=team.country,
            logo_url=team.logo_url,
            is_active=team.is_active,
            created_at=team.created_at,
            updated_at=team.updated_at
        )
    
    async def get_team_stats(
        self,
        team_id: int,
        season: Optional[str] = None,
        venue: Optional[str] = None,
        last_n_matches: Optional[int] = None,
        db: Session = None
    ) -> TeamStatsResponse:
        """Get comprehensive team statistics"""
        
        # Base query for matches
        query = db.query(Match).filter(
            or_(
                Match.home_team_id == team_id,
                Match.away_team_id == team_id
            )
        )
        
        # Apply filters
        if season:
            query = query.filter(Match.season == season)
        
        if venue == "home":
            query = query.filter(Match.home_team_id == team_id)
        elif venue == "away":
            query = query.filter(Match.away_team_id == team_id)
        
        # Order by date and limit if specified
        query = query.order_by(desc(Match.date))
        if last_n_matches:
            query = query.limit(last_n_matches)
        
        matches = query.all()
        
        if not matches:
            # Return default stats if no matches found
            return TeamStatsResponse(
                team_id=team_id,
                matches_played=0,
                wins=0,
                draws=0,
                losses=0,
                goals_scored=0,
                goals_conceded=0,
                goal_difference=0,
                points=0,
                win_percentage=0.0,
                goals_per_match=0.0,
                goals_conceded_per_match=0.0,
                shots_per_match=0.0,
                shots_on_target_per_match=0.0,
                possession_percentage=0.0,
                pass_accuracy=0.0,
                fouls_per_match=0.0,
                cards_per_match=0.0,
                corners_per_match=0.0,
                home_record={"wins": 0, "draws": 0, "losses": 0},
                away_record={"wins": 0, "draws": 0, "losses": 0},
                recent_form=[],
                season=season,
                venue=venue,
                last_updated=datetime.utcnow()
            )
        
        # Calculate statistics
        stats = self._calculate_team_stats(team_id, matches)
        
        return TeamStatsResponse(
            team_id=team_id,
            matches_played=stats["matches_played"],
            wins=stats["wins"],
            draws=stats["draws"],
            losses=stats["losses"],
            goals_scored=stats["goals_scored"],
            goals_conceded=stats["goals_conceded"],
            goal_difference=stats["goal_difference"],
            points=stats["points"],
            win_percentage=stats["win_percentage"],
            goals_per_match=stats["goals_per_match"],
            goals_conceded_per_match=stats["goals_conceded_per_match"],
            shots_per_match=stats["shots_per_match"],
            shots_on_target_per_match=stats["shots_on_target_per_match"],
            possession_percentage=stats["possession_percentage"],
            pass_accuracy=stats["pass_accuracy"],
            fouls_per_match=stats["fouls_per_match"],
            cards_per_match=stats["cards_per_match"],
            corners_per_match=stats["corners_per_match"],
            home_record=stats["home_record"],
            away_record=stats["away_record"],
            recent_form=stats["recent_form"],
            season=season,
            venue=venue,
            last_updated=datetime.utcnow()
        )
    
    def _calculate_team_stats(self, team_id: int, matches: List[Match]) -> Dict[str, Any]:
        """Calculate team statistics from matches"""
        
        wins = draws = losses = 0
        goals_scored = goals_conceded = 0
        shots = shots_on_target = 0
        fouls = cards = corners = 0
        home_wins = home_draws = home_losses = 0
        away_wins = away_draws = away_losses = 0
        recent_form = []
        
        for match in matches:
            is_home = match.home_team_id == team_id
            
            if is_home:
                team_goals = match.home_goals or 0
                opponent_goals = match.away_goals or 0
                team_shots = match.home_shots or 0
                team_shots_on_target = match.home_shots_on_target or 0
                team_fouls = match.home_fouls or 0
                team_cards = (match.home_yellow_cards or 0) + (match.home_red_cards or 0)
                team_corners = match.home_corners or 0
            else:
                team_goals = match.away_goals or 0
                opponent_goals = match.home_goals or 0
                team_shots = match.away_shots or 0
                team_shots_on_target = match.away_shots_on_target or 0
                team_fouls = match.away_fouls or 0
                team_cards = (match.away_yellow_cards or 0) + (match.away_red_cards or 0)
                team_corners = match.away_corners or 0
            
            # Accumulate totals
            goals_scored += team_goals
            goals_conceded += opponent_goals
            shots += team_shots
            shots_on_target += team_shots_on_target
            fouls += team_fouls
            cards += team_cards
            corners += team_corners
            
            # Determine result
            if team_goals > opponent_goals:
                wins += 1
                if is_home:
                    home_wins += 1
                else:
                    away_wins += 1
                recent_form.append("W")
            elif team_goals == opponent_goals:
                draws += 1
                if is_home:
                    home_draws += 1
                else:
                    away_draws += 1
                recent_form.append("D")
            else:
                losses += 1
                if is_home:
                    home_losses += 1
                else:
                    away_losses += 1
                recent_form.append("L")
        
        matches_played = len(matches)
        points = wins * 3 + draws
        
        return {
            "matches_played": matches_played,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "goals_scored": goals_scored,
            "goals_conceded": goals_conceded,
            "goal_difference": goals_scored - goals_conceded,
            "points": points,
            "win_percentage": (wins / matches_played * 100) if matches_played > 0 else 0.0,
            "goals_per_match": goals_scored / matches_played if matches_played > 0 else 0.0,
            "goals_conceded_per_match": goals_conceded / matches_played if matches_played > 0 else 0.0,
            "shots_per_match": shots / matches_played if matches_played > 0 else 0.0,
            "shots_on_target_per_match": shots_on_target / matches_played if matches_played > 0 else 0.0,
            "possession_percentage": 50.0,  # Placeholder - would need actual possession data
            "pass_accuracy": 80.0,  # Placeholder - would need actual pass data
            "fouls_per_match": fouls / matches_played if matches_played > 0 else 0.0,
            "cards_per_match": cards / matches_played if matches_played > 0 else 0.0,
            "corners_per_match": corners / matches_played if matches_played > 0 else 0.0,
            "home_record": {"wins": home_wins, "draws": home_draws, "losses": home_losses},
            "away_record": {"wins": away_wins, "draws": away_draws, "losses": away_losses},
            "recent_form": recent_form[:10]  # Last 10 matches
        }
    
    async def get_team_form(
        self,
        team_id: int,
        matches: int = 10,
        venue: Optional[str] = None,
        db: Session = None
    ) -> TeamFormResponse:
        """Get team's recent form"""
        
        query = db.query(Match).filter(
            or_(
                Match.home_team_id == team_id,
                Match.away_team_id == team_id
            )
        )
        
        if venue == "home":
            query = query.filter(Match.home_team_id == team_id)
        elif venue == "away":
            query = query.filter(Match.away_team_id == team_id)
        
        recent_matches = query.order_by(
            desc(Match.date)
        ).limit(matches).all()
        
        form_data = []
        points = 0
        
        for match in recent_matches:
            is_home = match.home_team_id == team_id
            
            if is_home:
                team_goals = match.home_goals or 0
                opponent_goals = match.away_goals or 0
                opponent_name = match.away_team.name
            else:
                team_goals = match.away_goals or 0
                opponent_goals = match.home_goals or 0
                opponent_name = match.home_team.name
            
            if team_goals > opponent_goals:
                result = "W"
                match_points = 3
            elif team_goals == opponent_goals:
                result = "D"
                match_points = 1
            else:
                result = "L"
                match_points = 0
            
            points += match_points
            
            form_data.append({
                "date": match.date,
                "opponent": opponent_name,
                "venue": "home" if is_home else "away",
                "result": result,
                "score": f"{team_goals}-{opponent_goals}",
                "points": match_points
            })
        
        form_string = "".join([match["result"] for match in form_data])
        
        return TeamFormResponse(
            team_id=team_id,
            matches_analyzed=len(recent_matches),
            form_string=form_string,
            points=points,
            points_per_match=points / len(recent_matches) if recent_matches else 0.0,
            recent_matches=form_data,
            venue=venue
        )
    
    async def get_team_matches(
        self,
        team_id: int,
        season: Optional[str] = None,
        venue: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        db: Session = None
    ) -> List[MatchResponse]:
        """Get matches for a team"""
        
        query = db.query(Match).filter(
            or_(
                Match.home_team_id == team_id,
                Match.away_team_id == team_id
            )
        )
        
        if season:
            query = query.filter(Match.season == season)
        
        if venue == "home":
            query = query.filter(Match.home_team_id == team_id)
        elif venue == "away":
            query = query.filter(Match.away_team_id == team_id)
        
        matches = query.order_by(
            desc(Match.date)
        ).offset(offset).limit(limit).all()
        
        result = []
        for match in matches:
            # Determine result from team's perspective
            if match.home_goals is not None and match.away_goals is not None:
                if match.home_team_id == team_id:
                    if match.home_goals > match.away_goals:
                        result_enum = MatchResult.HOME_WIN
                    elif match.home_goals == match.away_goals:
                        result_enum = MatchResult.DRAW
                    else:
                        result_enum = MatchResult.AWAY_WIN
                else:
                    if match.away_goals > match.home_goals:
                        result_enum = MatchResult.AWAY_WIN
                    elif match.away_goals == match.home_goals:
                        result_enum = MatchResult.DRAW
                    else:
                        result_enum = MatchResult.HOME_WIN
            else:
                result_enum = None
            
            result.append(MatchResponse(
                id=match.id,
                date=match.date,
                season=match.season,
                matchday=match.matchday,
                home_team=match.home_team.name,
                away_team=match.away_team.name,
                home_goals=match.home_goals,
                away_goals=match.away_goals,
                result=result_enum,
                referee=match.referee,
                venue=match.venue,
                attendance=match.attendance,
                home_shots=match.home_shots,
                away_shots=match.away_shots,
                home_shots_on_target=match.home_shots_on_target,
                away_shots_on_target=match.away_shots_on_target,
                home_fouls=match.home_fouls,
                away_fouls=match.away_fouls,
                home_corners=match.home_corners,
                away_corners=match.away_corners,
                home_yellow_cards=match.home_yellow_cards,
                away_yellow_cards=match.away_yellow_cards,
                home_red_cards=match.home_red_cards,
                away_red_cards=match.away_red_cards
            ))
        
        return result
    
    async def get_head_to_head(
        self,
        team_id: int,
        opponent_id: int,
        limit: int = 10,
        venue: Optional[str] = None,
        db: Session = None
    ) -> List[MatchResponse]:
        """Get head-to-head matches between two teams"""
        
        query = db.query(Match).filter(
            or_(
                and_(Match.home_team_id == team_id, Match.away_team_id == opponent_id),
                and_(Match.home_team_id == opponent_id, Match.away_team_id == team_id)
            )
        )
        
        if venue == "home":
            query = query.filter(Match.home_team_id == team_id)
        elif venue == "away":
            query = query.filter(Match.away_team_id == team_id)
        
        matches = query.order_by(
            desc(Match.date)
        ).limit(limit).all()
        
        return await self._convert_matches_to_response(matches)
    
    async def get_matches(
        self,
        season: Optional[str] = None,
        team_id: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
        db: Session = None
    ) -> List[MatchResponse]:
        """Get all matches with optional filtering"""
        
        query = db.query(Match)
        
        if season:
            query = query.filter(Match.season == season)
        
        if team_id:
            query = query.filter(
                or_(
                    Match.home_team_id == team_id,
                    Match.away_team_id == team_id
                )
            )
        
        matches = query.order_by(
            desc(Match.date)
        ).offset(offset).limit(limit).all()
        
        return await self._convert_matches_to_response(matches)

    async def get_match_by_id(self, match_id: int, db: Session) -> Optional[MatchResponse]:
        """Get a specific match by ID"""
        
        match = db.query(Match).filter(Match.id == match_id).first()
        
        if not match:
            return None
        
        matches = await self._convert_matches_to_response([match])
        return matches[0] if matches else None

    async def get_matches_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: int = 50,
        db: Session = None
    ) -> List[MatchResponse]:
        """Get matches within a date range"""
        
        matches = db.query(Match).filter(
            and_(
                Match.date >= start_date.date(),
                Match.date <= end_date.date()
            )
        ).order_by(desc(Match.date)).limit(limit).all()
        
        return await self._convert_matches_to_response(matches)

    async def search_teams(
        self,
        query: str,
        limit: int = 10,
        db: Session = None
    ) -> List[TeamResponse]:
        """Search teams by name"""
        
        teams = db.query(Team).filter(
            or_(
                Team.name.ilike(f"%{query}%"),
                Team.normalized_name.ilike(f"%{query}%"),
                Team.city.ilike(f"%{query}%")
            )
        ).limit(limit).all()
        
        result = []
        for team in teams:
            result.append(TeamResponse(
                id=team.id,
                name=team.name,
                normalized_name=team.normalized_name,
                founded_year=team.founded_year,
                stadium=team.stadium,
                city=team.city,
                is_active=team.is_active,
                created_at=team.created_at,
                updated_at=team.updated_at
            ))
        
        return result
    
    async def _convert_matches_to_response(self, matches: List[Match]) -> List[MatchResponse]:
        """Convert Match objects to MatchResponse objects"""
        
        result = []
        for match in matches:
            # Determine result
            if match.home_goals is not None and match.away_goals is not None:
                if match.home_goals > match.away_goals:
                    result_enum = MatchResult.HOME_WIN
                elif match.home_goals == match.away_goals:
                    result_enum = MatchResult.DRAW
                else:
                    result_enum = MatchResult.AWAY_WIN
            else:
                result_enum = None
            
            result.append(MatchResponse(
                id=match.id,
                date=match.date,
                season=match.season,
                matchday=match.matchday,
                home_team=match.home_team.name,
                away_team=match.away_team.name,
                home_goals=match.home_goals,
                away_goals=match.away_goals,
                result=result_enum,
                referee=match.referee,
                venue=match.venue,
                attendance=match.attendance,
                home_shots=match.home_shots,
                away_shots=match.away_shots,
                home_shots_on_target=match.home_shots_on_target,
                away_shots_on_target=match.away_shots_on_target,
                home_fouls=match.home_fouls,
                away_fouls=match.away_fouls,
                home_corners=match.home_corners,
                away_corners=match.away_corners,
                home_yellow_cards=match.home_yellow_cards,
                away_yellow_cards=match.away_yellow_cards,
                home_red_cards=match.home_red_cards,
                away_red_cards=match.away_red_cards
            ))
        
        return result