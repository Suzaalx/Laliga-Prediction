from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from app.models.database import get_db
from app.models.schemas import (
    TeamResponse, TeamStatsResponse, TeamFormResponse,
    MatchResponse, ErrorResponse
)
from app.services.team_service import TeamService
from app.services.cache_service import CacheService
from app.core.logging import get_logger
from app.core.config import settings

router = APIRouter()
logger = get_logger("teams")

# Initialize services
team_service = TeamService()
cache_service = CacheService()

@router.get("/", response_model=List[TeamResponse])
async def get_teams(
    season: Optional[str] = None,
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """Get all teams, optionally filtered by season"""
    
    try:
        teams = await team_service.get_teams(
            season=season,
            active_only=active_only,
            db=db
        )
        
        logger.info(
            "Teams retrieved",
            count=len(teams),
            season=season,
            active_only=active_only
        )
        
        return teams
        
    except Exception as e:
        logger.error("Failed to get teams", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get teams: {str(e)}"
        )

@router.get("/{team_id}", response_model=TeamResponse)
async def get_team(
    team_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific team by ID"""
    
    try:
        # Check cache first
        cache_key = f"team:{team_id}"
        cached_team = await cache_service.get(cache_key)
        
        if cached_team:
            logger.info("Returning cached team", team_id=team_id)
            return cached_team
        
        team = await team_service.get_team_by_id(team_id, db)
        
        if not team:
            raise HTTPException(
                status_code=404,
                detail=f"Team with ID {team_id} not found"
            )
        
        # Cache the result
        await cache_service.set(cache_key, team, ttl=settings.CACHE_TTL)
        
        logger.info("Team retrieved", team_id=team_id, team_name=team.name)
        
        return team
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get team", team_id=team_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get team: {str(e)}"
        )

@router.get("/name/{team_name}", response_model=TeamResponse)
async def get_team_by_name(
    team_name: str,
    db: Session = Depends(get_db)
):
    """Get a specific team by name"""
    
    try:
        # Check cache first
        cache_key = f"team:name:{team_name.lower()}"
        cached_team = await cache_service.get(cache_key)
        
        if cached_team:
            logger.info("Returning cached team", team_name=team_name)
            return cached_team
        
        team = await team_service.get_team_by_name(team_name, db)
        
        if not team:
            raise HTTPException(
                status_code=404,
                detail=f"Team '{team_name}' not found"
            )
        
        # Cache the result
        await cache_service.set(cache_key, team, ttl=settings.CACHE_TTL)
        
        logger.info("Team retrieved by name", team_name=team_name, team_id=team.id)
        
        return team
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get team by name", team_name=team_name, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get team: {str(e)}"
        )

@router.get("/{team_id}/stats", response_model=TeamStatsResponse)
async def get_team_stats(
    team_id: int,
    season: Optional[str] = None,
    venue: Optional[str] = Query(None, regex="^(home|away|all)$"),
    last_n_matches: Optional[int] = Query(None, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Get team statistics"""
    
    try:
        # Validate team exists
        team = await team_service.get_team_by_id(team_id, db)
        if not team:
            raise HTTPException(
                status_code=404,
                detail=f"Team with ID {team_id} not found"
            )
        
        # Check cache
        cache_key = f"team_stats:{team_id}:{season}:{venue}:{last_n_matches}"
        cached_stats = await cache_service.get(cache_key)
        
        if cached_stats:
            logger.info("Returning cached team stats", team_id=team_id)
            return cached_stats
        
        stats = await team_service.get_team_stats(
            team_id=team_id,
            season=season,
            venue=venue,
            last_n_matches=last_n_matches,
            db=db
        )
        
        # Cache the result
        await cache_service.set(cache_key, stats, ttl=settings.CACHE_TTL)
        
        logger.info(
            "Team stats retrieved",
            team_id=team_id,
            season=season,
            venue=venue,
            last_n_matches=last_n_matches
        )
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get team stats", team_id=team_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get team stats: {str(e)}"
        )

@router.get("/{team_id}/form", response_model=TeamFormResponse)
async def get_team_form(
    team_id: int,
    matches: int = Query(10, ge=1, le=20),
    venue: Optional[str] = Query(None, regex="^(home|away|all)$"),
    db: Session = Depends(get_db)
):
    """Get team recent form"""
    
    try:
        # Validate team exists
        team = await team_service.get_team_by_id(team_id, db)
        if not team:
            raise HTTPException(
                status_code=404,
                detail=f"Team with ID {team_id} not found"
            )
        
        # Check cache
        cache_key = f"team_form:{team_id}:{matches}:{venue}"
        cached_form = await cache_service.get(cache_key)
        
        if cached_form:
            logger.info("Returning cached team form", team_id=team_id)
            return cached_form
        
        form = await team_service.get_team_form(
            team_id=team_id,
            matches=matches,
            venue=venue,
            db=db
        )
        
        # Cache the result
        await cache_service.set(cache_key, form, ttl=settings.CACHE_TTL)
        
        logger.info(
            "Team form retrieved",
            team_id=team_id,
            matches=matches,
            venue=venue
        )
        
        return form
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get team form", team_id=team_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get team form: {str(e)}"
        )

@router.get("/{team_id}/matches", response_model=List[MatchResponse])
async def get_team_matches(
    team_id: int,
    season: Optional[str] = None,
    venue: Optional[str] = Query(None, regex="^(home|away|all)$"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """Get matches for a specific team"""
    
    try:
        # Validate team exists
        team = await team_service.get_team_by_id(team_id, db)
        if not team:
            raise HTTPException(
                status_code=404,
                detail=f"Team with ID {team_id} not found"
            )
        
        matches = await team_service.get_team_matches(
            team_id=team_id,
            season=season,
            venue=venue,
            limit=limit,
            offset=offset,
            db=db
        )
        
        logger.info(
            "Team matches retrieved",
            team_id=team_id,
            season=season,
            venue=venue,
            count=len(matches)
        )
        
        return matches
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get team matches", team_id=team_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get team matches: {str(e)}"
        )

@router.get("/{team_id}/head-to-head/{opponent_id}", response_model=List[MatchResponse])
async def get_head_to_head(
    team_id: int,
    opponent_id: int,
    limit: int = Query(10, ge=1, le=50),
    venue: Optional[str] = Query(None, regex="^(home|away|all)$"),
    db: Session = Depends(get_db)
):
    """Get head-to-head matches between two teams"""
    
    try:
        # Validate both teams exist
        team = await team_service.get_team_by_id(team_id, db)
        opponent = await team_service.get_team_by_id(opponent_id, db)
        
        if not team:
            raise HTTPException(
                status_code=404,
                detail=f"Team with ID {team_id} not found"
            )
        
        if not opponent:
            raise HTTPException(
                status_code=404,
                detail=f"Opponent team with ID {opponent_id} not found"
            )
        
        # Check cache
        cache_key = f"h2h:{team_id}:{opponent_id}:{limit}:{venue}"
        cached_h2h = await cache_service.get(cache_key)
        
        if cached_h2h:
            logger.info("Returning cached head-to-head", team_id=team_id, opponent_id=opponent_id)
            return cached_h2h
        
        matches = await team_service.get_head_to_head(
            team_id=team_id,
            opponent_id=opponent_id,
            limit=limit,
            venue=venue,
            db=db
        )
        
        # Cache the result
        await cache_service.set(cache_key, matches, ttl=settings.CACHE_TTL)
        
        logger.info(
            "Head-to-head matches retrieved",
            team_id=team_id,
            opponent_id=opponent_id,
            count=len(matches)
        )
        
        return matches
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get head-to-head matches",
            team_id=team_id,
            opponent_id=opponent_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get head-to-head matches: {str(e)}"
        )

@router.get("/search")
async def search_teams(
    query: str = Query(..., min_length=2),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Search teams by name"""
    
    try:
        teams = await team_service.search_teams(
            query=query,
            limit=limit,
            db=db
        )
        
        logger.info(
            "Team search completed",
            query=query,
            results_count=len(teams)
        )
        
        return teams
        
    except Exception as e:
        logger.error("Team search failed", query=query, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Team search failed: {str(e)}"
        )