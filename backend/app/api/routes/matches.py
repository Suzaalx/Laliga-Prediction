from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from app.models.database import get_db
from app.models.schemas import (
    MatchResponse, ErrorResponse
)
from app.services.team_service import TeamService
from app.services.cache_service import CacheService
from app.core.logging import get_logger
from app.core.config import settings

router = APIRouter()
logger = get_logger("matches")

# Initialize services
team_service = TeamService()
cache_service = CacheService()

@router.get("/", response_model=List[MatchResponse])
async def get_matches(
    season: Optional[str] = None,
    team_id: Optional[int] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """Get all matches, optionally filtered by season or team"""
    
    try:
        matches = await team_service.get_matches(
            season=season,
            team_id=team_id,
            limit=limit,
            offset=offset,
            db=db
        )
        
        logger.info(
            "Matches retrieved",
            count=len(matches),
            season=season,
            team_id=team_id,
            limit=limit,
            offset=offset
        )
        
        return matches
        
    except Exception as e:
        logger.error("Failed to get matches", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get matches: {str(e)}"
        )

@router.get("/{match_id}", response_model=MatchResponse)
async def get_match(
    match_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific match by ID"""
    
    try:
        match = await team_service.get_match_by_id(match_id, db)
        
        if not match:
            raise HTTPException(
                status_code=404,
                detail=f"Match with ID {match_id} not found"
            )
        
        logger.info("Match retrieved", match_id=match_id)
        return match
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get match", match_id=match_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get match: {str(e)}"
        )

@router.get("/recent", response_model=List[MatchResponse])
async def get_recent_matches(
    days: int = Query(7, ge=1, le=30),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get recent matches from the last N days"""
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        matches = await team_service.get_matches_by_date_range(
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            db=db
        )
        
        logger.info(
            "Recent matches retrieved",
            count=len(matches),
            days=days,
            limit=limit
        )
        
        return matches
        
    except Exception as e:
        logger.error("Failed to get recent matches", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get recent matches: {str(e)}"
        )

@router.get("/upcoming", response_model=List[MatchResponse])
async def get_upcoming_matches(
    days: int = Query(7, ge=1, le=30),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get upcoming matches for the next N days"""
    
    try:
        start_date = datetime.now()
        end_date = start_date + timedelta(days=days)
        
        matches = await team_service.get_matches_by_date_range(
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            db=db
        )
        
        logger.info(
            "Upcoming matches retrieved",
            count=len(matches),
            days=days,
            limit=limit
        )
        
        return matches
        
    except Exception as e:
        logger.error("Failed to get upcoming matches", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get upcoming matches: {str(e)}"
        )