from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from app.services.team_analytics_service import TeamAnalyticsService
from app.core.logging import get_logger

logger = get_logger("team_analytics_routes")

router = APIRouter()

# Dependency to get team analytics service
def get_team_analytics_service() -> TeamAnalyticsService:
    return TeamAnalyticsService()

@router.get("/teams/{team_name}/analytics")
async def get_team_analytics(
    team_name: str,
    service: TeamAnalyticsService = Depends(get_team_analytics_service)
) -> Dict[str, Any]:
    """
    Get comprehensive analytics for a specific team.
    
    Args:
        team_name: Name of the team to get analytics for
        
    Returns:
        Dictionary containing team analytics data including:
        - form_metrics: Team form metrics from team_form.csv
        - match_statistics: Overall match statistics
        - performance_trends: Performance trends over time
        - recent_form: Recent form data
        - head_to_head: Head-to-head summary
    """
    try:
        logger.info(f"Getting analytics for team: {team_name}")
        analytics = service.get_team_analytics(team_name)
        
        if "error" in analytics:
            raise HTTPException(status_code=404, detail=analytics["error"])
            
        return {
            "success": True,
            "team_name": team_name,
            "analytics": analytics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting team analytics for {team_name}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error while getting analytics for {team_name}"
        )

@router.get("/teams/{team_name}/form")
async def get_team_form(
    team_name: str,
    service: TeamAnalyticsService = Depends(get_team_analytics_service)
) -> Dict[str, Any]:
    """
    Get team form metrics only.
    
    Args:
        team_name: Name of the team to get form data for
        
    Returns:
        Dictionary containing team form metrics
    """
    try:
        logger.info(f"Getting form data for team: {team_name}")
        analytics = service.get_team_analytics(team_name)
        
        if "error" in analytics:
            raise HTTPException(status_code=404, detail=analytics["error"])
            
        return {
            "success": True,
            "team_name": team_name,
            "form_metrics": analytics.get("form_metrics", {}),
            "recent_form": analytics.get("recent_form", [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting team form for {team_name}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error while getting form data for {team_name}"
        )

@router.get("/teams/{team_name}/statistics")
async def get_team_statistics(
    team_name: str,
    service: TeamAnalyticsService = Depends(get_team_analytics_service)
) -> Dict[str, Any]:
    """
    Get team match statistics only.
    
    Args:
        team_name: Name of the team to get statistics for
        
    Returns:
        Dictionary containing team match statistics
    """
    try:
        logger.info(f"Getting statistics for team: {team_name}")
        analytics = service.get_team_analytics(team_name)
        
        if "error" in analytics:
            raise HTTPException(status_code=404, detail=analytics["error"])
            
        return {
            "success": True,
            "team_name": team_name,
            "match_statistics": analytics.get("match_statistics", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting team statistics for {team_name}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error while getting statistics for {team_name}"
        )

@router.get("/teams/{team_name}/trends")
async def get_team_trends(
    team_name: str,
    service: TeamAnalyticsService = Depends(get_team_analytics_service)
) -> Dict[str, Any]:
    """
    Get team performance trends only.
    
    Args:
        team_name: Name of the team to get trends for
        
    Returns:
        Dictionary containing team performance trends
    """
    try:
        logger.info(f"Getting trends for team: {team_name}")
        analytics = service.get_team_analytics(team_name)
        
        if "error" in analytics:
            raise HTTPException(status_code=404, detail=analytics["error"])
            
        return {
            "success": True,
            "team_name": team_name,
            "performance_trends": analytics.get("performance_trends", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting team trends for {team_name}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error while getting trends for {team_name}"
        )