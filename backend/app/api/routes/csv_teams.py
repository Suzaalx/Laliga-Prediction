from typing import List, Dict
from fastapi import APIRouter, HTTPException, Depends
from app.services.csv_team_service import CSVTeamService
from app.core.database import get_db
from sqlalchemy.orm import Session

router = APIRouter()

@router.get("/teams", response_model=List[Dict])
async def get_csv_teams():
    """
    Get all teams from CSV data with enriched information.
    """
    try:
        csv_team_service = CSVTeamService()
        teams_data = csv_team_service.get_all_teams_data()
        return teams_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving teams data: {str(e)}")

@router.get("/teams/{team_name}", response_model=Dict)
async def get_csv_team_by_name(team_name: str):
    """
    Get specific team data by name with enriched information.
    """
    try:
        csv_team_service = CSVTeamService()
        team_data = csv_team_service.get_enriched_team_data(team_name)
        team_stats = csv_team_service.get_team_statistics(team_name)
        
        # Combine team data with statistics
        team_data.update({"statistics": team_stats})
        
        return team_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving team data: {str(e)}")

@router.get("/teams-summary", response_model=Dict)
async def get_csv_teams_summary():
    """
    Get summary of teams data from CSV.
    """
    try:
        csv_team_service = CSVTeamService()
        summary = csv_team_service.get_teams_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving teams summary: {str(e)}")

@router.get("/teams-list", response_model=List[str])
async def get_csv_teams_list():
    """
    Get simple list of team names from CSV.
    """
    try:
        csv_team_service = CSVTeamService()
        teams = csv_team_service.get_teams_from_csv()
        return teams
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving teams list: {str(e)}")