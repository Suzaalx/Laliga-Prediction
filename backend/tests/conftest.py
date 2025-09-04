import pytest
import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

# Set test environment
os.environ["ENVIRONMENT"] = "testing"
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["LOG_LEVEL"] = "DEBUG"

from app.main import app
from app.models.database import get_db, Base
from app.models.models import Team, Match, Prediction, TeamStats
from app.services import PredictionService, CacheService, TeamService, ModelService, DataService
from app.core.config import get_settings

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def settings():
    """Get test settings"""
    return get_settings()

@pytest.fixture(scope="session")
def test_engine(settings):
    """Create test database engine"""
    engine = create_engine(
        settings.database.url,
        connect_args={"check_same_thread": False} if "sqlite" in settings.database.url else {}
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def test_db(test_engine):
    """Create test database session"""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    
    try:
        yield session
    finally:
        session.rollback()
        session.close()

@pytest.fixture(scope="function")
def client(test_db):
    """Create test client with database dependency override"""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()

@pytest.fixture
def sample_teams() -> List[Dict[str, Any]]:
    """Sample team data for testing"""
    return [
        {
            "name": "Real Madrid",
            "short_name": "RMA",
            "country": "Spain",
            "founded": 1902,
            "stadium": "Santiago Bernabéu"
        },
        {
            "name": "FC Barcelona",
            "short_name": "BAR",
            "country": "Spain",
            "founded": 1899,
            "stadium": "Camp Nou"
        },
        {
            "name": "Atlético Madrid",
            "short_name": "ATM",
            "country": "Spain",
            "founded": 1903,
            "stadium": "Wanda Metropolitano"
        },
        {
            "name": "Sevilla FC",
            "short_name": "SEV",
            "country": "Spain",
            "founded": 1890,
            "stadium": "Ramón Sánchez Pizjuán"
        }
    ]

@pytest.fixture
def sample_matches() -> List[Dict[str, Any]]:
    """Sample match data for testing"""
    base_date = datetime(2024, 1, 1)
    
    return [
        {
            "date": base_date,
            "home_team": "Real Madrid",
            "away_team": "FC Barcelona",
            "home_goals": 2,
            "away_goals": 1,
            "home_goals_ht": 1,
            "away_goals_ht": 0,
            "result": "H",
            "result_ht": "H",
            "referee": "Antonio Mateu Lahoz",
            "home_shots": 12,
            "away_shots": 8,
            "home_shots_target": 6,
            "away_shots_target": 3,
            "home_fouls": 14,
            "away_fouls": 16,
            "home_corners": 5,
            "away_corners": 7,
            "home_yellow_cards": 2,
            "away_yellow_cards": 3,
            "home_red_cards": 0,
            "away_red_cards": 1
        },
        {
            "date": base_date + timedelta(days=7),
            "home_team": "Atlético Madrid",
            "away_team": "Sevilla FC",
            "home_goals": 1,
            "away_goals": 1,
            "home_goals_ht": 0,
            "away_goals_ht": 1,
            "result": "D",
            "result_ht": "A",
            "referee": "José María Sánchez Martínez",
            "home_shots": 10,
            "away_shots": 9,
            "home_shots_target": 4,
            "away_shots_target": 4,
            "home_fouls": 12,
            "away_fouls": 11,
            "home_corners": 6,
            "away_corners": 4,
            "home_yellow_cards": 1,
            "away_yellow_cards": 2,
            "home_red_cards": 0,
            "away_red_cards": 0
        },
        {
            "date": base_date + timedelta(days=14),
            "home_team": "FC Barcelona",
            "away_team": "Atlético Madrid",
            "home_goals": 3,
            "away_goals": 0,
            "home_goals_ht": 2,
            "away_goals_ht": 0,
            "result": "H",
            "result_ht": "H",
            "referee": "Jesús Gil Manzano",
            "home_shots": 15,
            "away_shots": 6,
            "home_shots_target": 8,
            "away_shots_target": 2,
            "home_fouls": 8,
            "away_fouls": 18,
            "home_corners": 8,
            "away_corners": 3,
            "home_yellow_cards": 1,
            "away_yellow_cards": 4,
            "home_red_cards": 0,
            "away_red_cards": 0
        }
    ]

@pytest.fixture
def sample_csv_data() -> pd.DataFrame:
    """Sample CSV data for testing"""
    data = {
        'Date': ['01/01/2024', '08/01/2024', '15/01/2024'],
        'HomeTeam': ['Real Madrid', 'Atletico Madrid', 'Barcelona'],
        'AwayTeam': ['Barcelona', 'Sevilla', 'Atletico Madrid'],
        'FTHG': [2, 1, 3],
        'FTAG': [1, 1, 0],
        'HTHG': [1, 0, 2],
        'HTAG': [0, 1, 0],
        'FTR': ['H', 'D', 'H'],
        'HTR': ['H', 'A', 'H'],
        'Referee': ['A. Mateu Lahoz', 'J. Sanchez Martinez', 'J. Gil Manzano'],
        'HS': [12, 10, 15],
        'AS': [8, 9, 6],
        'HST': [6, 4, 8],
        'AST': [3, 4, 2],
        'HF': [14, 12, 8],
        'AF': [16, 11, 18],
        'HC': [5, 6, 8],
        'AC': [7, 4, 3],
        'HY': [2, 1, 1],
        'AY': [3, 2, 4],
        'HR': [0, 0, 0],
        'AR': [1, 0, 0]
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def mock_cache_service():
    """Mock cache service for testing"""
    cache_service = AsyncMock(spec=CacheService)
    
    # Configure default return values
    cache_service.get.return_value = None
    cache_service.set.return_value = True
    cache_service.delete.return_value = True
    cache_service.exists.return_value = False
    cache_service.get_health.return_value = {
        "status": "healthy",
        "connected": True,
        "memory_usage": "10MB",
        "total_keys": 0
    }
    
    return cache_service

@pytest.fixture
def mock_prediction_service():
    """Mock prediction service for testing"""
    service = AsyncMock(spec=PredictionService)
    
    # Configure default return values
    service.predict_match.return_value = {
        "home_win_probability": 0.45,
        "draw_probability": 0.25,
        "away_win_probability": 0.30,
        "expected_home_goals": 1.8,
        "expected_away_goals": 1.2,
        "confidence": 0.75,
        "model_version": "1.0.0"
    }
    
    service.get_upcoming_matches.return_value = []
    service.get_prediction_history.return_value = []
    service.get_prediction_stats.return_value = {
        "total_predictions": 100,
        "accuracy": 0.65,
        "brier_score": 0.25,
        "log_loss": 0.55
    }
    
    return service

@pytest.fixture
def mock_team_service():
    """Mock team service for testing"""
    service = AsyncMock(spec=TeamService)
    
    # Configure default return values
    service.get_all_teams.return_value = []
    service.get_team_by_id.return_value = None
    service.get_team_by_name.return_value = None
    service.get_team_stats.return_value = {
        "matches_played": 20,
        "wins": 12,
        "draws": 5,
        "losses": 3,
        "goals_scored": 35,
        "goals_conceded": 18,
        "points": 41
    }
    service.get_team_form.return_value = ["W", "W", "D", "W", "L"]
    service.search_teams.return_value = []
    
    return service

@pytest.fixture
def mock_model_service():
    """Mock model service for testing"""
    service = AsyncMock(spec=ModelService)
    
    # Configure default return values
    service.get_active_model.return_value = {
        "id": 1,
        "version": "1.0.0",
        "name": "Dixon-Coles v1.0",
        "accuracy": 0.65,
        "created_at": datetime.now(),
        "is_active": True
    }
    
    service.get_all_models.return_value = []
    service.get_model_performance.return_value = {
        "accuracy": 0.65,
        "precision": 0.62,
        "recall": 0.68,
        "f1_score": 0.65,
        "brier_score": 0.25,
        "log_loss": 0.55
    }
    
    service.start_retraining.return_value = {"task_id": "test-task-123"}
    service.get_retraining_status.return_value = {
        "status": "completed",
        "progress": 100,
        "message": "Training completed successfully"
    }
    
    return service

@pytest.fixture
def mock_data_service():
    """Mock data service for testing"""
    service = AsyncMock(spec=DataService)
    
    # Configure default return values
    service.ingest_csv_data.return_value = {
        "matches_processed": 100,
        "teams_created": 20,
        "errors": []
    }
    
    service.get_data_summary.return_value = {
        "total_matches": 1000,
        "total_teams": 20,
        "seasons": ["2023-24", "2022-23"],
        "date_range": {
            "start": "2022-08-01",
            "end": "2024-05-31"
        }
    }
    
    return service

@pytest.fixture
def populated_test_db(test_db, sample_teams, sample_matches):
    """Populate test database with sample data"""
    # Add teams
    teams = []
    for team_data in sample_teams:
        team = Team(**team_data)
        test_db.add(team)
        teams.append(team)
    
    test_db.commit()
    
    # Add matches
    for match_data in sample_matches:
        # Find teams
        home_team = test_db.query(Team).filter(Team.name == match_data["home_team"]).first()
        away_team = test_db.query(Team).filter(Team.name == match_data["away_team"]).first()
        
        if home_team and away_team:
            match = Match(
                date=match_data["date"],
                home_team_id=home_team.id,
                away_team_id=away_team.id,
                home_goals=match_data["home_goals"],
                away_goals=match_data["away_goals"],
                home_goals_ht=match_data["home_goals_ht"],
                away_goals_ht=match_data["away_goals_ht"],
                result=match_data["result"],
                result_ht=match_data["result_ht"],
                referee=match_data["referee"],
                home_shots=match_data["home_shots"],
                away_shots=match_data["away_shots"],
                home_shots_target=match_data["home_shots_target"],
                away_shots_target=match_data["away_shots_target"],
                home_fouls=match_data["home_fouls"],
                away_fouls=match_data["away_fouls"],
                home_corners=match_data["home_corners"],
                away_corners=match_data["away_corners"],
                home_yellow_cards=match_data["home_yellow_cards"],
                away_yellow_cards=match_data["away_yellow_cards"],
                home_red_cards=match_data["home_red_cards"],
                away_red_cards=match_data["away_red_cards"]
            )
            test_db.add(match)
    
    test_db.commit()
    
    return test_db

# Async test utilities
@pytest.fixture
def anyio_backend():
    return "asyncio"

# Custom markers
pytestmark = pytest.mark.asyncio