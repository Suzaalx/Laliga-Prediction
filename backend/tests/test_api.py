import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from app.main import app

class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_check(self, client):
        """Test main health check endpoint"""
        response = client.get("/api/health/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "database" in data
        assert "cache" in data
        assert "model" in data
        assert "system" in data
    
    def test_liveness_probe(self, client):
        """Test Kubernetes liveness probe"""
        response = client.get("/api/health/liveness")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "alive"
        assert "timestamp" in data
    
    def test_readiness_probe(self, client):
        """Test Kubernetes readiness probe"""
        response = client.get("/api/health/readiness")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "checks" in data
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint"""
        response = client.get("/api/health/metrics")
        
        assert response.status_code == 200
        # Should return plain text metrics
        assert "text/plain" in response.headers["content-type"]

class TestPredictionEndpoints:
    """Test prediction API endpoints"""
    
    @patch('app.services.PredictionService')
    def test_predict_match(self, mock_service, client):
        """Test single match prediction"""
        # Mock service response
        mock_instance = AsyncMock()
        mock_instance.predict_match.return_value = {
            "home_win_probability": 0.45,
            "draw_probability": 0.25,
            "away_win_probability": 0.30,
            "expected_home_goals": 1.8,
            "expected_away_goals": 1.2,
            "confidence": 0.75,
            "model_version": "1.0.0"
        }
        mock_service.return_value = mock_instance
        
        response = client.post(
            "/api/predictions/predict",
            json={
                "home_team": "Real Madrid",
                "away_team": "FC Barcelona",
                "match_date": "2024-03-15T20:00:00"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "home_win_probability" in data
        assert "draw_probability" in data
        assert "away_win_probability" in data
        assert "expected_home_goals" in data
        assert "expected_away_goals" in data
        assert "confidence" in data
        assert "model_version" in data
    
    def test_predict_match_invalid_data(self, client):
        """Test prediction with invalid data"""
        response = client.post(
            "/api/predictions/predict",
            json={
                "home_team": "",  # Empty team name
                "away_team": "FC Barcelona"
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_match_same_teams(self, client):
        """Test prediction with same home and away teams"""
        response = client.post(
            "/api/predictions/predict",
            json={
                "home_team": "Real Madrid",
                "away_team": "Real Madrid"
            }
        )
        
        assert response.status_code == 400  # Bad request
    
    @patch('app.services.PredictionService')
    def test_batch_predictions(self, mock_service, client):
        """Test batch predictions"""
        # Mock service response
        mock_instance = AsyncMock()
        mock_instance.predict_batch.return_value = [
            {
                "match_id": 1,
                "home_team": "Real Madrid",
                "away_team": "FC Barcelona",
                "home_win_probability": 0.45,
                "draw_probability": 0.25,
                "away_win_probability": 0.30
            }
        ]
        mock_service.return_value = mock_instance
        
        response = client.post(
            "/api/predictions/batch",
            json={
                "matches": [
                    {
                        "home_team": "Real Madrid",
                        "away_team": "FC Barcelona",
                        "match_date": "2024-03-15T20:00:00"
                    }
                ]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert len(data["predictions"]) == 1
    
    @patch('app.services.PredictionService')
    def test_upcoming_matches(self, mock_service, client):
        """Test upcoming matches endpoint"""
        # Mock service response
        mock_instance = AsyncMock()
        mock_instance.get_upcoming_matches.return_value = [
            {
                "match_date": "2024-03-15T20:00:00",
                "home_team": "Real Madrid",
                "away_team": "FC Barcelona",
                "home_win_probability": 0.45,
                "draw_probability": 0.25,
                "away_win_probability": 0.30
            }
        ]
        mock_service.return_value = mock_instance
        
        response = client.get("/api/predictions/upcoming")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "matches" in data
        assert len(data["matches"]) == 1
    
    @patch('app.services.PredictionService')
    def test_prediction_history(self, mock_service, client):
        """Test prediction history endpoint"""
        # Mock service response
        mock_instance = AsyncMock()
        mock_instance.get_prediction_history.return_value = [
            {
                "id": 1,
                "match_date": "2024-03-01T20:00:00",
                "home_team": "Real Madrid",
                "away_team": "FC Barcelona",
                "predicted_result": "H",
                "actual_result": "H",
                "correct": True
            }
        ]
        mock_service.return_value = mock_instance
        
        response = client.get("/api/predictions/history")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert len(data["predictions"]) == 1
    
    @patch('app.services.PredictionService')
    def test_prediction_stats(self, mock_service, client):
        """Test prediction statistics endpoint"""
        # Mock service response
        mock_instance = AsyncMock()
        mock_instance.get_prediction_stats.return_value = {
            "total_predictions": 100,
            "accuracy": 0.65,
            "brier_score": 0.25,
            "log_loss": 0.55,
            "by_month": {}
        }
        mock_service.return_value = mock_instance
        
        response = client.get("/api/predictions/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_predictions" in data
        assert "accuracy" in data
        assert "brier_score" in data
        assert "log_loss" in data
    
    @patch('app.services.CacheService')
    def test_clear_cache(self, mock_cache, client):
        """Test cache clearing endpoint"""
        # Mock cache service
        mock_instance = AsyncMock()
        mock_instance.delete_pattern.return_value = 5
        mock_cache.return_value = mock_instance
        
        response = client.delete("/api/predictions/cache")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "cleared_keys" in data

class TestTeamEndpoints:
    """Test team API endpoints"""
    
    @patch('app.services.TeamService')
    def test_get_all_teams(self, mock_service, client):
        """Test get all teams endpoint"""
        # Mock service response
        mock_instance = AsyncMock()
        mock_instance.get_all_teams.return_value = [
            {
                "id": 1,
                "name": "Real Madrid",
                "short_name": "RMA",
                "country": "Spain"
            },
            {
                "id": 2,
                "name": "FC Barcelona",
                "short_name": "BAR",
                "country": "Spain"
            }
        ]
        mock_service.return_value = mock_instance
        
        response = client.get("/api/teams/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "teams" in data
        assert len(data["teams"]) == 2
    
    @patch('app.services.TeamService')
    def test_get_team_by_id(self, mock_service, client):
        """Test get team by ID endpoint"""
        # Mock service response
        mock_instance = AsyncMock()
        mock_instance.get_team_by_id.return_value = {
            "id": 1,
            "name": "Real Madrid",
            "short_name": "RMA",
            "country": "Spain",
            "founded": 1902
        }
        mock_service.return_value = mock_instance
        
        response = client.get("/api/teams/1")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == 1
        assert data["name"] == "Real Madrid"
    
    @patch('app.services.TeamService')
    def test_get_team_not_found(self, mock_service, client):
        """Test get team that doesn't exist"""
        # Mock service response
        mock_instance = AsyncMock()
        mock_instance.get_team_by_id.return_value = None
        mock_service.return_value = mock_instance
        
        response = client.get("/api/teams/999")
        
        assert response.status_code == 404
    
    @patch('app.services.TeamService')
    def test_get_team_stats(self, mock_service, client):
        """Test get team statistics endpoint"""
        # Mock service response
        mock_instance = AsyncMock()
        mock_instance.get_team_stats.return_value = {
            "matches_played": 20,
            "wins": 12,
            "draws": 5,
            "losses": 3,
            "goals_scored": 35,
            "goals_conceded": 18,
            "points": 41
        }
        mock_service.return_value = mock_instance
        
        response = client.get("/api/teams/1/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "matches_played" in data
        assert "wins" in data
        assert "goals_scored" in data
    
    @patch('app.services.TeamService')
    def test_get_team_form(self, mock_service, client):
        """Test get team form endpoint"""
        # Mock service response
        mock_instance = AsyncMock()
        mock_instance.get_team_form.return_value = {
            "recent_results": ["W", "W", "D", "L", "W"],
            "form_points": 2.4,
            "matches_analyzed": 5
        }
        mock_service.return_value = mock_instance
        
        response = client.get("/api/teams/1/form")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "recent_results" in data
        assert "form_points" in data
    
    @patch('app.services.TeamService')
    def test_search_teams(self, mock_service, client):
        """Test team search endpoint"""
        # Mock service response
        mock_instance = AsyncMock()
        mock_instance.search_teams.return_value = [
            {
                "id": 1,
                "name": "Real Madrid",
                "short_name": "RMA",
                "match_score": 0.95
            }
        ]
        mock_service.return_value = mock_instance
        
        response = client.get("/api/teams/search?q=Real")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "teams" in data
        assert len(data["teams"]) == 1
        assert data["teams"][0]["name"] == "Real Madrid"
    
    def test_search_teams_no_query(self, client):
        """Test team search without query parameter"""
        response = client.get("/api/teams/search")
        
        assert response.status_code == 400

class TestModelEndpoints:
    """Test model management API endpoints"""
    
    @patch('app.services.ModelService')
    def test_get_active_model(self, mock_service, client):
        """Test get active model endpoint"""
        # Mock service response
        mock_instance = AsyncMock()
        mock_instance.get_active_model.return_value = {
            "id": 1,
            "version": "1.0.0",
            "name": "Dixon-Coles v1.0",
            "accuracy": 0.65,
            "created_at": "2024-01-01T00:00:00",
            "is_active": True
        }
        mock_service.return_value = mock_instance
        
        response = client.get("/api/models/active")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["version"] == "1.0.0"
        assert data["is_active"] is True
    
    @patch('app.services.ModelService')
    def test_get_all_models(self, mock_service, client):
        """Test get all models endpoint"""
        # Mock service response
        mock_instance = AsyncMock()
        mock_instance.get_all_models.return_value = [
            {
                "id": 1,
                "version": "1.0.0",
                "name": "Dixon-Coles v1.0",
                "is_active": True
            },
            {
                "id": 2,
                "version": "1.1.0",
                "name": "Dixon-Coles v1.1",
                "is_active": False
            }
        ]
        mock_service.return_value = mock_instance
        
        response = client.get("/api/models/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "models" in data
        assert len(data["models"]) == 2
    
    @patch('app.services.ModelService')
    def test_get_model_performance(self, mock_service, client):
        """Test get model performance endpoint"""
        # Mock service response
        mock_instance = AsyncMock()
        mock_instance.get_model_performance.return_value = {
            "accuracy": 0.65,
            "precision": 0.62,
            "recall": 0.68,
            "f1_score": 0.65,
            "brier_score": 0.25,
            "log_loss": 0.55,
            "calibration_curve": []
        }
        mock_service.return_value = mock_instance
        
        response = client.get("/api/models/1/performance")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "accuracy" in data
        assert "brier_score" in data
        assert "log_loss" in data
    
    @patch('app.services.ModelService')
    def test_start_retraining(self, mock_service, client):
        """Test start model retraining endpoint"""
        # Mock service response
        mock_instance = AsyncMock()
        mock_instance.start_retraining.return_value = {
            "task_id": "retrain-123",
            "status": "started",
            "message": "Retraining started successfully"
        }
        mock_service.return_value = mock_instance
        
        response = client.post("/api/models/retrain")
        
        assert response.status_code == 202  # Accepted
        data = response.json()
        
        assert "task_id" in data
        assert "status" in data
    
    @patch('app.services.ModelService')
    def test_get_retraining_status(self, mock_service, client):
        """Test get retraining status endpoint"""
        # Mock service response
        mock_instance = AsyncMock()
        mock_instance.get_retraining_status.return_value = {
            "task_id": "retrain-123",
            "status": "running",
            "progress": 75,
            "message": "Training in progress",
            "estimated_completion": "2024-01-01T12:00:00"
        }
        mock_service.return_value = mock_instance
        
        response = client.get("/api/models/retrain/retrain-123")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["task_id"] == "retrain-123"
        assert data["status"] == "running"
        assert data["progress"] == 75
    
    @patch('app.services.ModelService')
    def test_activate_model(self, mock_service, client):
        """Test activate model endpoint"""
        # Mock service response
        mock_instance = AsyncMock()
        mock_instance.activate_model.return_value = {
            "id": 2,
            "version": "1.1.0",
            "is_active": True,
            "message": "Model activated successfully"
        }
        mock_service.return_value = mock_instance
        
        response = client.post("/api/models/2/activate")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == 2
        assert data["is_active"] is True
    
    @patch('app.services.ModelService')
    def test_delete_model(self, mock_service, client):
        """Test delete model endpoint"""
        # Mock service response
        mock_instance = AsyncMock()
        mock_instance.delete_model.return_value = {
            "message": "Model deleted successfully"
        }
        mock_service.return_value = mock_instance
        
        response = client.delete("/api/models/2")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data

class TestAPIErrorHandling:
    """Test API error handling"""
    
    def test_404_not_found(self, client):
        """Test 404 error handling"""
        response = client.get("/api/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        
        assert "detail" in data
    
    def test_method_not_allowed(self, client):
        """Test 405 error handling"""
        response = client.patch("/api/health/")  # PATCH not allowed
        
        assert response.status_code == 405
    
    @patch('app.services.PredictionService')
    def test_internal_server_error(self, mock_service, client):
        """Test 500 error handling"""
        # Mock service to raise exception
        mock_instance = AsyncMock()
        mock_instance.predict_match.side_effect = Exception("Database connection failed")
        mock_service.return_value = mock_instance
        
        response = client.post(
            "/api/predictions/predict",
            json={
                "home_team": "Real Madrid",
                "away_team": "FC Barcelona"
            }
        )
        
        assert response.status_code == 500
        data = response.json()
        
        assert "detail" in data
    
    def test_validation_error(self, client):
        """Test validation error handling"""
        response = client.post(
            "/api/predictions/predict",
            json={
                "home_team": 123,  # Should be string
                "away_team": "FC Barcelona"
            }
        )
        
        assert response.status_code == 422
        data = response.json()
        
        assert "detail" in data
        assert isinstance(data["detail"], list)

class TestAPIAuthentication:
    """Test API authentication (if implemented)"""
    
    @pytest.mark.skip(reason="Authentication not implemented yet")
    def test_protected_endpoint_without_auth(self, client):
        """Test accessing protected endpoint without authentication"""
        response = client.post("/api/models/retrain")
        
        assert response.status_code == 401
    
    @pytest.mark.skip(reason="Authentication not implemented yet")
    def test_protected_endpoint_with_invalid_token(self, client):
        """Test accessing protected endpoint with invalid token"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.post("/api/models/retrain", headers=headers)
        
        assert response.status_code == 401

class TestAPIRateLimiting:
    """Test API rate limiting (if implemented)"""
    
    @pytest.mark.skip(reason="Rate limiting not implemented yet")
    def test_rate_limit_exceeded(self, client):
        """Test rate limiting"""
        # Make multiple requests quickly
        for _ in range(100):
            response = client.get("/api/health/")
            if response.status_code == 429:
                break
        
        assert response.status_code == 429
        data = response.json()
        
        assert "detail" in data
        assert "rate limit" in data["detail"].lower()