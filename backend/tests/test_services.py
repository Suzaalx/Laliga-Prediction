import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.services import (
    PredictionService,
    CacheService,
    TeamService,
    ModelService,
    DataService
)
from app.models.database import Team, Match, Model
from app.utils import DixonColesModel

class TestPredictionService:
    """Test PredictionService functionality"""
    
    @pytest.fixture
    def prediction_service(self, db_session):
        """Create PredictionService instance"""
        return PredictionService(db_session)
    
    @pytest.mark.asyncio
    async def test_fuzzy_match_team(self, prediction_service, sample_teams):
        """Test fuzzy team name matching"""
        # Test exact match
        result = await prediction_service.fuzzy_match_team("Real Madrid")
        assert result is not None
        assert result.name == "Real Madrid"
        
        # Test fuzzy match
        result = await prediction_service.fuzzy_match_team("Real")
        assert result is not None
        assert "Real" in result.name
        
        # Test no match
        result = await prediction_service.fuzzy_match_team("NonExistentTeam")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_team_stats(self, prediction_service, sample_teams, sample_matches):
        """Test team statistics calculation"""
        team = sample_teams[0]
        stats = await prediction_service.get_team_stats(team.id)
        
        assert "attack_strength" in stats
        assert "defense_strength" in stats
        assert "goals_scored" in stats
        assert "goals_conceded" in stats
        assert "shots_per_game" in stats
        assert "form_points" in stats
        
        # Check that values are reasonable
        assert stats["attack_strength"] >= 0
        assert stats["defense_strength"] >= 0
        assert stats["goals_scored"] >= 0
        assert stats["goals_conceded"] >= 0
    
    @pytest.mark.asyncio
    async def test_calculate_dixon_coles_probabilities(self, prediction_service):
        """Test Dixon-Coles probability calculation"""
        home_stats = {
            "attack_strength": 1.2,
            "defense_strength": 0.8
        }
        away_stats = {
            "attack_strength": 1.0,
            "defense_strength": 1.1
        }
        
        probabilities = await prediction_service.calculate_dixon_coles_probabilities(
            home_stats, away_stats
        )
        
        assert "home_win" in probabilities
        assert "draw" in probabilities
        assert "away_win" in probabilities
        assert "expected_home_goals" in probabilities
        assert "expected_away_goals" in probabilities
        
        # Check probabilities sum to 1
        total_prob = (
            probabilities["home_win"] + 
            probabilities["draw"] + 
            probabilities["away_win"]
        )
        assert abs(total_prob - 1.0) < 0.01
        
        # Check all probabilities are between 0 and 1
        for prob in [probabilities["home_win"], probabilities["draw"], probabilities["away_win"]]:
            assert 0 <= prob <= 1
    
    @pytest.mark.asyncio
    async def test_predict_match(self, prediction_service, sample_teams):
        """Test single match prediction"""
        home_team = sample_teams[0].name
        away_team = sample_teams[1].name
        
        prediction = await prediction_service.predict_match(home_team, away_team)
        
        assert "home_win_probability" in prediction
        assert "draw_probability" in prediction
        assert "away_win_probability" in prediction
        assert "expected_home_goals" in prediction
        assert "expected_away_goals" in prediction
        assert "confidence" in prediction
        assert "model_version" in prediction
        
        # Check probabilities sum to 1
        total_prob = (
            prediction["home_win_probability"] + 
            prediction["draw_probability"] + 
            prediction["away_win_probability"]
        )
        assert abs(total_prob - 1.0) < 0.01
    
    @pytest.mark.asyncio
    async def test_predict_batch(self, prediction_service, sample_teams):
        """Test batch predictions"""
        matches = [
            {
                "home_team": sample_teams[0].name,
                "away_team": sample_teams[1].name,
                "match_date": datetime.now() + timedelta(days=1)
            },
            {
                "home_team": sample_teams[1].name,
                "away_team": sample_teams[0].name,
                "match_date": datetime.now() + timedelta(days=2)
            }
        ]
        
        predictions = await prediction_service.predict_batch(matches)
        
        assert len(predictions) == 2
        for prediction in predictions:
            assert "home_win_probability" in prediction
            assert "draw_probability" in prediction
            assert "away_win_probability" in prediction
    
    @pytest.mark.asyncio
    async def test_get_upcoming_matches(self, prediction_service, sample_matches):
        """Test getting upcoming matches"""
        matches = await prediction_service.get_upcoming_matches(days=7)
        
        assert isinstance(matches, list)
        # All matches should be in the future
        for match in matches:
            assert match["match_date"] > datetime.now()
    
    @pytest.mark.asyncio
    async def test_get_prediction_history(self, prediction_service):
        """Test getting prediction history"""
        history = await prediction_service.get_prediction_history(limit=10)
        
        assert isinstance(history, list)
        assert len(history) <= 10
        
        for prediction in history:
            assert "id" in prediction
            assert "match_date" in prediction
            assert "home_team" in prediction
            assert "away_team" in prediction
    
    @pytest.mark.asyncio
    async def test_get_prediction_stats(self, prediction_service):
        """Test getting prediction statistics"""
        stats = await prediction_service.get_prediction_stats()
        
        assert "total_predictions" in stats
        assert "accuracy" in stats
        assert "brier_score" in stats
        assert "log_loss" in stats
        assert "by_month" in stats
        
        # Check that values are reasonable
        assert stats["total_predictions"] >= 0
        assert 0 <= stats["accuracy"] <= 1
        assert stats["brier_score"] >= 0
        assert stats["log_loss"] >= 0

class TestCacheService:
    """Test CacheService functionality"""
    
    @pytest.fixture
    def cache_service(self):
        """Create CacheService instance with mock Redis"""
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_instance = AsyncMock()
            mock_redis.return_value = mock_instance
            service = CacheService()
            service.redis = mock_instance
            return service, mock_instance
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self, cache_service):
        """Test Redis connection and disconnection"""
        service, mock_redis = cache_service
        
        await service.connect()
        mock_redis.ping.assert_called_once()
        
        await service.disconnect()
        mock_redis.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_set_get_operations(self, cache_service):
        """Test basic set and get operations"""
        service, mock_redis = cache_service
        
        # Mock Redis responses
        mock_redis.set.return_value = True
        mock_redis.get.return_value = b'{"test": "data"}'
        
        # Test set
        result = await service.set("test_key", {"test": "data"}, ttl=300)
        assert result is True
        mock_redis.set.assert_called_once()
        
        # Test get
        result = await service.get("test_key")
        assert result == {"test": "data"}
        mock_redis.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_operations(self, cache_service):
        """Test delete operations"""
        service, mock_redis = cache_service
        
        # Mock Redis responses
        mock_redis.delete.return_value = 1
        mock_redis.scan_iter.return_value = [b'key1', b'key2']
        
        # Test single delete
        result = await service.delete("test_key")
        assert result is True
        mock_redis.delete.assert_called_once()
        
        # Test pattern delete
        result = await service.delete_pattern("test_*")
        assert result == 2
    
    @pytest.mark.asyncio
    async def test_multiple_operations(self, cache_service):
        """Test multiple key operations"""
        service, mock_redis = cache_service
        
        # Mock Redis responses
        mock_redis.mset.return_value = True
        mock_redis.mget.return_value = [b'{"key1": "value1"}', b'{"key2": "value2"}']
        
        # Test mset
        data = {"key1": {"key1": "value1"}, "key2": {"key2": "value2"}}
        result = await service.mset(data, ttl=300)
        assert result is True
        
        # Test mget
        result = await service.mget(["key1", "key2"])
        assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_ttl_operations(self, cache_service):
        """Test TTL operations"""
        service, mock_redis = cache_service
        
        # Mock Redis responses
        mock_redis.ttl.return_value = 300
        mock_redis.expire.return_value = True
        
        # Test get TTL
        result = await service.get_ttl("test_key")
        assert result == 300
        
        # Test set TTL
        result = await service.set_ttl("test_key", 600)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_increment_operations(self, cache_service):
        """Test increment operations"""
        service, mock_redis = cache_service
        
        # Mock Redis responses
        mock_redis.incr.return_value = 5
        mock_redis.incrby.return_value = 15
        
        # Test increment
        result = await service.increment("counter")
        assert result == 5
        
        # Test increment by amount
        result = await service.increment("counter", amount=10)
        assert result == 15
    
    @pytest.mark.asyncio
    async def test_health_check(self, cache_service):
        """Test cache health check"""
        service, mock_redis = cache_service
        
        # Mock Redis responses
        mock_redis.ping.return_value = True
        mock_redis.info.return_value = {
            'used_memory': 1024000,
            'connected_clients': 5,
            'total_commands_processed': 1000
        }
        
        health = await service.get_health()
        
        assert health["status"] == "healthy"
        assert "memory_usage" in health
        assert "connected_clients" in health
        assert "total_commands" in health
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, cache_service):
        """Test cache statistics"""
        service, mock_redis = cache_service
        
        # Mock Redis responses
        mock_redis.info.return_value = {
            'keyspace_hits': 1000,
            'keyspace_misses': 100,
            'used_memory': 2048000,
            'connected_clients': 10
        }
        
        stats = await service.get_stats()
        
        assert "hit_rate" in stats
        assert "memory_usage" in stats
        assert "connected_clients" in stats
        assert stats["hit_rate"] == 0.909  # 1000 / (1000 + 100)

class TestTeamService:
    """Test TeamService functionality"""
    
    @pytest.fixture
    def team_service(self, db_session):
        """Create TeamService instance"""
        return TeamService(db_session)
    
    @pytest.mark.asyncio
    async def test_get_all_teams(self, team_service, sample_teams):
        """Test getting all teams"""
        teams = await team_service.get_all_teams()
        
        assert len(teams) == len(sample_teams)
        for team in teams:
            assert "id" in team
            assert "name" in team
            assert "short_name" in team
    
    @pytest.mark.asyncio
    async def test_get_team_by_id(self, team_service, sample_teams):
        """Test getting team by ID"""
        team_id = sample_teams[0].id
        team = await team_service.get_team_by_id(team_id)
        
        assert team is not None
        assert team["id"] == team_id
        assert team["name"] == sample_teams[0].name
    
    @pytest.mark.asyncio
    async def test_get_team_by_name(self, team_service, sample_teams):
        """Test getting team by name"""
        team_name = sample_teams[0].name
        team = await team_service.get_team_by_name(team_name)
        
        assert team is not None
        assert team["name"] == team_name
    
    @pytest.mark.asyncio
    async def test_get_team_stats(self, team_service, sample_teams, sample_matches):
        """Test getting team statistics"""
        team_id = sample_teams[0].id
        stats = await team_service.get_team_stats(team_id)
        
        assert "matches_played" in stats
        assert "wins" in stats
        assert "draws" in stats
        assert "losses" in stats
        assert "goals_scored" in stats
        assert "goals_conceded" in stats
        assert "points" in stats
        
        # Check that values are reasonable
        assert stats["matches_played"] >= 0
        assert stats["wins"] >= 0
        assert stats["draws"] >= 0
        assert stats["losses"] >= 0
        assert stats["wins"] + stats["draws"] + stats["losses"] == stats["matches_played"]
    
    @pytest.mark.asyncio
    async def test_get_team_form(self, team_service, sample_teams, sample_matches):
        """Test getting team form"""
        team_id = sample_teams[0].id
        form = await team_service.get_team_form(team_id, matches=5)
        
        assert "recent_results" in form
        assert "form_points" in form
        assert "matches_analyzed" in form
        
        # Check that recent results are valid
        for result in form["recent_results"]:
            assert result in ["W", "D", "L"]
        
        # Check that form points are reasonable
        assert 0 <= form["form_points"] <= 3
    
    @pytest.mark.asyncio
    async def test_get_team_matches(self, team_service, sample_teams, sample_matches):
        """Test getting team matches"""
        team_id = sample_teams[0].id
        matches = await team_service.get_team_matches(team_id, limit=10)
        
        assert isinstance(matches, list)
        assert len(matches) <= 10
        
        for match in matches:
            assert "id" in match
            assert "date" in match
            assert "home_team" in match
            assert "away_team" in match
            assert "result" in match
    
    @pytest.mark.asyncio
    async def test_get_head_to_head(self, team_service, sample_teams, sample_matches):
        """Test getting head-to-head statistics"""
        team1_id = sample_teams[0].id
        team2_id = sample_teams[1].id
        
        h2h = await team_service.get_head_to_head(team1_id, team2_id)
        
        assert "total_matches" in h2h
        assert "team1_wins" in h2h
        assert "team2_wins" in h2h
        assert "draws" in h2h
        assert "recent_matches" in h2h
        
        # Check that totals add up
        total = h2h["team1_wins"] + h2h["team2_wins"] + h2h["draws"]
        assert total == h2h["total_matches"]
    
    @pytest.mark.asyncio
    async def test_search_teams(self, team_service, sample_teams):
        """Test team search functionality"""
        # Test exact match
        results = await team_service.search_teams("Real Madrid")
        assert len(results) >= 1
        assert any(team["name"] == "Real Madrid" for team in results)
        
        # Test partial match
        results = await team_service.search_teams("Real")
        assert len(results) >= 1
        assert any("Real" in team["name"] for team in results)
        
        # Test no match
        results = await team_service.search_teams("NonExistentTeam")
        assert len(results) == 0

class TestModelService:
    """Test ModelService functionality"""
    
    @pytest.fixture
    def model_service(self, db_session):
        """Create ModelService instance"""
        return ModelService(db_session)
    
    @pytest.mark.asyncio
    async def test_get_active_model(self, model_service, db_session):
        """Test getting active model"""
        # Create a test model
        model = Model(
            version="1.0.0",
            name="Test Model",
            model_type="dixon_coles",
            parameters={"test": "params"},
            accuracy=0.65,
            is_active=True
        )
        db_session.add(model)
        db_session.commit()
        
        active_model = await model_service.get_active_model()
        
        assert active_model is not None
        assert active_model["version"] == "1.0.0"
        assert active_model["is_active"] is True
    
    @pytest.mark.asyncio
    async def test_get_all_models(self, model_service, db_session):
        """Test getting all models"""
        # Create test models
        models = [
            Model(
                version="1.0.0",
                name="Test Model 1",
                model_type="dixon_coles",
                parameters={"test": "params1"},
                accuracy=0.65,
                is_active=True
            ),
            Model(
                version="1.1.0",
                name="Test Model 2",
                model_type="dixon_coles",
                parameters={"test": "params2"},
                accuracy=0.67,
                is_active=False
            )
        ]
        for model in models:
            db_session.add(model)
        db_session.commit()
        
        all_models = await model_service.get_all_models()
        
        assert len(all_models) >= 2
        versions = [model["version"] for model in all_models]
        assert "1.0.0" in versions
        assert "1.1.0" in versions
    
    @pytest.mark.asyncio
    async def test_get_model_by_id(self, model_service, db_session):
        """Test getting model by ID"""
        # Create a test model
        model = Model(
            version="1.0.0",
            name="Test Model",
            model_type="dixon_coles",
            parameters={"test": "params"},
            accuracy=0.65,
            is_active=True
        )
        db_session.add(model)
        db_session.commit()
        
        retrieved_model = await model_service.get_model_by_id(model.id)
        
        assert retrieved_model is not None
        assert retrieved_model["id"] == model.id
        assert retrieved_model["version"] == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_get_model_performance(self, model_service, db_session):
        """Test getting model performance metrics"""
        # Create a test model
        model = Model(
            version="1.0.0",
            name="Test Model",
            model_type="dixon_coles",
            parameters={"test": "params"},
            accuracy=0.65,
            precision=0.62,
            recall=0.68,
            f1_score=0.65,
            brier_score=0.25,
            log_loss=0.55,
            is_active=True
        )
        db_session.add(model)
        db_session.commit()
        
        performance = await model_service.get_model_performance(model.id)
        
        assert "accuracy" in performance
        assert "precision" in performance
        assert "recall" in performance
        assert "f1_score" in performance
        assert "brier_score" in performance
        assert "log_loss" in performance
        assert "calibration_curve" in performance
        
        assert performance["accuracy"] == 0.65
        assert performance["precision"] == 0.62
    
    @pytest.mark.asyncio
    async def test_start_retraining(self, model_service):
        """Test starting model retraining"""
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value.hex = "test-task-id"
            
            result = await model_service.start_retraining()
            
            assert "task_id" in result
            assert "status" in result
            assert "message" in result
            assert result["task_id"] == "retrain-test-task-id"
            assert result["status"] == "started"
    
    @pytest.mark.asyncio
    async def test_get_retraining_status(self, model_service):
        """Test getting retraining status"""
        # Start a retraining task first
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value.hex = "test-task-id"
            await model_service.start_retraining()
            
            status = await model_service.get_retraining_status("retrain-test-task-id")
            
            assert "task_id" in status
            assert "status" in status
            assert "progress" in status
            assert "message" in status
            assert status["task_id"] == "retrain-test-task-id"
    
    @pytest.mark.asyncio
    async def test_activate_model(self, model_service, db_session):
        """Test activating a model"""
        # Create test models
        model1 = Model(
            version="1.0.0",
            name="Test Model 1",
            model_type="dixon_coles",
            parameters={"test": "params1"},
            accuracy=0.65,
            is_active=True
        )
        model2 = Model(
            version="1.1.0",
            name="Test Model 2",
            model_type="dixon_coles",
            parameters={"test": "params2"},
            accuracy=0.67,
            is_active=False
        )
        db_session.add_all([model1, model2])
        db_session.commit()
        
        result = await model_service.activate_model(model2.id)
        
        assert "id" in result
        assert "version" in result
        assert "is_active" in result
        assert "message" in result
        assert result["id"] == model2.id
        assert result["is_active"] is True
        
        # Check that the previous active model is now inactive
        db_session.refresh(model1)
        assert model1.is_active is False
    
    @pytest.mark.asyncio
    async def test_delete_model(self, model_service, db_session):
        """Test deleting a model"""
        # Create a test model
        model = Model(
            version="1.0.0",
            name="Test Model",
            model_type="dixon_coles",
            parameters={"test": "params"},
            accuracy=0.65,
            is_active=False  # Can't delete active model
        )
        db_session.add(model)
        db_session.commit()
        model_id = model.id
        
        result = await model_service.delete_model(model_id)
        
        assert "message" in result
        
        # Check that model is deleted
        deleted_model = db_session.query(Model).filter(Model.id == model_id).first()
        assert deleted_model is None
    
    @pytest.mark.asyncio
    async def test_delete_active_model_fails(self, model_service, db_session):
        """Test that deleting active model fails"""
        # Create an active model
        model = Model(
            version="1.0.0",
            name="Test Model",
            model_type="dixon_coles",
            parameters={"test": "params"},
            accuracy=0.65,
            is_active=True
        )
        db_session.add(model)
        db_session.commit()
        
        with pytest.raises(ValueError, match="Cannot delete active model"):
            await model_service.delete_model(model.id)

class TestDataService:
    """Test DataService functionality"""
    
    @pytest.fixture
    def data_service(self, db_session):
        """Create DataService instance"""
        return DataService(db_session)
    
    @pytest.mark.asyncio
    async def test_ingest_csv_data(self, data_service, sample_csv_data):
        """Test CSV data ingestion"""
        result = await data_service.ingest_csv_data(sample_csv_data, "2023-24")
        
        assert "matches_processed" in result
        assert "teams_created" in result
        assert "errors" in result
        assert "summary" in result
        
        assert result["matches_processed"] > 0
        assert result["teams_created"] > 0
        assert len(result["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_process_match_data(self, data_service):
        """Test match data processing"""
        raw_match = {
            "Date": "15/08/2023",
            "HomeTeam": "Real Madrid",
            "AwayTeam": "Athletic Bilbao",
            "FTHG": "2",
            "FTAG": "1",
            "FTR": "H",
            "HTHG": "1",
            "HTAG": "0",
            "HTR": "H",
            "HS": "15",
            "AS": "8",
            "HST": "6",
            "AST": "3",
            "HC": "7",
            "AC": "4",
            "HF": "12",
            "AF": "18",
            "HY": "2",
            "AY": "3",
            "HR": "0",
            "AR": "1",
            "Referee": "Antonio Mateu Lahoz"
        }
        
        processed = await data_service.process_match_data(raw_match)
        
        assert "date" in processed
        assert "home_team" in processed
        assert "away_team" in processed
        assert "home_goals" in processed
        assert "away_goals" in processed
        assert "result" in processed
        assert "home_shots" in processed
        assert "away_shots" in processed
        assert "referee" in processed
        
        assert processed["home_goals"] == 2
        assert processed["away_goals"] == 1
        assert processed["result"] == "H"
    
    @pytest.mark.asyncio
    async def test_standardize_team_name(self, data_service):
        """Test team name standardization"""
        # Test various team name formats
        test_cases = [
            ("Real Madrid", "Real Madrid"),
            ("real madrid", "Real Madrid"),
            ("REAL MADRID", "Real Madrid"),
            ("Real  Madrid", "Real Madrid"),  # Extra spaces
            ("Real Madrid CF", "Real Madrid"),  # Remove CF suffix
            ("FC Barcelona", "Barcelona"),  # Remove FC prefix
        ]
        
        for input_name, expected in test_cases:
            result = await data_service.standardize_team_name(input_name)
            assert result == expected
    
    @pytest.mark.asyncio
    async def test_get_or_create_team(self, data_service, db_session):
        """Test team creation and retrieval"""
        # Test creating new team
        team1 = await data_service.get_or_create_team("New Team")
        assert team1.name == "New Team"
        
        # Test retrieving existing team
        team2 = await data_service.get_or_create_team("New Team")
        assert team1.id == team2.id
        
        # Check that only one team was created
        teams = db_session.query(Team).filter(Team.name == "New Team").all()
        assert len(teams) == 1
    
    @pytest.mark.asyncio
    async def test_calculate_team_stats(self, data_service, sample_teams, sample_matches):
        """Test team statistics calculation"""
        team = sample_teams[0]
        stats = await data_service.calculate_team_stats(team.id, "2023-24")
        
        assert "matches_played" in stats
        assert "wins" in stats
        assert "draws" in stats
        assert "losses" in stats
        assert "goals_scored" in stats
        assert "goals_conceded" in stats
        assert "shots_per_game" in stats
        assert "shots_on_target_per_game" in stats
        assert "fouls_per_game" in stats
        assert "cards_per_game" in stats
        
        # Check that values are reasonable
        assert stats["matches_played"] >= 0
        assert stats["wins"] + stats["draws"] + stats["losses"] == stats["matches_played"]
        assert stats["goals_scored"] >= 0
        assert stats["goals_conceded"] >= 0
    
    @pytest.mark.asyncio
    async def test_get_data_summary(self, data_service, sample_matches):
        """Test data summary generation"""
        summary = await data_service.get_data_summary()
        
        assert "total_matches" in summary
        assert "total_teams" in summary
        assert "seasons" in summary
        assert "date_range" in summary
        assert "data_quality" in summary
        
        assert summary["total_matches"] >= 0
        assert summary["total_teams"] >= 0
        assert isinstance(summary["seasons"], list)
        assert "start_date" in summary["date_range"]
        assert "end_date" in summary["date_range"]
    
    @pytest.mark.asyncio
    async def test_update_team_statistics(self, data_service, sample_teams, sample_matches):
        """Test team statistics update"""
        team = sample_teams[0]
        
        # Update statistics
        await data_service.update_team_statistics(team.id, "2023-24")
        
        # Check that team statistics were updated
        # This would typically update fields in the Team model
        # For now, we just check that the method runs without error
        assert True
    
    @pytest.mark.asyncio
    async def test_calculate_referee_stats(self, data_service, sample_matches):
        """Test referee statistics calculation"""
        referee_name = "Test Referee"
        stats = await data_service.calculate_referee_stats(referee_name, "2023-24")
        
        assert "matches_officiated" in stats
        assert "avg_fouls_per_match" in stats
        assert "avg_cards_per_match" in stats
        assert "avg_goals_per_match" in stats
        assert "home_win_rate" in stats
        
        # Check that values are reasonable
        assert stats["matches_officiated"] >= 0
        assert stats["avg_fouls_per_match"] >= 0
        assert stats["avg_cards_per_match"] >= 0
        assert 0 <= stats["home_win_rate"] <= 1