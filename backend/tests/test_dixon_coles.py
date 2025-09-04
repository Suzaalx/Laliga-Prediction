import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from app.utils.dixon_coles import DixonColesModel, DCParameters

class TestDCParameters:
    """Test DCParameters dataclass"""
    
    def test_dc_parameters_creation(self):
        """Test creating DCParameters with default values"""
        params = DCParameters()
        
        assert params.home_advantage == 0.0
        assert params.rho == 0.0
        assert params.gamma == 0.0
        assert len(params.attack_params) == 0
        assert len(params.defense_params) == 0
        assert params.created_at is not None
    
    def test_dc_parameters_with_values(self):
        """Test creating DCParameters with specific values"""
        attack_params = {"Team A": 0.1, "Team B": -0.05}
        defense_params = {"Team A": -0.02, "Team B": 0.08}
        
        params = DCParameters(
            home_advantage=0.3,
            rho=0.15,
            gamma=1.2,
            attack_params=attack_params,
            defense_params=defense_params
        )
        
        assert params.home_advantage == 0.3
        assert params.rho == 0.15
        assert params.gamma == 1.2
        assert params.attack_params == attack_params
        assert params.defense_params == defense_params
    
    def test_to_dict(self):
        """Test converting DCParameters to dictionary"""
        params = DCParameters(
            home_advantage=0.3,
            rho=0.15,
            attack_params={"Team A": 0.1},
            defense_params={"Team A": -0.02}
        )
        
        params_dict = params.to_dict()
        
        assert params_dict["home_advantage"] == 0.3
        assert params_dict["rho"] == 0.15
        assert "attack_params" in params_dict
        assert "defense_params" in params_dict
        assert "created_at" in params_dict
    
    def test_from_dict(self):
        """Test creating DCParameters from dictionary"""
        params_dict = {
            "home_advantage": 0.3,
            "rho": 0.15,
            "gamma": 1.2,
            "attack_params": {"Team A": 0.1},
            "defense_params": {"Team A": -0.02},
            "created_at": datetime.now().isoformat()
        }
        
        params = DCParameters.from_dict(params_dict)
        
        assert params.home_advantage == 0.3
        assert params.rho == 0.15
        assert params.gamma == 1.2
        assert params.attack_params == {"Team A": 0.1}
        assert params.defense_params == {"Team A": -0.02}

class TestDixonColesModel:
    """Test DixonColesModel class"""
    
    @pytest.fixture
    def sample_matches_df(self):
        """Create sample matches DataFrame for testing"""
        data = {
            'date': [
                datetime(2024, 1, 1),
                datetime(2024, 1, 8),
                datetime(2024, 1, 15),
                datetime(2024, 1, 22),
                datetime(2024, 1, 29)
            ],
            'home_team': ['Team A', 'Team B', 'Team A', 'Team C', 'Team B'],
            'away_team': ['Team B', 'Team C', 'Team C', 'Team A', 'Team A'],
            'home_goals': [2, 1, 3, 0, 1],
            'away_goals': [1, 2, 0, 2, 1]
        }
        return pd.DataFrame(data)
    
    def test_model_initialization(self):
        """Test model initialization with default parameters"""
        model = DixonColesModel()
        
        assert model.xi == 0.001
        assert model.tau == 0.01
        assert model.max_iterations == 100
        assert model.convergence_threshold == 1e-6
        assert model.parameters is None
        assert model.is_fitted is False
    
    def test_model_initialization_with_params(self):
        """Test model initialization with custom parameters"""
        model = DixonColesModel(
            xi=0.002,
            tau=0.02,
            max_iterations=200,
            convergence_threshold=1e-5
        )
        
        assert model.xi == 0.002
        assert model.tau == 0.02
        assert model.max_iterations == 200
        assert model.convergence_threshold == 1e-5
    
    def test_time_weights_calculation(self, sample_matches_df):
        """Test time weights calculation"""
        model = DixonColesModel(xi=0.01)
        
        # Set current date for weight calculation
        current_date = datetime(2024, 2, 1)
        weights = model._calculate_time_weights(sample_matches_df, current_date)
        
        assert len(weights) == len(sample_matches_df)
        assert all(w > 0 for w in weights)
        # More recent matches should have higher weights
        assert weights[4] > weights[0]  # Most recent vs oldest
    
    def test_low_score_correction(self):
        """Test low score correction factor calculation"""
        model = DixonColesModel()
        
        # Test specific score combinations
        assert model._low_score_correction(0, 0, 0.1) != 1.0  # Should apply correction
        assert model._low_score_correction(1, 0, 0.1) != 1.0  # Should apply correction
        assert model._low_score_correction(0, 1, 0.1) != 1.0  # Should apply correction
        assert model._low_score_correction(1, 1, 0.1) != 1.0  # Should apply correction
        assert model._low_score_correction(2, 1, 0.1) == 1.0  # No correction
        assert model._low_score_correction(0, 2, 0.1) == 1.0  # No correction
    
    def test_expected_goals_calculation(self):
        """Test expected goals calculation"""
        model = DixonColesModel()
        
        # Create mock parameters
        model.parameters = DCParameters(
            home_advantage=0.3,
            gamma=1.2,
            attack_params={"Team A": 0.1, "Team B": -0.05},
            defense_params={"Team A": -0.02, "Team B": 0.08}
        )
        model.is_fitted = True
        
        home_goals, away_goals = model._calculate_expected_goals("Team A", "Team B")
        
        assert home_goals > 0
        assert away_goals > 0
        assert isinstance(home_goals, float)
        assert isinstance(away_goals, float)
    
    def test_expected_goals_unknown_team(self):
        """Test expected goals calculation with unknown team"""
        model = DixonColesModel()
        
        # Create mock parameters
        model.parameters = DCParameters(
            home_advantage=0.3,
            gamma=1.2,
            attack_params={"Team A": 0.1},
            defense_params={"Team A": -0.02}
        )
        model.is_fitted = True
        
        home_goals, away_goals = model._calculate_expected_goals("Team A", "Unknown Team")
        
        assert home_goals > 0
        assert away_goals > 0
    
    def test_scoreline_probability(self):
        """Test scoreline probability calculation"""
        model = DixonColesModel()
        
        # Create mock parameters
        model.parameters = DCParameters(
            home_advantage=0.3,
            rho=0.1,
            gamma=1.2,
            attack_params={"Team A": 0.1, "Team B": -0.05},
            defense_params={"Team A": -0.02, "Team B": 0.08}
        )
        model.is_fitted = True
        
        prob = model.calculate_scoreline_probability("Team A", "Team B", 1, 1)
        
        assert 0 <= prob <= 1
        assert isinstance(prob, float)
    
    def test_match_probabilities(self):
        """Test match outcome probabilities calculation"""
        model = DixonColesModel()
        
        # Create mock parameters
        model.parameters = DCParameters(
            home_advantage=0.3,
            rho=0.1,
            gamma=1.2,
            attack_params={"Team A": 0.1, "Team B": -0.05},
            defense_params={"Team A": -0.02, "Team B": 0.08}
        )
        model.is_fitted = True
        
        probs = model.predict_match("Team A", "Team B")
        
        assert "home_win" in probs
        assert "draw" in probs
        assert "away_win" in probs
        assert "expected_home_goals" in probs
        assert "expected_away_goals" in probs
        
        # Probabilities should sum to approximately 1
        total_prob = probs["home_win"] + probs["draw"] + probs["away_win"]
        assert abs(total_prob - 1.0) < 0.01
        
        # All probabilities should be between 0 and 1
        assert 0 <= probs["home_win"] <= 1
        assert 0 <= probs["draw"] <= 1
        assert 0 <= probs["away_win"] <= 1
        assert probs["expected_home_goals"] >= 0
        assert probs["expected_away_goals"] >= 0
    
    def test_predict_match_not_fitted(self):
        """Test prediction with unfitted model"""
        model = DixonColesModel()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict_match("Team A", "Team B")
    
    def test_scoreline_matrix(self):
        """Test scoreline probability matrix generation"""
        model = DixonColesModel()
        
        # Create mock parameters
        model.parameters = DCParameters(
            home_advantage=0.3,
            rho=0.1,
            gamma=1.2,
            attack_params={"Team A": 0.1, "Team B": -0.05},
            defense_params={"Team A": -0.02, "Team B": 0.08}
        )
        model.is_fitted = True
        
        matrix = model.get_scoreline_matrix("Team A", "Team B", max_goals=3)
        
        assert matrix.shape == (4, 4)  # 0-3 goals for each team
        assert np.all(matrix >= 0)
        assert np.all(matrix <= 1)
        # Matrix should sum to approximately 1 (allowing for truncation)
        assert matrix.sum() > 0.8
    
    @patch('app.utils.dixon_coles.minimize')
    def test_fit_model(self, mock_minimize, sample_matches_df):
        """Test model fitting process"""
        # Mock optimization result
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.x = np.array([0.3, 0.1, 1.2, 0.1, -0.05, -0.02, 0.08])  # Example parameters
        mock_minimize.return_value = mock_result
        
        model = DixonColesModel()
        model.fit(sample_matches_df)
        
        assert model.is_fitted
        assert model.parameters is not None
        assert mock_minimize.called
    
    @patch('app.utils.dixon_coles.minimize')
    def test_fit_model_failure(self, mock_minimize, sample_matches_df):
        """Test model fitting failure"""
        # Mock optimization failure
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.message = "Optimization failed"
        mock_minimize.return_value = mock_result
        
        model = DixonColesModel()
        
        with pytest.raises(RuntimeError, match="Model fitting failed"):
            model.fit(sample_matches_df)
    
    def test_save_load_parameters(self, tmp_path):
        """Test saving and loading model parameters"""
        model = DixonColesModel()
        
        # Create mock parameters
        model.parameters = DCParameters(
            home_advantage=0.3,
            rho=0.1,
            gamma=1.2,
            attack_params={"Team A": 0.1, "Team B": -0.05},
            defense_params={"Team A": -0.02, "Team B": 0.08}
        )
        model.is_fitted = True
        
        # Save parameters
        file_path = tmp_path / "test_params.json"
        model.save_parameters(str(file_path))
        
        # Create new model and load parameters
        new_model = DixonColesModel()
        new_model.load_parameters(str(file_path))
        
        assert new_model.is_fitted
        assert new_model.parameters.home_advantage == 0.3
        assert new_model.parameters.rho == 0.1
        assert new_model.parameters.gamma == 1.2
        assert new_model.parameters.attack_params == {"Team A": 0.1, "Team B": -0.05}
        assert new_model.parameters.defense_params == {"Team A": -0.02, "Team B": 0.08}
    
    def test_load_parameters_file_not_found(self):
        """Test loading parameters from non-existent file"""
        model = DixonColesModel()
        
        with pytest.raises(FileNotFoundError):
            model.load_parameters("non_existent_file.json")
    
    def test_model_evaluation_metrics(self):
        """Test model evaluation metrics calculation"""
        model = DixonColesModel()
        
        # Create mock parameters
        model.parameters = DCParameters(
            home_advantage=0.3,
            rho=0.1,
            gamma=1.2,
            attack_params={"Team A": 0.1, "Team B": -0.05, "Team C": 0.02},
            defense_params={"Team A": -0.02, "Team B": 0.08, "Team C": -0.01}
        )
        model.is_fitted = True
        
        # Create test data
        test_data = pd.DataFrame({
            'home_team': ['Team A', 'Team B'],
            'away_team': ['Team B', 'Team C'],
            'home_goals': [2, 1],
            'away_goals': [1, 1],
            'result': ['H', 'D']
        })
        
        metrics = model.evaluate(test_data)
        
        assert "accuracy" in metrics
        assert "log_loss" in metrics
        assert "brier_score" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["log_loss"] >= 0
        assert 0 <= metrics["brier_score"] <= 1
    
    def test_model_evaluation_not_fitted(self):
        """Test evaluation with unfitted model"""
        model = DixonColesModel()
        test_data = pd.DataFrame({
            'home_team': ['Team A'],
            'away_team': ['Team B'],
            'result': ['H']
        })
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.evaluate(test_data)
    
    def test_cross_validation(self, sample_matches_df):
        """Test cross-validation functionality"""
        model = DixonColesModel(max_iterations=10)  # Reduce iterations for testing
        
        with patch.object(model, 'fit') as mock_fit, \
             patch.object(model, 'evaluate') as mock_evaluate:
            
            # Mock the fit and evaluate methods
            mock_evaluate.return_value = {
                "accuracy": 0.6,
                "log_loss": 0.5,
                "brier_score": 0.3
            }
            
            cv_results = model.cross_validate(sample_matches_df, cv_folds=2)
            
            assert "mean_accuracy" in cv_results
            assert "std_accuracy" in cv_results
            assert "mean_log_loss" in cv_results
            assert "std_log_loss" in cv_results
            assert "mean_brier_score" in cv_results
            assert "std_brier_score" in cv_results
            
            # Check that fit was called for each fold
            assert mock_fit.call_count == 2
            assert mock_evaluate.call_count == 2
    
    def test_feature_importance(self):
        """Test feature importance calculation"""
        model = DixonColesModel()
        
        # Create mock parameters
        model.parameters = DCParameters(
            home_advantage=0.3,
            rho=0.1,
            attack_params={"Team A": 0.1, "Team B": -0.05, "Team C": 0.02},
            defense_params={"Team A": -0.02, "Team B": 0.08, "Team C": -0.01}
        )
        model.is_fitted = True
        
        importance = model.get_feature_importance()
        
        assert "home_advantage" in importance
        assert "rho" in importance
        assert "attack_strength" in importance
        assert "defense_strength" in importance
        
        # All importance values should be non-negative
        assert all(v >= 0 for v in importance.values())
    
    def test_model_summary(self):
        """Test model summary generation"""
        model = DixonColesModel()
        
        # Create mock parameters
        model.parameters = DCParameters(
            home_advantage=0.3,
            rho=0.1,
            gamma=1.2,
            attack_params={"Team A": 0.1, "Team B": -0.05},
            defense_params={"Team A": -0.02, "Team B": 0.08}
        )
        model.is_fitted = True
        
        summary = model.get_model_summary()
        
        assert "model_type" in summary
        assert "parameters" in summary
        assert "teams" in summary
        assert "fitted" in summary
        assert summary["fitted"] is True
        assert summary["teams"] == 2
        assert summary["model_type"] == "Dixon-Coles"