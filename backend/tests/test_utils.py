import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from app.utils.feature_engineering import FeatureEngineer, TeamFeatures
from app.utils.validation import (
    MatchDataValidator,
    TeamValidator,
    validate_match_data,
    validate_team_name,
    validate_csv_data,
    validate_prediction_request
)
from app.utils.helpers import (
    fuzzy_match_team,
    clean_team_name,
    calculate_form_points,
    get_rolling_average,
    calculate_elo_rating,
    calculate_poisson_probability,
    calculate_match_importance,
    normalize_features,
    calculate_team_strength,
    calculate_head_to_head_stats,
    safe_divide,
    clamp,
    exponential_decay_weights,
    linear_decay_weights
)

class TestFeatureEngineering:
    """Test FeatureEngineer functionality"""
    
    @pytest.fixture
    def feature_engineer(self, db_session):
        """Create FeatureEngineer instance"""
        return FeatureEngineer(db_session)
    
    @pytest.fixture
    def sample_match_data(self):
        """Create sample match data for testing"""
        return pd.DataFrame([
            {
                'date': datetime(2023, 8, 15),
                'home_team_id': 1,
                'away_team_id': 2,
                'home_goals': 2,
                'away_goals': 1,
                'home_shots': 15,
                'away_shots': 8,
                'home_shots_on_target': 6,
                'away_shots_on_target': 3,
                'home_corners': 7,
                'away_corners': 4,
                'home_fouls': 12,
                'away_fouls': 18,
                'home_yellow_cards': 2,
                'away_yellow_cards': 3,
                'home_red_cards': 0,
                'away_red_cards': 1,
                'referee': 'Test Referee'
            },
            {
                'date': datetime(2023, 8, 22),
                'home_team_id': 2,
                'away_team_id': 1,
                'home_goals': 1,
                'away_goals': 3,
                'home_shots': 10,
                'away_shots': 18,
                'home_shots_on_target': 4,
                'away_shots_on_target': 8,
                'home_corners': 5,
                'away_corners': 9,
                'home_fouls': 15,
                'away_fouls': 10,
                'home_yellow_cards': 3,
                'away_yellow_cards': 1,
                'home_red_cards': 1,
                'away_red_cards': 0,
                'referee': 'Test Referee'
            }
        ])
    
    def test_team_features_creation(self):
        """Test TeamFeatures dataclass creation"""
        features = TeamFeatures(
            goals_scored_avg=2.5,
            goals_conceded_avg=1.2,
            shots_avg=15.0,
            shots_on_target_avg=6.0,
            form_points=2.4,
            home_advantage=0.3
        )
        
        assert features.goals_scored_avg == 2.5
        assert features.goals_conceded_avg == 1.2
        assert features.shots_avg == 15.0
        assert features.form_points == 2.4
        assert features.home_advantage == 0.3
    
    def test_calculate_basic_stats(self, feature_engineer, sample_match_data):
        """Test basic statistics calculation"""
        with patch.object(feature_engineer, '_get_team_matches') as mock_get_matches:
            mock_get_matches.return_value = sample_match_data
            
            stats = feature_engineer._calculate_basic_stats(1, datetime(2023, 9, 1), 10)
            
            assert 'goals_scored_avg' in stats
            assert 'goals_conceded_avg' in stats
            assert 'shots_avg' in stats
            assert 'shots_on_target_avg' in stats
            assert 'corners_avg' in stats
            assert 'fouls_avg' in stats
            assert 'yellow_cards_avg' in stats
            assert 'red_cards_avg' in stats
            
            # Check that averages are calculated correctly
            assert stats['goals_scored_avg'] > 0
            assert stats['goals_conceded_avg'] > 0
            assert stats['shots_avg'] > 0
    
    def test_calculate_form_metrics(self, feature_engineer, sample_match_data):
        """Test form metrics calculation"""
        with patch.object(feature_engineer, '_get_team_matches') as mock_get_matches:
            mock_get_matches.return_value = sample_match_data
            
            form = feature_engineer._calculate_form_metrics(1, datetime(2023, 9, 1), 5)
            
            assert 'form_points' in form
            assert 'recent_results' in form
            assert 'win_rate' in form
            assert 'goal_difference_avg' in form
            
            # Check that form points are reasonable
            assert 0 <= form['form_points'] <= 3
            assert 0 <= form['win_rate'] <= 1
    
    def test_calculate_venue_specific_metrics(self, feature_engineer, sample_match_data):
        """Test venue-specific metrics calculation"""
        with patch.object(feature_engineer, '_get_team_matches') as mock_get_matches:
            mock_get_matches.return_value = sample_match_data
            
            # Test home metrics
            home_metrics = feature_engineer._calculate_venue_specific_metrics(
                1, datetime(2023, 9, 1), 10, venue='home'
            )
            
            assert 'goals_scored_avg' in home_metrics
            assert 'goals_conceded_avg' in home_metrics
            assert 'win_rate' in home_metrics
            
            # Test away metrics
            away_metrics = feature_engineer._calculate_venue_specific_metrics(
                1, datetime(2023, 9, 1), 10, venue='away'
            )
            
            assert 'goals_scored_avg' in away_metrics
            assert 'goals_conceded_avg' in away_metrics
            assert 'win_rate' in away_metrics
    
    def test_calculate_head_to_head_metrics(self, feature_engineer, sample_match_data):
        """Test head-to-head metrics calculation"""
        with patch.object(feature_engineer, '_get_head_to_head_matches') as mock_get_h2h:
            mock_get_h2h.return_value = sample_match_data
            
            h2h = feature_engineer._calculate_head_to_head_metrics(
                1, 2, datetime(2023, 9, 1), 10
            )
            
            assert 'total_matches' in h2h
            assert 'wins' in h2h
            assert 'draws' in h2h
            assert 'losses' in h2h
            assert 'goals_scored_avg' in h2h
            assert 'goals_conceded_avg' in h2h
            assert 'win_rate' in h2h
            
            # Check that totals add up
            total = h2h['wins'] + h2h['draws'] + h2h['losses']
            assert total == h2h['total_matches']
    
    def test_calculate_advanced_metrics(self, feature_engineer, sample_match_data):
        """Test advanced metrics calculation"""
        with patch.object(feature_engineer, '_get_team_matches') as mock_get_matches:
            mock_get_matches.return_value = sample_match_data
            
            advanced = feature_engineer._calculate_advanced_metrics(1, datetime(2023, 9, 1), 10)
            
            assert 'shot_accuracy' in advanced
            assert 'shots_per_goal' in advanced
            assert 'corners_per_goal' in advanced
            assert 'discipline_index' in advanced
            assert 'attacking_efficiency' in advanced
            assert 'defensive_efficiency' in advanced
            
            # Check that values are reasonable
            assert 0 <= advanced['shot_accuracy'] <= 1
            assert advanced['shots_per_goal'] >= 0
            assert advanced['discipline_index'] >= 0
    
    def test_apply_referee_adjustments(self, feature_engineer):
        """Test referee adjustments"""
        base_features = {
            'fouls_avg': 12.0,
            'yellow_cards_avg': 2.5,
            'red_cards_avg': 0.1
        }
        
        referee_stats = {
            'avg_fouls_per_match': 15.0,
            'avg_cards_per_match': 4.0,
            'card_tendency': 1.2
        }
        
        adjusted = feature_engineer._apply_referee_adjustments(base_features, referee_stats)
        
        assert 'fouls_avg_adjusted' in adjusted
        assert 'cards_avg_adjusted' in adjusted
        assert 'referee_factor' in adjusted
        
        # Check that adjustments are applied
        assert adjusted['fouls_avg_adjusted'] != base_features['fouls_avg']
        assert adjusted['cards_avg_adjusted'] != base_features['yellow_cards_avg']
    
    def test_engineer_team_features(self, feature_engineer):
        """Test complete team feature engineering"""
        with patch.object(feature_engineer, '_get_team_matches') as mock_get_matches, \
             patch.object(feature_engineer, '_get_head_to_head_matches') as mock_get_h2h, \
             patch.object(feature_engineer, '_get_referee_stats') as mock_get_referee:
            
            # Mock data
            mock_get_matches.return_value = pd.DataFrame([
                {
                    'date': datetime(2023, 8, 15),
                    'home_team_id': 1,
                    'away_team_id': 2,
                    'home_goals': 2,
                    'away_goals': 1,
                    'home_shots': 15,
                    'away_shots': 8,
                    'home_shots_on_target': 6,
                    'away_shots_on_target': 3
                }
            ])
            mock_get_h2h.return_value = pd.DataFrame()
            mock_get_referee.return_value = {'avg_fouls_per_match': 12.0, 'avg_cards_per_match': 3.0}
            
            features = feature_engineer.engineer_team_features(
                team_id=1,
                opponent_id=2,
                match_date=datetime(2023, 9, 1),
                venue='home',
                referee='Test Referee'
            )
            
            assert isinstance(features, TeamFeatures)
            assert features.goals_scored_avg >= 0
            assert features.goals_conceded_avg >= 0
            assert features.shots_avg >= 0
            assert 0 <= features.form_points <= 3
    
    def test_engineer_match_features(self, feature_engineer):
        """Test complete match feature engineering"""
        with patch.object(feature_engineer, 'engineer_team_features') as mock_engineer_team:
            # Mock team features
            home_features = TeamFeatures(
                goals_scored_avg=2.0,
                goals_conceded_avg=1.0,
                shots_avg=15.0,
                form_points=2.5,
                home_advantage=0.3
            )
            away_features = TeamFeatures(
                goals_scored_avg=1.5,
                goals_conceded_avg=1.2,
                shots_avg=12.0,
                form_points=2.0,
                home_advantage=0.0
            )
            
            mock_engineer_team.side_effect = [home_features, away_features]
            
            match_features = feature_engineer.engineer_match_features(
                home_team_id=1,
                away_team_id=2,
                match_date=datetime(2023, 9, 1),
                referee='Test Referee'
            )
            
            assert 'home_team' in match_features
            assert 'away_team' in match_features
            assert isinstance(match_features['home_team'], TeamFeatures)
            assert isinstance(match_features['away_team'], TeamFeatures)
    
    def test_create_feature_matrix(self, feature_engineer):
        """Test feature matrix creation"""
        matches = [
            {
                'home_team_id': 1,
                'away_team_id': 2,
                'match_date': datetime(2023, 9, 1),
                'referee': 'Test Referee 1'
            },
            {
                'home_team_id': 2,
                'away_team_id': 1,
                'match_date': datetime(2023, 9, 8),
                'referee': 'Test Referee 2'
            }
        ]
        
        with patch.object(feature_engineer, 'engineer_match_features') as mock_engineer_match:
            # Mock match features
            mock_engineer_match.return_value = {
                'home_team': TeamFeatures(goals_scored_avg=2.0, goals_conceded_avg=1.0),
                'away_team': TeamFeatures(goals_scored_avg=1.5, goals_conceded_avg=1.2)
            }
            
            feature_matrix = feature_engineer.create_feature_matrix(matches)
            
            assert isinstance(feature_matrix, pd.DataFrame)
            assert len(feature_matrix) == len(matches)
            assert 'home_goals_scored_avg' in feature_matrix.columns
            assert 'away_goals_scored_avg' in feature_matrix.columns

class TestValidation:
    """Test validation utilities"""
    
    def test_match_data_validator(self):
        """Test MatchDataValidator"""
        # Valid data
        valid_data = {
            'date': datetime(2023, 8, 15),
            'home_team': 'Real Madrid',
            'away_team': 'Barcelona',
            'home_goals': 2,
            'away_goals': 1,
            'home_shots': 15,
            'away_shots': 8
        }
        
        validator = MatchDataValidator(**valid_data)
        assert validator.home_team == 'Real Madrid'
        assert validator.away_team == 'Barcelona'
        assert validator.home_goals == 2
        
        # Invalid data - negative goals
        with pytest.raises(ValueError):
            MatchDataValidator(
                date=datetime(2023, 8, 15),
                home_team='Real Madrid',
                away_team='Barcelona',
                home_goals=-1,
                away_goals=1
            )
        
        # Invalid data - same teams
        with pytest.raises(ValueError):
            MatchDataValidator(
                date=datetime(2023, 8, 15),
                home_team='Real Madrid',
                away_team='Real Madrid',
                home_goals=2,
                away_goals=1
            )
    
    def test_team_validator(self):
        """Test TeamValidator"""
        # Valid data
        valid_data = {
            'name': 'Real Madrid',
            'short_name': 'RMA',
            'country': 'Spain'
        }
        
        validator = TeamValidator(**valid_data)
        assert validator.name == 'Real Madrid'
        assert validator.short_name == 'RMA'
        
        # Invalid data - empty name
        with pytest.raises(ValueError):
            TeamValidator(
                name='',
                short_name='RMA',
                country='Spain'
            )
    
    def test_validate_match_data(self):
        """Test match data validation function"""
        # Valid data
        valid_data = {
            'Date': '15/08/2023',
            'HomeTeam': 'Real Madrid',
            'AwayTeam': 'Barcelona',
            'FTHG': '2',
            'FTAG': '1',
            'FTR': 'H'
        }
        
        result = validate_match_data(valid_data)
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
        
        # Invalid data - missing required field
        invalid_data = {
            'Date': '15/08/2023',
            'HomeTeam': 'Real Madrid',
            # Missing AwayTeam
            'FTHG': '2',
            'FTAG': '1'
        }
        
        result = validate_match_data(invalid_data)
        assert result['is_valid'] is False
        assert len(result['errors']) > 0
    
    def test_validate_team_name(self):
        """Test team name validation"""
        # Valid team names
        assert validate_team_name('Real Madrid') is True
        assert validate_team_name('FC Barcelona') is True
        assert validate_team_name('Athletic Bilbao') is True
        
        # Invalid team names
        assert validate_team_name('') is False
        assert validate_team_name('   ') is False
        assert validate_team_name('A') is False  # Too short
        assert validate_team_name('A' * 101) is False  # Too long
    
    def test_validate_csv_data(self):
        """Test CSV data validation"""
        # Valid CSV data
        valid_csv = pd.DataFrame([
            {
                'Date': '15/08/2023',
                'HomeTeam': 'Real Madrid',
                'AwayTeam': 'Barcelona',
                'FTHG': 2,
                'FTAG': 1,
                'FTR': 'H'
            },
            {
                'Date': '22/08/2023',
                'HomeTeam': 'Barcelona',
                'AwayTeam': 'Real Madrid',
                'FTHG': 1,
                'FTAG': 3,
                'FTR': 'A'
            }
        ])
        
        result = validate_csv_data(valid_csv)
        assert result['is_valid'] is True
        assert result['total_rows'] == 2
        assert result['valid_rows'] == 2
        assert len(result['errors']) == 0
        
        # Invalid CSV data - missing columns
        invalid_csv = pd.DataFrame([
            {
                'Date': '15/08/2023',
                'HomeTeam': 'Real Madrid',
                # Missing required columns
            }
        ])
        
        result = validate_csv_data(invalid_csv)
        assert result['is_valid'] is False
        assert len(result['errors']) > 0
    
    def test_validate_prediction_request(self):
        """Test prediction request validation"""
        # Valid request
        valid_request = {
            'home_team': 'Real Madrid',
            'away_team': 'Barcelona',
            'match_date': '2023-09-15T20:00:00'
        }
        
        result = validate_prediction_request(valid_request)
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
        
        # Invalid request - same teams
        invalid_request = {
            'home_team': 'Real Madrid',
            'away_team': 'Real Madrid',
            'match_date': '2023-09-15T20:00:00'
        }
        
        result = validate_prediction_request(invalid_request)
        assert result['is_valid'] is False
        assert len(result['errors']) > 0
        
        # Invalid request - past date
        past_request = {
            'home_team': 'Real Madrid',
            'away_team': 'Barcelona',
            'match_date': '2020-01-01T20:00:00'
        }
        
        result = validate_prediction_request(past_request)
        assert result['is_valid'] is False
        assert any('past' in error.lower() for error in result['errors'])

class TestHelpers:
    """Test helper utilities"""
    
    def test_clean_team_name(self):
        """Test team name cleaning"""
        test_cases = [
            ('Real Madrid', 'Real Madrid'),
            ('real madrid', 'Real Madrid'),
            ('REAL MADRID', 'Real Madrid'),
            ('Real  Madrid', 'Real Madrid'),  # Extra spaces
            ('  Real Madrid  ', 'Real Madrid'),  # Leading/trailing spaces
            ('Real Madrid CF', 'Real Madrid'),  # Remove CF suffix
            ('FC Barcelona', 'Barcelona'),  # Remove FC prefix
            ('Club Atlético de Madrid', 'Atlético de Madrid'),  # Remove Club prefix
        ]
        
        for input_name, expected in test_cases:
            result = clean_team_name(input_name)
            assert result == expected
    
    def test_fuzzy_match_team(self):
        """Test fuzzy team matching"""
        team_names = ['Real Madrid', 'FC Barcelona', 'Atlético Madrid', 'Athletic Bilbao']
        
        # Exact match
        result = fuzzy_match_team('Real Madrid', team_names)
        assert result == 'Real Madrid'
        
        # Fuzzy match
        result = fuzzy_match_team('Real', team_names)
        assert result == 'Real Madrid'
        
        result = fuzzy_match_team('Barca', team_names)
        assert result == 'FC Barcelona'
        
        result = fuzzy_match_team('Atletico', team_names)  # Missing accent
        assert result == 'Atlético Madrid'
        
        # No match
        result = fuzzy_match_team('NonExistentTeam', team_names)
        assert result is None
        
        # Low threshold
        result = fuzzy_match_team('Real', team_names, threshold=0.9)
        assert result is None  # 'Real' vs 'Real Madrid' might not meet high threshold
    
    def test_calculate_form_points(self):
        """Test form points calculation"""
        # All wins
        results = ['W', 'W', 'W', 'W', 'W']
        form_points = calculate_form_points(results)
        assert form_points == 3.0
        
        # All draws
        results = ['D', 'D', 'D', 'D', 'D']
        form_points = calculate_form_points(results)
        assert form_points == 1.0
        
        # All losses
        results = ['L', 'L', 'L', 'L', 'L']
        form_points = calculate_form_points(results)
        assert form_points == 0.0
        
        # Mixed results
        results = ['W', 'D', 'L', 'W', 'D']
        form_points = calculate_form_points(results)
        assert 0 < form_points < 3
        
        # With weights (recent matches more important)
        results = ['L', 'L', 'L', 'W', 'W']  # Recent wins
        form_points_weighted = calculate_form_points(results, weights=[0.1, 0.1, 0.2, 0.3, 0.3])
        form_points_unweighted = calculate_form_points(results)
        assert form_points_weighted > form_points_unweighted
    
    def test_get_rolling_average(self):
        """Test rolling average calculation"""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # Simple rolling average
        avg = get_rolling_average(values, window=3)
        assert avg == 8.0  # Average of [8, 9, 10]
        
        # Rolling average with weights
        weights = [0.1, 0.3, 0.6]
        avg_weighted = get_rolling_average(values, window=3, weights=weights)
        expected = 0.1 * 8 + 0.3 * 9 + 0.6 * 10
        assert abs(avg_weighted - expected) < 0.001
        
        # Window larger than data
        avg = get_rolling_average([1, 2], window=5)
        assert avg == 1.5  # Average of all available data
        
        # Empty data
        avg = get_rolling_average([], window=3)
        assert avg == 0.0
    
    def test_calculate_elo_rating(self):
        """Test Elo rating calculation"""
        # Team wins (expected)
        new_rating = calculate_elo_rating(
            current_rating=1500,
            opponent_rating=1400,
            actual_score=1.0,  # Win
            k_factor=32
        )
        assert new_rating > 1500  # Rating should increase
        
        # Team loses (unexpected)
        new_rating = calculate_elo_rating(
            current_rating=1500,
            opponent_rating=1400,
            actual_score=0.0,  # Loss
            k_factor=32
        )
        assert new_rating < 1500  # Rating should decrease significantly
        
        # Draw
        new_rating = calculate_elo_rating(
            current_rating=1500,
            opponent_rating=1500,
            actual_score=0.5,  # Draw
            k_factor=32
        )
        assert new_rating == 1500  # Rating should stay the same
    
    def test_calculate_poisson_probability(self):
        """Test Poisson probability calculation"""
        # P(X = 2) with lambda = 1.5
        prob = calculate_poisson_probability(2, 1.5)
        expected = (1.5**2 * np.exp(-1.5)) / np.math.factorial(2)
        assert abs(prob - expected) < 0.001
        
        # P(X = 0) with lambda = 2.0
        prob = calculate_poisson_probability(0, 2.0)
        expected = np.exp(-2.0)
        assert abs(prob - expected) < 0.001
        
        # Edge case: lambda = 0
        prob = calculate_poisson_probability(0, 0.0)
        assert prob == 1.0
        
        prob = calculate_poisson_probability(1, 0.0)
        assert prob == 0.0
    
    def test_calculate_match_importance(self):
        """Test match importance calculation"""
        season_start = datetime(2023, 8, 1)
        season_end = datetime(2024, 5, 31)
        
        # Early season match
        early_match = datetime(2023, 8, 15)
        importance = calculate_match_importance(early_match, season_start, season_end)
        assert 0 <= importance <= 1
        
        # Mid-season match
        mid_match = datetime(2024, 1, 15)
        importance_mid = calculate_match_importance(mid_match, season_start, season_end)
        
        # Late season match
        late_match = datetime(2024, 5, 15)
        importance_late = calculate_match_importance(late_match, season_start, season_end)
        
        # Late season matches should be more important
        assert importance_late > importance_mid > importance
    
    def test_normalize_features(self):
        """Test feature normalization"""
        features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Z-score normalization
        normalized = normalize_features(features, method='zscore')
        assert normalized.shape == features.shape
        # Check that each column has mean ≈ 0 and std ≈ 1
        assert abs(np.mean(normalized[:, 0])) < 0.001
        assert abs(np.std(normalized[:, 0]) - 1.0) < 0.001
        
        # Min-max normalization
        normalized = normalize_features(features, method='minmax')
        assert normalized.shape == features.shape
        # Check that values are between 0 and 1
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)
        # Check that min is 0 and max is 1 for each column
        assert np.all(np.min(normalized, axis=0) == 0)
        assert np.all(np.max(normalized, axis=0) == 1)
    
    def test_calculate_team_strength(self):
        """Test team strength calculation"""
        matches = pd.DataFrame([
            {
                'date': datetime(2023, 8, 15),
                'goals_scored': 2,
                'goals_conceded': 1,
                'venue': 'home'
            },
            {
                'date': datetime(2023, 8, 22),
                'goals_scored': 1,
                'goals_conceded': 2,
                'venue': 'away'
            }
        ])
        
        strength = calculate_team_strength(
            matches,
            reference_date=datetime(2023, 9, 1),
            decay_factor=0.01
        )
        
        assert 'attack_strength' in strength
        assert 'defense_strength' in strength
        assert 'home_advantage' in strength
        
        assert strength['attack_strength'] > 0
        assert strength['defense_strength'] > 0
    
    def test_calculate_head_to_head_stats(self):
        """Test head-to-head statistics calculation"""
        matches = pd.DataFrame([
            {
                'date': datetime(2023, 8, 15),
                'team1_goals': 2,
                'team2_goals': 1,
                'result': 'team1_win'
            },
            {
                'date': datetime(2023, 8, 22),
                'team1_goals': 1,
                'team2_goals': 1,
                'result': 'draw'
            },
            {
                'date': datetime(2023, 8, 29),
                'team1_goals': 0,
                'team2_goals': 2,
                'result': 'team2_win'
            }
        ])
        
        stats = calculate_head_to_head_stats(matches)
        
        assert 'total_matches' in stats
        assert 'team1_wins' in stats
        assert 'team2_wins' in stats
        assert 'draws' in stats
        assert 'team1_goals_avg' in stats
        assert 'team2_goals_avg' in stats
        
        assert stats['total_matches'] == 3
        assert stats['team1_wins'] == 1
        assert stats['team2_wins'] == 1
        assert stats['draws'] == 1
        assert stats['team1_goals_avg'] == 1.0  # (2+1+0)/3
        assert stats['team2_goals_avg'] == 4.0/3  # (1+1+2)/3
    
    def test_safe_divide(self):
        """Test safe division"""
        # Normal division
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(7, 3) == 7/3
        
        # Division by zero
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=1.0) == 1.0
        
        # Zero divided by number
        assert safe_divide(0, 5) == 0.0
    
    def test_clamp(self):
        """Test value clamping"""
        # Value within range
        assert clamp(5, 0, 10) == 5
        assert clamp(0, 0, 10) == 0
        assert clamp(10, 0, 10) == 10
        
        # Value below range
        assert clamp(-5, 0, 10) == 0
        
        # Value above range
        assert clamp(15, 0, 10) == 10
        
        # Float values
        assert clamp(5.5, 0.0, 10.0) == 5.5
        assert clamp(-1.5, 0.0, 10.0) == 0.0
        assert clamp(12.5, 0.0, 10.0) == 10.0
    
    def test_exponential_decay_weights(self):
        """Test exponential decay weights"""
        weights = exponential_decay_weights(5, decay_factor=0.1)
        
        assert len(weights) == 5
        assert all(w > 0 for w in weights)
        assert weights[0] < weights[-1]  # Recent values have higher weights
        
        # Check that weights sum to 1
        assert abs(sum(weights) - 1.0) < 0.001
        
        # Check exponential decay property
        for i in range(len(weights) - 1):
            ratio = weights[i+1] / weights[i]
            assert abs(ratio - np.exp(0.1)) < 0.001
    
    def test_linear_decay_weights(self):
        """Test linear decay weights"""
        weights = linear_decay_weights(5)
        
        assert len(weights) == 5
        assert all(w > 0 for w in weights)
        assert weights[0] < weights[-1]  # Recent values have higher weights
        
        # Check that weights sum to 1
        assert abs(sum(weights) - 1.0) < 0.001
        
        # Check linear decay property
        differences = [weights[i+1] - weights[i] for i in range(len(weights) - 1)]
        # All differences should be approximately equal
        for i in range(len(differences) - 1):
            assert abs(differences[i] - differences[i+1]) < 0.001