from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum

class MatchResult(str, Enum):
    """Match result enumeration"""
    HOME_WIN = "H"
    DRAW = "D"
    AWAY_WIN = "A"

class VenueType(str, Enum):
    """Venue type enumeration"""
    HOME = "home"
    AWAY = "away"
    ALL = "all"

class ModelType(str, Enum):
    """Model type enumeration"""
    DIXON_COLES = "dixon_coles"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"

# Base schemas
class TeamBase(BaseModel):
    """Base team schema"""
    name: str = Field(..., min_length=1, max_length=100)
    normalized_name: str = Field(..., min_length=1, max_length=100)
    founded_year: Optional[int] = Field(None, ge=1800, le=2030)
    stadium: Optional[str] = Field(None, max_length=100)
    city: Optional[str] = Field(None, max_length=100)
    is_active: bool = Field(True, description="Whether the team is currently active")

class TeamCreate(TeamBase):
    """Team creation schema"""
    pass

class TeamUpdate(BaseModel):
    """Team update schema"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    stadium: Optional[str] = Field(None, max_length=100)
    city: Optional[str] = Field(None, max_length=100)

class Team(TeamBase):
    """Team response schema"""
    id: int
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True

# Team response schemas
class TeamResponse(Team):
    """Team response schema for API endpoints"""
    pass

class TeamFormResponse(BaseModel):
    """Team form response schema"""
    team_id: int
    team_name: str
    season: str
    window_size: int
    venue: VenueType
    as_of_date: datetime
    
    # Recent form metrics
    recent_matches: int
    recent_wins: int
    recent_draws: int
    recent_losses: int
    recent_points: int
    recent_goals_for: int
    recent_goals_against: int
    
    # Form scores
    form_score: float
    momentum_score: float
    
    class Config:
        from_attributes = True

# Match schemas
class MatchBase(BaseModel):
    """Base match schema"""
    date: datetime
    season: str = Field(..., pattern=r"^\d{4}$")
    home_team_id: int
    away_team_id: int
    referee: Optional[str] = Field(None, max_length=100)
    
    @validator('home_team_id', 'away_team_id')
    def teams_must_be_different(cls, v, values):
        if 'home_team_id' in values and v == values['home_team_id']:
            raise ValueError('Home and away teams must be different')
        return v

class MatchCreate(MatchBase):
    """Match creation schema"""
    pass

class MatchUpdate(BaseModel):
    """Match update schema"""
    home_goals: Optional[int] = Field(None, ge=0)
    away_goals: Optional[int] = Field(None, ge=0)
    result: Optional[MatchResult]
    ht_home_goals: Optional[int] = Field(None, ge=0)
    ht_away_goals: Optional[int] = Field(None, ge=0)
    ht_result: Optional[MatchResult]
    home_shots: Optional[int] = Field(None, ge=0)
    away_shots: Optional[int] = Field(None, ge=0)
    home_shots_target: Optional[int] = Field(None, ge=0)
    away_shots_target: Optional[int] = Field(None, ge=0)
    home_fouls: Optional[int] = Field(None, ge=0)
    away_fouls: Optional[int] = Field(None, ge=0)
    home_corners: Optional[int] = Field(None, ge=0)
    away_corners: Optional[int] = Field(None, ge=0)
    home_yellow_cards: Optional[int] = Field(None, ge=0)
    away_yellow_cards: Optional[int] = Field(None, ge=0)
    home_red_cards: Optional[int] = Field(None, ge=0)
    away_red_cards: Optional[int] = Field(None, ge=0)
    is_completed: Optional[bool] = False

class Match(MatchBase):
    """Match response schema"""
    id: int
    home_goals: Optional[int]
    away_goals: Optional[int]
    result: Optional[MatchResult]
    ht_home_goals: Optional[int]
    ht_away_goals: Optional[int]
    ht_result: Optional[MatchResult]
    home_shots: Optional[int]
    away_shots: Optional[int]
    home_shots_target: Optional[int]
    away_shots_target: Optional[int]
    home_fouls: Optional[int]
    away_fouls: Optional[int]
    home_corners: Optional[int]
    away_corners: Optional[int]
    home_yellow_cards: Optional[int]
    away_yellow_cards: Optional[int]
    home_red_cards: Optional[int]
    away_red_cards: Optional[int]
    is_completed: bool
    created_at: datetime
    updated_at: Optional[datetime]
    
    # Nested relationships
    home_team: Optional[Team]
    away_team: Optional[Team]
    
    class Config:
        from_attributes = True

# Match response schemas
class MatchResponse(Match):
    """Match response schema for API endpoints"""
    pass

# Prediction schemas
class PredictionRequest(BaseModel):
    """Prediction request schema"""
    home_team: str = Field(..., min_length=1)
    away_team: str = Field(..., min_length=1)
    match_date: Optional[datetime] = None
    venue: Optional[str] = None
    referee: Optional[str] = None
    
    @validator('home_team', 'away_team')
    def teams_must_be_different(cls, v, values):
        if 'home_team' in values and v == values['home_team']:
            raise ValueError('Home and away teams must be different')
        return v

class PredictionResponse(BaseModel):
    """Prediction response schema"""
    match_id: Optional[int]
    home_team: str
    away_team: str
    match_date: Optional[datetime]
    
    # Probabilities
    home_win_prob: float = Field(..., ge=0.0, le=1.0)
    draw_prob: float = Field(..., ge=0.0, le=1.0)
    away_win_prob: float = Field(..., ge=0.0, le=1.0)
    
    # Expected goals
    expected_home_goals: Optional[float] = Field(None, ge=0.0)
    expected_away_goals: Optional[float] = Field(None, ge=0.0)
    
    # Confidence metrics
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    prediction_variance: Optional[float] = Field(None, ge=0.0)
    
    # Model metadata
    model_version: str
    model_type: ModelType
    
    # Feature importance
    feature_importance: Optional[Dict[str, float]]
    
    @validator('home_win_prob', 'draw_prob', 'away_win_prob')
    def probabilities_sum_to_one(cls, v, values):
        probs = [v]
        for field in ['home_win_prob', 'draw_prob']:
            if field in values:
                probs.append(values[field])
        
        if len(probs) == 3 and abs(sum(probs) - 1.0) > 0.01:
            raise ValueError('Probabilities must sum to 1.0')
        return v

class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema"""
    matches: List[PredictionRequest] = Field(..., min_items=1, max_items=100)

class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema"""
    predictions: List[PredictionResponse]
    total_matches: int
    processing_time_ms: float

# Team statistics schemas
class TeamStatsResponse(BaseModel):
    """Team statistics response schema"""
    team_id: int
    team_name: str
    season: str
    window_size: int
    venue: VenueType
    as_of_date: datetime
    
    # Performance metrics
    matches_played: int
    wins: int
    draws: int
    losses: int
    
    # Goals
    goals_for_per_90: float
    goals_against_per_90: float
    
    # Shots
    shots_per_90: float
    shots_target_per_90: float
    shots_allowed_per_90: float
    shots_target_allowed_per_90: float
    
    # Discipline
    fouls_per_90: float
    yellow_cards_per_90: float
    red_cards_per_90: float
    
    # Set pieces
    corners_per_90: float
    corners_allowed_per_90: float
    
    # Form
    points_per_game: float
    form_score: float
    
    class Config:
        from_attributes = True

# Model metadata schemas
class ModelMetadataResponse(BaseModel):
    """Model metadata response schema"""
    version: str
    model_type: ModelType
    training_start_date: datetime
    training_end_date: datetime
    training_matches: int
    
    # Performance metrics
    log_loss: Optional[float]
    brier_score: Optional[float]
    calibration_score: Optional[float]
    accuracy: Optional[float]
    
    # Status
    is_active: bool
    deployment_date: Optional[datetime]
    created_at: datetime
    
    # Parameters (simplified for API)
    parameter_count: Optional[int]
    hyperparameters: Optional[Dict[str, Any]]
    
    class Config:
        from_attributes = True

class ModelPerformanceResponse(BaseModel):
    """Model performance metrics response schema"""
    model_id: int
    model_version: str
    model_type: ModelType
    
    # Performance metrics
    accuracy: float = Field(..., ge=0.0, le=1.0)
    log_loss: float = Field(..., ge=0.0)
    brier_score: float = Field(..., ge=0.0, le=1.0)
    calibration_score: float = Field(..., ge=0.0, le=1.0)
    
    # Detailed metrics by outcome
    home_win_accuracy: float = Field(..., ge=0.0, le=1.0)
    draw_accuracy: float = Field(..., ge=0.0, le=1.0)
    away_win_accuracy: float = Field(..., ge=0.0, le=1.0)
    
    # Calibration metrics
    home_win_calibration: float = Field(..., ge=0.0, le=1.0)
    draw_calibration: float = Field(..., ge=0.0, le=1.0)
    away_win_calibration: float = Field(..., ge=0.0, le=1.0)
    
    # Test period
    test_start_date: datetime
    test_end_date: datetime
    test_matches: int
    
    # Additional metrics
    roc_auc_home: Optional[float] = Field(None, ge=0.0, le=1.0)
    roc_auc_draw: Optional[float] = Field(None, ge=0.0, le=1.0)
    roc_auc_away: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    class Config:
        from_attributes = True

class CalibrationResponse(BaseModel):
    """Model calibration response schema"""
    model_id: int
    model_version: str
    model_type: ModelType
    
    # Calibration bins
    bin_boundaries: List[float]
    bin_counts: List[int]
    bin_accuracies: List[float]
    bin_confidences: List[float]
    
    # Overall calibration metrics
    expected_calibration_error: float = Field(..., ge=0.0, le=1.0)
    maximum_calibration_error: float = Field(..., ge=0.0, le=1.0)
    reliability_score: float = Field(..., ge=0.0, le=1.0)
    
    # Outcome-specific calibration
    home_win_calibration: Dict[str, Any]
    draw_calibration: Dict[str, Any]
    away_win_calibration: Dict[str, Any]
    
    # Test period
    test_start_date: datetime
    test_end_date: datetime
    test_matches: int
    
    class Config:
        from_attributes = True

class ModelRetrainingRequest(BaseModel):
    """Model retraining request schema"""
    model_type: ModelType = ModelType.DIXON_COLES
    training_start_date: Optional[datetime] = None
    training_end_date: Optional[datetime] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    force_retrain: bool = False
    
    class Config:
        from_attributes = True

class ModelRetrainingResponse(BaseModel):
    """Model retraining response schema"""
    task_id: str
    model_type: ModelType
    status: str = Field(..., pattern=r"^(started|running|completed|failed)$")
    
    # Training configuration
    training_start_date: datetime
    training_end_date: datetime
    hyperparameters: Dict[str, Any]
    
    # Progress information
    progress_percent: Optional[float] = Field(None, ge=0.0, le=100.0)
    current_step: Optional[str] = None
    estimated_completion: Optional[datetime] = None
    
    # Results (when completed)
    new_model_version: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    
    # Timestamps
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Health and error schemas
class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., pattern=r"^(healthy|unhealthy|degraded)$")
    timestamp: datetime
    version: str
    
    # Service checks
    database: bool
    cache: bool
    model: bool
    
    # Performance metrics
    uptime_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    
    # Additional info
    active_model_version: Optional[str]
    last_prediction_time: Optional[datetime]

# Error schemas
class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    message: str
    timestamp: datetime
    request_id: Optional[str]
    details: Optional[Dict[str, Any]]