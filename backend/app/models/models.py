from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any

from app.models.database import Base

class Team(Base):
    """Team model"""
    __tablename__ = "teams"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    normalized_name = Column(String(100), unique=True, nullable=False, index=True)
    founded_year = Column(Integer, nullable=True)
    stadium = Column(String(100), nullable=True)
    city = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")
    team_stats = relationship("TeamStats", back_populates="team")

class Match(Base):
    """Match model"""
    __tablename__ = "matches"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, nullable=False, index=True)
    season = Column(String(10), nullable=False, index=True)
    
    # Teams
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    
    # Match results
    home_goals = Column(Integer, nullable=True)
    away_goals = Column(Integer, nullable=True)
    result = Column(String(1), nullable=True)  # H, D, A
    
    # Half-time results
    ht_home_goals = Column(Integer, nullable=True)
    ht_away_goals = Column(Integer, nullable=True)
    ht_result = Column(String(1), nullable=True)
    
    # Match statistics
    home_shots = Column(Integer, nullable=True)
    away_shots = Column(Integer, nullable=True)
    home_shots_target = Column(Integer, nullable=True)
    away_shots_target = Column(Integer, nullable=True)
    home_fouls = Column(Integer, nullable=True)
    away_fouls = Column(Integer, nullable=True)
    home_corners = Column(Integer, nullable=True)
    away_corners = Column(Integer, nullable=True)
    home_yellow_cards = Column(Integer, nullable=True)
    away_yellow_cards = Column(Integer, nullable=True)
    home_red_cards = Column(Integer, nullable=True)
    away_red_cards = Column(Integer, nullable=True)
    
    # Officials
    referee = Column(String(100), nullable=True)
    
    # Metadata
    is_completed = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    predictions = relationship("Prediction", back_populates="match")

class Prediction(Base):
    """Prediction model"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    
    # Model information
    model_version = Column(String(20), nullable=False)
    model_type = Column(String(50), default="dixon_coles")
    
    # Predictions
    home_win_prob = Column(Float, nullable=False)
    draw_prob = Column(Float, nullable=False)
    away_win_prob = Column(Float, nullable=False)
    
    # Expected goals
    expected_home_goals = Column(Float, nullable=True)
    expected_away_goals = Column(Float, nullable=True)
    
    # Confidence metrics
    confidence_score = Column(Float, nullable=True)
    prediction_variance = Column(Float, nullable=True)
    
    # Feature importance (JSON)
    feature_importance = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    match = relationship("Match", back_populates="predictions")

class TeamStats(Base):
    """Team statistics model for rolling aggregates"""
    __tablename__ = "team_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    season = Column(String(10), nullable=False)
    
    # Time window
    window_size = Column(Integer, nullable=False)  # Number of matches
    as_of_date = Column(DateTime, nullable=False)
    
    # Venue (home/away/all)
    venue = Column(String(10), nullable=False, default="all")
    
    # Performance metrics
    matches_played = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    
    # Goals
    goals_for = Column(Float, default=0.0)
    goals_against = Column(Float, default=0.0)
    goals_for_per_90 = Column(Float, default=0.0)
    goals_against_per_90 = Column(Float, default=0.0)
    
    # Shots
    shots_per_90 = Column(Float, default=0.0)
    shots_target_per_90 = Column(Float, default=0.0)
    shots_allowed_per_90 = Column(Float, default=0.0)
    shots_target_allowed_per_90 = Column(Float, default=0.0)
    
    # Discipline
    fouls_per_90 = Column(Float, default=0.0)
    yellow_cards_per_90 = Column(Float, default=0.0)
    red_cards_per_90 = Column(Float, default=0.0)
    
    # Set pieces
    corners_per_90 = Column(Float, default=0.0)
    corners_allowed_per_90 = Column(Float, default=0.0)
    
    # Form indicators
    points_per_game = Column(Float, default=0.0)
    form_score = Column(Float, default=0.0)  # Weighted recent performance
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    team = relationship("Team", back_populates="team_stats")

class ModelMetadata(Base):
    """Model metadata and performance tracking"""
    __tablename__ = "model_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(20), unique=True, nullable=False)
    model_type = Column(String(50), nullable=False)
    
    # Training information
    training_start_date = Column(DateTime, nullable=False)
    training_end_date = Column(DateTime, nullable=False)
    training_matches = Column(Integer, nullable=False)
    
    # Model parameters (JSON)
    parameters = Column(JSON, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    
    # Performance metrics
    log_loss = Column(Float, nullable=True)
    brier_score = Column(Float, nullable=True)
    calibration_score = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    
    # Deployment status
    is_active = Column(Boolean, default=False)
    deployment_date = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class RefereeStats(Base):
    """Referee statistics for modeling discipline effects"""
    __tablename__ = "referee_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    referee_name = Column(String(100), nullable=False, index=True)
    season = Column(String(10), nullable=False)
    
    # Match statistics
    matches_officiated = Column(Integer, default=0)
    
    # Discipline averages
    avg_yellow_cards_per_match = Column(Float, default=0.0)
    avg_red_cards_per_match = Column(Float, default=0.0)
    avg_fouls_per_match = Column(Float, default=0.0)
    
    # Goal statistics
    avg_goals_per_match = Column(Float, default=0.0)
    avg_home_goals = Column(Float, default=0.0)
    avg_away_goals = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())