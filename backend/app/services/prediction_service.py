from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
import numpy as np
import uuid

from app.models.models import Team, Match, Prediction, TeamStats
from app.models.schemas import (
    PredictionResponse, TeamResponse, MatchResult, VenueType
)
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("prediction_service")

class PredictionService:
    """Service for generating match predictions using Dixon-Coles model"""
    
    def __init__(self):
        self.model_version = settings.MODEL_VERSION
        self.home_advantage = 0.3  # Default home advantage
        self.time_decay = 0.01  # Time decay parameter
        
    async def get_team_by_name(self, team_name: str, db: Session) -> Optional[Team]:
        """Get team by name with fuzzy matching"""
        
        # Exact match first
        team = db.query(Team).filter(
            Team.name.ilike(f"%{team_name}%")
        ).first()
        
        if not team:
            # Try alternative names or common variations
            variations = [
                team_name.replace(" ", ""),
                team_name.replace("-", " "),
                team_name.replace("_", " ")
            ]
            
            for variation in variations:
                team = db.query(Team).filter(
                    or_(
                        Team.name.ilike(f"%{variation}%"),
                        Team.alternative_names.ilike(f"%{variation}%")
                    )
                ).first()
                if team:
                    break
        
        return team
    
    async def get_team_stats(
        self, 
        team: Team, 
        venue: Optional[str] = None,
        last_n_matches: int = 10,
        db: Session = None
    ) -> Dict[str, float]:
        """Get team statistics for prediction"""
        
        # Get recent matches
        query = db.query(Match).filter(
            or_(
                Match.home_team_id == team.id,
                Match.away_team_id == team.id
            )
        )
        
        if venue == "home":
            query = query.filter(Match.home_team_id == team.id)
        elif venue == "away":
            query = query.filter(Match.away_team_id == team.id)
        
        recent_matches = query.order_by(
            Match.date.desc()
        ).limit(last_n_matches).all()
        
        if not recent_matches:
            # Return default stats if no matches found
            return {
                "attack_strength": 1.0,
                "defense_strength": 1.0,
                "goals_scored_avg": 1.5,
                "goals_conceded_avg": 1.5,
                "shots_per_game": 12.0,
                "shots_on_target_per_game": 4.0,
                "form_points": 1.0
            }
        
        # Calculate statistics
        total_goals_scored = 0
        total_goals_conceded = 0
        total_shots = 0
        total_shots_on_target = 0
        total_points = 0
        
        for match in recent_matches:
            if match.home_team_id == team.id:
                # Team playing at home
                total_goals_scored += match.home_goals or 0
                total_goals_conceded += match.away_goals or 0
                total_shots += match.home_shots or 0
                total_shots_on_target += match.home_shots_on_target or 0
                
                # Points calculation
                if match.home_goals > match.away_goals:
                    total_points += 3
                elif match.home_goals == match.away_goals:
                    total_points += 1
            else:
                # Team playing away
                total_goals_scored += match.away_goals or 0
                total_goals_conceded += match.home_goals or 0
                total_shots += match.away_shots or 0
                total_shots_on_target += match.away_shots_on_target or 0
                
                # Points calculation
                if match.away_goals > match.home_goals:
                    total_points += 3
                elif match.away_goals == match.home_goals:
                    total_points += 1
        
        num_matches = len(recent_matches)
        league_avg_goals = 2.5  # La Liga average
        
        return {
            "attack_strength": (total_goals_scored / num_matches) / league_avg_goals,
            "defense_strength": league_avg_goals / (total_goals_conceded / num_matches) if total_goals_conceded > 0 else 2.0,
            "goals_scored_avg": total_goals_scored / num_matches,
            "goals_conceded_avg": total_goals_conceded / num_matches,
            "shots_per_game": total_shots / num_matches,
            "shots_on_target_per_game": total_shots_on_target / num_matches,
            "form_points": total_points / num_matches
        }
    
    def calculate_dixon_coles_probabilities(
        self,
        home_stats: Dict[str, float],
        away_stats: Dict[str, float],
        home_advantage: float = None
    ) -> Dict[str, float]:
        """Calculate match probabilities using Dixon-Coles model"""
        
        if home_advantage is None:
            home_advantage = self.home_advantage
        
        # Calculate expected goals using Dixon-Coles intensities
        # λ_home = exp(μ + α_home - δ_away + γ)
        # λ_away = exp(μ + α_away - δ_home)
        
        mu = 0.3  # Base scoring rate
        
        lambda_home = np.exp(
            mu + 
            np.log(home_stats["attack_strength"]) - 
            np.log(away_stats["defense_strength"]) + 
            home_advantage
        )
        
        lambda_away = np.exp(
            mu + 
            np.log(away_stats["attack_strength"]) - 
            np.log(home_stats["defense_strength"])
        )
        
        # Calculate probabilities for different scorelines
        max_goals = 6
        probabilities = np.zeros((max_goals + 1, max_goals + 1))
        
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                # Basic Poisson probability
                prob = (
                    (lambda_home ** home_goals * np.exp(-lambda_home)) / np.math.factorial(home_goals) *
                    (lambda_away ** away_goals * np.exp(-lambda_away)) / np.math.factorial(away_goals)
                )
                
                # Dixon-Coles low-score correction
                if home_goals <= 1 and away_goals <= 1:
                    rho = -0.1  # Correlation parameter
                    if home_goals == 0 and away_goals == 0:
                        prob *= (1 - lambda_home * lambda_away * rho)
                    elif home_goals == 0 and away_goals == 1:
                        prob *= (1 + lambda_home * rho)
                    elif home_goals == 1 and away_goals == 0:
                        prob *= (1 + lambda_away * rho)
                    elif home_goals == 1 and away_goals == 1:
                        prob *= (1 - rho)
                
                probabilities[home_goals, away_goals] = prob
        
        # Normalize probabilities
        total_prob = np.sum(probabilities)
        probabilities /= total_prob
        
        # Calculate match outcome probabilities
        home_win_prob = np.sum(probabilities[np.triu_indices_from(probabilities, k=1)])
        away_win_prob = np.sum(probabilities[np.tril_indices_from(probabilities, k=-1)])
        draw_prob = np.sum(np.diag(probabilities))
        
        # Calculate expected goals
        expected_home_goals = lambda_home
        expected_away_goals = lambda_away
        
        # Most likely scoreline
        most_likely_idx = np.unravel_index(np.argmax(probabilities), probabilities.shape)
        most_likely_score = f"{most_likely_idx[0]}-{most_likely_idx[1]}"
        
        return {
            "home_win_prob": float(home_win_prob),
            "draw_prob": float(draw_prob),
            "away_win_prob": float(away_win_prob),
            "expected_home_goals": float(expected_home_goals),
            "expected_away_goals": float(expected_away_goals),
            "most_likely_score": most_likely_score,
            "scoreline_probabilities": probabilities.tolist()
        }
    
    async def predict_match(
        self,
        home_team: Team,
        away_team: Team,
        match_date: datetime,
        venue: Optional[str] = None,
        referee: Optional[str] = None,
        db: Session = None
    ) -> PredictionResponse:
        """Generate prediction for a single match"""
        
        try:
            # Get team statistics
            home_stats = await self.get_team_stats(
                team=home_team,
                venue="home",
                db=db
            )
            
            away_stats = await self.get_team_stats(
                team=away_team,
                venue="away",
                db=db
            )
            
            # Adjust home advantage based on venue
            home_advantage = self.home_advantage
            if venue and venue.lower() == "neutral":
                home_advantage = 0.0
            
            # Calculate probabilities
            probabilities = self.calculate_dixon_coles_probabilities(
                home_stats=home_stats,
                away_stats=away_stats,
                home_advantage=home_advantage
            )
            
            # Calculate confidence based on probability spread
            prob_values = [
                probabilities["home_win_prob"],
                probabilities["draw_prob"],
                probabilities["away_win_prob"]
            ]
            confidence = 1.0 - (np.std(prob_values) / np.mean(prob_values))
            
            # Create prediction response
            prediction = PredictionResponse(
                match_id=None,  # Will be set if match exists in DB
                home_team=home_team.name,
                away_team=away_team.name,
                match_date=match_date,
                home_win_prob=probabilities["home_win_prob"],
                draw_prob=probabilities["draw_prob"],
                away_win_prob=probabilities["away_win_prob"],
                expected_home_goals=probabilities["expected_home_goals"],
                expected_away_goals=probabilities["expected_away_goals"],
                most_likely_score=probabilities["most_likely_score"],
                confidence=confidence,
                model_version=self.model_version,
                created_at=datetime.utcnow(),
                venue=venue or "home",
                referee=referee,
                features={
                    "home_attack_strength": home_stats["attack_strength"],
                    "home_defense_strength": home_stats["defense_strength"],
                    "away_attack_strength": away_stats["attack_strength"],
                    "away_defense_strength": away_stats["defense_strength"],
                    "home_form": home_stats["form_points"],
                    "away_form": away_stats["form_points"],
                    "home_advantage": home_advantage
                }
            )
            
            # Save prediction to database if match exists
            existing_match = db.query(Match).filter(
                and_(
                    Match.home_team_id == home_team.id,
                    Match.away_team_id == away_team.id,
                    Match.date == match_date.date()
                )
            ).first()
            
            if existing_match:
                prediction.match_id = existing_match.id
                
                # Save prediction to database
                db_prediction = Prediction(
                    id=str(uuid.uuid4()),
                    match_id=existing_match.id,
                    home_win_prob=prediction.home_win_prob,
                    draw_prob=prediction.draw_prob,
                    away_win_prob=prediction.away_win_prob,
                    expected_home_goals=prediction.expected_home_goals,
                    expected_away_goals=prediction.expected_away_goals,
                    confidence=prediction.confidence,
                    model_version=prediction.model_version,
                    features=prediction.features,
                    created_at=prediction.created_at
                )
                
                db.add(db_prediction)
                db.commit()
            
            logger.info(
                "Prediction generated",
                home_team=home_team.name,
                away_team=away_team.name,
                home_win_prob=prediction.home_win_prob,
                confidence=prediction.confidence
            )
            
            return prediction
            
        except Exception as e:
            logger.error(
                "Prediction generation failed",
                home_team=home_team.name,
                away_team=away_team.name,
                error=str(e)
            )
            raise
    
    async def get_upcoming_matches(
        self,
        days: int = 7,
        limit: int = 50,
        db: Session = None
    ) -> List[Match]:
        """Get upcoming matches"""
        
        start_date = datetime.utcnow().date()
        end_date = start_date + timedelta(days=days)
        
        matches = db.query(Match).filter(
            and_(
                Match.date >= start_date,
                Match.date <= end_date,
                Match.home_goals.is_(None),  # Match not played yet
                Match.away_goals.is_(None)
            )
        ).order_by(Match.date.asc()).limit(limit).all()
        
        return matches
    
    async def get_prediction_history(
        self,
        match_id: int,
        db: Session = None
    ) -> List[PredictionResponse]:
        """Get prediction history for a match"""
        
        predictions = db.query(Prediction).filter(
            Prediction.match_id == match_id
        ).order_by(Prediction.created_at.desc()).all()
        
        result = []
        for pred in predictions:
            match = db.query(Match).filter(Match.id == pred.match_id).first()
            if match:
                result.append(PredictionResponse(
                    match_id=pred.match_id,
                    home_team=match.home_team.name,
                    away_team=match.away_team.name,
                    match_date=match.date,
                    home_win_prob=pred.home_win_prob,
                    draw_prob=pred.draw_prob,
                    away_win_prob=pred.away_win_prob,
                    expected_home_goals=pred.expected_home_goals,
                    expected_away_goals=pred.expected_away_goals,
                    confidence=pred.confidence,
                    model_version=pred.model_version,
                    created_at=pred.created_at,
                    features=pred.features
                ))
        
        return result
    
    async def get_prediction_stats(self, db: Session) -> Dict[str, Any]:
        """Get prediction statistics"""
        
        today = datetime.utcnow().date()
        
        # Total predictions
        total_predictions = db.query(Prediction).count()
        
        # Predictions today
        predictions_today = db.query(Prediction).filter(
            Prediction.created_at >= today
        ).count()
        
        # Average confidence
        avg_confidence_result = db.query(
            func.avg(Prediction.confidence)
        ).scalar()
        avg_confidence = float(avg_confidence_result) if avg_confidence_result else 0.0
        
        # Model accuracy (simplified - would need actual match results)
        # This is a placeholder - real implementation would compare predictions to outcomes
        model_accuracy = 0.65  # Placeholder
        
        return {
            "total_predictions": total_predictions,
            "predictions_today": predictions_today,
            "average_confidence": avg_confidence,
            "model_accuracy": model_accuracy
        }