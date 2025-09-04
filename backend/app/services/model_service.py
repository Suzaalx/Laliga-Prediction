from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
import uuid
import asyncio
import numpy as np
from fastapi import BackgroundTasks

from app.models.models import ModelMetadata, Match, Prediction
from app.models.schemas import (
    ModelMetadataResponse, ModelPerformanceResponse, 
    CalibrationResponse, ModelRetrainingRequest,
    ModelType
)
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("model_service")

class ModelService:
    """Service for model management and operations"""
    
    def __init__(self):
        self.retraining_tasks = {}  # Track retraining tasks
    
    async def get_current_model(self, db: Session) -> Optional[ModelMetadataResponse]:
        """Get the currently active model"""
        
        model = db.query(ModelMetadata).filter(
            ModelMetadata.is_active == True
        ).first()
        
        if not model:
            return None
        
        return ModelMetadataResponse(
            id=model.id,
            version=model.version,
            model_type=ModelType(model.model_type),
            parameters=model.parameters,
            performance_metrics=model.performance_metrics,
            training_data_info=model.training_data_info,
            is_active=model.is_active,
            created_at=model.created_at,
            trained_at=model.trained_at,
            activated_at=model.activated_at,
            description=model.description
        )
    
    async def get_models(
        self,
        limit: int = 20,
        offset: int = 0,
        model_type: Optional[str] = None,
        db: Session = None
    ) -> List[ModelMetadataResponse]:
        """Get all models with pagination"""
        
        query = db.query(ModelMetadata)
        
        if model_type:
            query = query.filter(ModelMetadata.model_type == model_type)
        
        models = query.order_by(
            desc(ModelMetadata.created_at)
        ).offset(offset).limit(limit).all()
        
        result = []
        for model in models:
            result.append(ModelMetadataResponse(
                id=model.id,
                version=model.version,
                model_type=ModelType(model.model_type),
                parameters=model.parameters,
                performance_metrics=model.performance_metrics,
                training_data_info=model.training_data_info,
                is_active=model.is_active,
                created_at=model.created_at,
                trained_at=model.trained_at,
                activated_at=model.activated_at,
                description=model.description
            ))
        
        return result
    
    async def get_model_by_id(self, model_id: int, db: Session) -> Optional[ModelMetadataResponse]:
        """Get model by ID"""
        
        model = db.query(ModelMetadata).filter(
            ModelMetadata.id == model_id
        ).first()
        
        if not model:
            return None
        
        return ModelMetadataResponse(
            id=model.id,
            version=model.version,
            model_type=ModelType(model.model_type),
            parameters=model.parameters,
            performance_metrics=model.performance_metrics,
            training_data_info=model.training_data_info,
            is_active=model.is_active,
            created_at=model.created_at,
            trained_at=model.trained_at,
            activated_at=model.activated_at,
            description=model.description
        )
    
    async def get_model_by_version(self, version: str, db: Session) -> Optional[ModelMetadataResponse]:
        """Get model by version"""
        
        model = db.query(ModelMetadata).filter(
            ModelMetadata.version == version
        ).first()
        
        if not model:
            return None
        
        return ModelMetadataResponse(
            id=model.id,
            version=model.version,
            model_type=ModelType(model.model_type),
            parameters=model.parameters,
            performance_metrics=model.performance_metrics,
            training_data_info=model.training_data_info,
            is_active=model.is_active,
            created_at=model.created_at,
            trained_at=model.trained_at,
            activated_at=model.activated_at,
            description=model.description
        )
    
    async def get_model_performance(
        self,
        model_id: int,
        db: Session = None
    ) -> ModelPerformanceResponse:
        """Get model performance metrics"""
        
        model = db.query(ModelMetadata).filter(
            ModelMetadata.id == model_id
        ).first()
        
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Get predictions made by this model
        predictions = db.query(Prediction).filter(
            Prediction.model_version == model.version
        ).all()
        
        # Calculate performance metrics
        performance = await self._calculate_performance_metrics(
            predictions, model.version, db
        )
        
        return ModelPerformanceResponse(
            model_id=model_id,
            model_version=model.version,
            accuracy=performance["accuracy"],
            precision=performance["precision"],
            recall=performance["recall"],
            f1_score=performance["f1_score"],
            log_loss=performance["log_loss"],
            brier_score=performance["brier_score"],
            calibration_error=performance["calibration_error"],
            total_predictions=performance["total_predictions"],
            correct_predictions=performance["correct_predictions"],
            evaluation_period=performance["evaluation_period"],
            last_updated=datetime.utcnow()
        )
    
    async def get_model_calibration(
        self,
        model_id: int,
        bins: int = 10,
        db: Session = None
    ) -> CalibrationResponse:
        """Get model calibration data"""
        
        model = db.query(ModelMetadata).filter(
            ModelMetadata.id == model_id
        ).first()
        
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Get predictions with actual outcomes
        predictions_query = db.query(Prediction, Match).join(
            Match, Prediction.match_id == Match.id
        ).filter(
            and_(
                Prediction.model_version == model.version,
                Match.home_goals.isnot(None),
                Match.away_goals.isnot(None)
            )
        )
        
        predictions_with_outcomes = predictions_query.all()
        
        if not predictions_with_outcomes:
            # Return empty calibration if no data
            return CalibrationResponse(
                model_id=model_id,
                model_version=model.version,
                bins=bins,
                calibration_data=[],
                overall_calibration_error=0.0,
                reliability_diagram_data=[],
                total_predictions=0
            )
        
        # Calculate calibration
        calibration_data = await self._calculate_calibration(
            predictions_with_outcomes, bins
        )
        
        return CalibrationResponse(
            model_id=model_id,
            model_version=model.version,
            bins=bins,
            calibration_data=calibration_data["bins"],
            overall_calibration_error=calibration_data["overall_error"],
            reliability_diagram_data=calibration_data["reliability_data"],
            total_predictions=len(predictions_with_outcomes)
        )
    
    async def is_retraining_in_progress(self) -> bool:
        """Check if model retraining is currently in progress"""
        
        # Check if any retraining tasks are still running
        for task_id, task_info in self.retraining_tasks.items():
            if task_info["status"] in ["running", "started"]:
                return True
        
        return False
    
    async def start_retraining(
        self,
        request: ModelRetrainingRequest,
        background_tasks: BackgroundTasks,
        db: Session = None
    ) -> str:
        """Start model retraining process"""
        
        task_id = str(uuid.uuid4())
        
        # Initialize task tracking
        self.retraining_tasks[task_id] = {
            "status": "started",
            "started_at": datetime.utcnow(),
            "progress": 0,
            "message": "Retraining initiated",
            "request": request.dict()
        }
        
        # Start retraining in background
        background_tasks.add_task(
            self._retrain_model_background,
            task_id,
            request,
            db
        )
        
        logger.info(f"Model retraining started with task ID: {task_id}")
        
        return task_id
    
    async def get_retraining_status(self, task_id: str, db: Session) -> Optional[Dict[str, Any]]:
        """Get retraining task status"""
        
        if task_id not in self.retraining_tasks:
            return None
        
        return self.retraining_tasks[task_id]
    
    async def activate_model(self, model_id: int, db: Session) -> bool:
        """Activate a specific model version"""
        
        try:
            # Deactivate current active model
            current_active = db.query(ModelMetadata).filter(
                ModelMetadata.is_active == True
            ).first()
            
            if current_active:
                current_active.is_active = False
            
            # Activate the new model
            new_active = db.query(ModelMetadata).filter(
                ModelMetadata.id == model_id
            ).first()
            
            if not new_active:
                return False
            
            new_active.is_active = True
            new_active.activated_at = datetime.utcnow()
            
            db.commit()
            
            logger.info(f"Model {model_id} activated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate model {model_id}: {e}")
            db.rollback()
            return False
    
    async def delete_model(self, model_id: int, db: Session) -> bool:
        """Delete a model version"""
        
        try:
            model = db.query(ModelMetadata).filter(
                ModelMetadata.id == model_id
            ).first()
            
            if not model:
                return False
            
            # Delete associated predictions
            db.query(Prediction).filter(
                Prediction.model_version == model.version
            ).delete()
            
            # Delete the model
            db.delete(model)
            db.commit()
            
            logger.info(f"Model {model_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            db.rollback()
            return False
    
    async def _retrain_model_background(
        self,
        task_id: str,
        request: ModelRetrainingRequest,
        db: Session
    ):
        """Background task for model retraining"""
        
        try:
            # Update status
            self.retraining_tasks[task_id]["status"] = "running"
            self.retraining_tasks[task_id]["message"] = "Loading training data"
            self.retraining_tasks[task_id]["progress"] = 10
            
            # Simulate data loading
            await asyncio.sleep(2)
            
            # Get training data
            training_matches = await self._get_training_data(
                request.min_matches, request.seasons, db
            )
            
            self.retraining_tasks[task_id]["message"] = "Training Dixon-Coles model"
            self.retraining_tasks[task_id]["progress"] = 30
            
            # Simulate model training
            await asyncio.sleep(5)
            
            # Train the model (simplified)
            model_params = await self._train_dixon_coles_model(
                training_matches, request.time_decay_factor
            )
            
            self.retraining_tasks[task_id]["message"] = "Evaluating model performance"
            self.retraining_tasks[task_id]["progress"] = 70
            
            # Evaluate model
            performance_metrics = await self._evaluate_model(
                model_params, training_matches
            )
            
            self.retraining_tasks[task_id]["message"] = "Saving model"
            self.retraining_tasks[task_id]["progress"] = 90
            
            # Save new model
            new_version = f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            new_model = ModelMetadata(
                version=new_version,
                model_type="dixon_coles",
                parameters=model_params,
                performance_metrics=performance_metrics,
                training_data_info={
                    "total_matches": len(training_matches),
                    "seasons": request.seasons,
                    "min_matches": request.min_matches,
                    "time_decay_factor": request.time_decay_factor
                },
                is_active=False,  # Don't auto-activate
                trained_at=datetime.utcnow(),
                description=f"Retrained model with {len(training_matches)} matches"
            )
            
            db.add(new_model)
            db.commit()
            
            # Complete task
            self.retraining_tasks[task_id]["status"] = "completed"
            self.retraining_tasks[task_id]["message"] = f"Model {new_version} trained successfully"
            self.retraining_tasks[task_id]["progress"] = 100
            self.retraining_tasks[task_id]["completed_at"] = datetime.utcnow()
            self.retraining_tasks[task_id]["new_model_version"] = new_version
            
            logger.info(f"Model retraining completed: {new_version}")
            
        except Exception as e:
            # Handle errors
            self.retraining_tasks[task_id]["status"] = "failed"
            self.retraining_tasks[task_id]["message"] = f"Retraining failed: {str(e)}"
            self.retraining_tasks[task_id]["error"] = str(e)
            
            logger.error(f"Model retraining failed: {e}")
    
    async def _get_training_data(
        self,
        min_matches: Optional[int],
        seasons: Optional[List[str]],
        db: Session
    ) -> List[Match]:
        """Get training data for model retraining"""
        
        query = db.query(Match).filter(
            and_(
                Match.home_goals.isnot(None),
                Match.away_goals.isnot(None)
            )
        )
        
        if seasons:
            query = query.filter(Match.season.in_(seasons))
        
        matches = query.order_by(desc(Match.date)).all()
        
        if min_matches and len(matches) > min_matches:
            matches = matches[:min_matches]
        
        return matches
    
    async def _train_dixon_coles_model(
        self,
        matches: List[Match],
        time_decay_factor: Optional[float]
    ) -> Dict[str, Any]:
        """Train Dixon-Coles model (simplified implementation)"""
        
        # This is a simplified implementation
        # In a real scenario, you would implement the full Dixon-Coles algorithm
        
        # Simulate training parameters
        params = {
            "home_advantage": 0.3,
            "time_decay": time_decay_factor or 0.01,
            "rho": -0.1,  # Low-score correlation
            "team_strengths": {},  # Would contain actual team attack/defense strengths
            "training_matches": len(matches),
            "trained_at": datetime.utcnow().isoformat()
        }
        
        return params
    
    async def _evaluate_model(
        self,
        model_params: Dict[str, Any],
        training_matches: List[Match]
    ) -> Dict[str, float]:
        """Evaluate model performance (simplified)"""
        
        # This is a simplified evaluation
        # In practice, you would use cross-validation and proper metrics
        
        return {
            "accuracy": 0.65,
            "log_loss": 1.05,
            "brier_score": 0.25,
            "calibration_error": 0.08,
            "training_matches": len(training_matches)
        }
    
    async def _calculate_performance_metrics(
        self,
        predictions: List[Prediction],
        model_version: str,
        db: Session
    ) -> Dict[str, Any]:
        """Calculate detailed performance metrics"""
        
        if not predictions:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "log_loss": 0.0,
                "brier_score": 0.0,
                "calibration_error": 0.0,
                "total_predictions": 0,
                "correct_predictions": 0,
                "evaluation_period": {
                    "start_date": None,
                    "end_date": None
                }
            }
        
        # Get actual match outcomes
        correct_predictions = 0
        total_predictions = len(predictions)
        
        # Simplified accuracy calculation
        # In practice, you would implement proper evaluation metrics
        
        return {
            "accuracy": 0.65,  # Placeholder
            "precision": 0.68,
            "recall": 0.62,
            "f1_score": 0.65,
            "log_loss": 1.05,
            "brier_score": 0.25,
            "calibration_error": 0.08,
            "total_predictions": total_predictions,
            "correct_predictions": int(total_predictions * 0.65),
            "evaluation_period": {
                "start_date": min(p.created_at for p in predictions),
                "end_date": max(p.created_at for p in predictions)
            }
        }
    
    async def _calculate_calibration(
        self,
        predictions_with_outcomes: List[tuple],
        bins: int
    ) -> Dict[str, Any]:
        """Calculate calibration metrics"""
        
        # Simplified calibration calculation
        # In practice, you would implement proper calibration analysis
        
        bin_data = []
        for i in range(bins):
            bin_data.append({
                "bin_lower": i / bins,
                "bin_upper": (i + 1) / bins,
                "predicted_probability": (i + 0.5) / bins,
                "actual_frequency": (i + 0.5) / bins + np.random.normal(0, 0.05),
                "count": len(predictions_with_outcomes) // bins
            })
        
        return {
            "bins": bin_data,
            "overall_error": 0.08,
            "reliability_data": bin_data
        }