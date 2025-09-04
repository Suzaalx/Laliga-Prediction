from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import time

from app.models.database import get_db
from app.models.schemas import (
    ModelMetadataResponse, ModelPerformanceResponse, 
    ModelRetrainingRequest, ModelRetrainingResponse,
    CalibrationResponse, ErrorResponse
)
from app.services.model_service import ModelService
from app.services.cache_service import CacheService
from app.core.logging import get_logger, ModelLoggingMixin
from app.core.config import settings

router = APIRouter()
logger = get_logger("models")

# Initialize services
model_service = ModelService()
cache_service = CacheService()

class ModelLogger(ModelLoggingMixin):
    def __init__(self):
        super().__init__()

model_logger = ModelLogger()

@router.get("/current", response_model=ModelMetadataResponse)
async def get_current_model(
    db: Session = Depends(get_db)
):
    """Get current active model metadata"""
    
    try:
        # Check cache first
        cache_key = "current_model"
        cached_model = await cache_service.get(cache_key)
        
        if cached_model:
            logger.info("Returning cached current model")
            return cached_model
        
        model = await model_service.get_current_model(db)
        
        if not model:
            raise HTTPException(
                status_code=404,
                detail="No active model found"
            )
        
        # Cache the result
        await cache_service.set(cache_key, model, ttl=settings.CACHE_TTL)
        
        logger.info(
            "Current model retrieved",
            model_version=model.version,
            model_type=model.model_type
        )
        
        return model
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get current model", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get current model: {str(e)}"
        )

@router.get("/", response_model=List[ModelMetadataResponse])
async def get_models(
    limit: int = 20,
    offset: int = 0,
    model_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all model versions with pagination"""
    
    try:
        models = await model_service.get_models(
            limit=limit,
            offset=offset,
            model_type=model_type,
            db=db
        )
        
        logger.info(
            "Models retrieved",
            count=len(models),
            model_type=model_type
        )
        
        return models
        
    except Exception as e:
        logger.error("Failed to get models", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get models: {str(e)}"
        )

@router.get("/{model_id}", response_model=ModelMetadataResponse)
async def get_model(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Get specific model by ID"""
    
    try:
        # Check cache first
        cache_key = f"model:{model_id}"
        cached_model = await cache_service.get(cache_key)
        
        if cached_model:
            logger.info("Returning cached model", model_id=model_id)
            return cached_model
        
        model = await model_service.get_model_by_id(model_id, db)
        
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model with ID {model_id} not found"
            )
        
        # Cache the result
        await cache_service.set(cache_key, model, ttl=settings.CACHE_TTL)
        
        logger.info(
            "Model retrieved",
            model_id=model_id,
            model_version=model.version
        )
        
        return model
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model", model_id=model_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model: {str(e)}"
        )

@router.get("/version/{version}", response_model=ModelMetadataResponse)
async def get_model_by_version(
    version: str,
    db: Session = Depends(get_db)
):
    """Get specific model by version"""
    
    try:
        # Check cache first
        cache_key = f"model:version:{version}"
        cached_model = await cache_service.get(cache_key)
        
        if cached_model:
            logger.info("Returning cached model", version=version)
            return cached_model
        
        model = await model_service.get_model_by_version(version, db)
        
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model version '{version}' not found"
            )
        
        # Cache the result
        await cache_service.set(cache_key, model, ttl=settings.CACHE_TTL)
        
        logger.info(
            "Model retrieved by version",
            version=version,
            model_id=model.id
        )
        
        return model
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model by version", version=version, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model: {str(e)}"
        )

@router.get("/performance/{model_id}", response_model=ModelPerformanceResponse)
async def get_model_performance(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Get model performance metrics"""
    
    try:
        # Check cache first
        cache_key = f"model_performance:{model_id}"
        cached_performance = await cache_service.get(cache_key)
        
        if cached_performance:
            logger.info("Returning cached model performance", model_id=model_id)
            return cached_performance
        
        # Validate model exists
        model = await model_service.get_model_by_id(model_id, db)
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model with ID {model_id} not found"
            )
        
        performance = await model_service.get_model_performance(
            model_id=model_id,
            db=db
        )
        
        # Cache the result
        await cache_service.set(cache_key, performance, ttl=settings.CACHE_TTL)
        
        logger.info(
            "Model performance retrieved",
            model_id=model_id,
            accuracy=performance.accuracy
        )
        
        return performance
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model performance", model_id=model_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model performance: {str(e)}"
        )

@router.get("/performance")
async def get_active_model_performance(
    db: Session = Depends(get_db)
):
    """Get performance metrics for the active model"""
    
    try:
        # Get active model first
        active_model = await model_service.get_active_model(db)
        if not active_model:
            raise HTTPException(
                status_code=404,
                detail="No active model found"
            )
        
        # Get performance for active model
        return await get_model_performance(active_model.id, db)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get active model performance", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get active model performance: {str(e)}"
        )

@router.get("/calibration/{model_id}", response_model=CalibrationResponse)
async def get_model_calibration(
    model_id: int,
    bins: int = 10,
    db: Session = Depends(get_db)
):
    """Get model calibration data"""
    
    try:
        if bins < 5 or bins > 20:
            raise HTTPException(
                status_code=400,
                detail="Number of bins must be between 5 and 20"
            )
        
        # Check cache first
        cache_key = f"model_calibration:{model_id}:{bins}"
        cached_calibration = await cache_service.get(cache_key)
        
        if cached_calibration:
            logger.info("Returning cached model calibration", model_id=model_id)
            return cached_calibration
        
        # Validate model exists
        model = await model_service.get_model_by_id(model_id, db)
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model with ID {model_id} not found"
            )
        
        calibration = await model_service.get_model_calibration(
            model_id=model_id,
            bins=bins,
            db=db
        )
        
        # Cache the result
        await cache_service.set(cache_key, calibration, ttl=settings.CACHE_TTL)
        
        logger.info(
            "Model calibration retrieved",
            model_id=model_id,
            bins=bins
        )
        
        return calibration
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model calibration", model_id=model_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model calibration: {str(e)}"
        )

@router.post("/retrain", response_model=ModelRetrainingResponse)
async def retrain_model(
    request: ModelRetrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Trigger model retraining"""
    
    start_time = time.time()
    
    try:
        # Validate request
        if request.min_matches and request.min_matches < 100:
            raise HTTPException(
                status_code=400,
                detail="Minimum matches must be at least 100"
            )
        
        # Check if retraining is already in progress
        if await model_service.is_retraining_in_progress():
            raise HTTPException(
                status_code=409,
                detail="Model retraining is already in progress"
            )
        
        # Log retraining start
        model_logger.log_model_loading(
            model_version="retraining",
            model_type="dixon_coles",
            parameters=request.dict()
        )
        
        # Start retraining in background
        task_id = await model_service.start_retraining(
            request=request,
            background_tasks=background_tasks,
            db=db
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(
            "Model retraining started",
            task_id=task_id,
            force_retrain=request.force_retrain,
            processing_time_ms=processing_time
        )
        
        return ModelRetrainingResponse(
            task_id=task_id,
            status="started",
            message="Model retraining has been initiated",
            estimated_duration_minutes=request.estimated_duration or 30,
            started_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        model_logger.log_model_error(
            error_type="retraining_failed",
            error_message=str(e),
            model_version="retraining"
        )
        logger.error("Model retraining failed to start", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start model retraining: {str(e)}"
        )

@router.get("/retrain/status/{task_id}")
async def get_retraining_status(
    task_id: str,
    db: Session = Depends(get_db)
):
    """Get retraining task status"""
    
    try:
        status = await model_service.get_retraining_status(task_id, db)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Retraining task {task_id} not found"
            )
        
        logger.info(
            "Retraining status retrieved",
            task_id=task_id,
            status=status.get("status")
        )
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get retraining status", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get retraining status: {str(e)}"
        )

@router.post("/activate/{model_id}")
async def activate_model(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Activate a specific model version"""
    
    try:
        # Validate model exists
        model = await model_service.get_model_by_id(model_id, db)
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model with ID {model_id} not found"
            )
        
        # Activate the model
        success = await model_service.activate_model(model_id, db)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to activate model"
            )
        
        # Clear cache
        await cache_service.delete("current_model")
        
        logger.info(
            "Model activated",
            model_id=model_id,
            model_version=model.version
        )
        
        return {
            "message": f"Model {model.version} activated successfully",
            "model_id": model_id,
            "model_version": model.version,
            "activated_at": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to activate model", model_id=model_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to activate model: {str(e)}"
        )

@router.delete("/{model_id}")
async def delete_model(
    model_id: int,
    force: bool = False,
    db: Session = Depends(get_db)
):
    """Delete a model version (admin endpoint)"""
    
    try:
        # Validate model exists
        model = await model_service.get_model_by_id(model_id, db)
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model with ID {model_id} not found"
            )
        
        # Check if it's the current active model
        current_model = await model_service.get_current_model(db)
        if current_model and current_model.id == model_id and not force:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete active model without force=true"
            )
        
        # Delete the model
        success = await model_service.delete_model(model_id, db)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete model"
            )
        
        # Clear related cache
        await cache_service.delete_pattern(f"model*{model_id}*")
        
        logger.info(
            "Model deleted",
            model_id=model_id,
            model_version=model.version,
            force=force
        )
        
        return {
            "message": f"Model {model.version} deleted successfully",
            "model_id": model_id,
            "deleted_at": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete model", model_id=model_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete model: {str(e)}"
        )