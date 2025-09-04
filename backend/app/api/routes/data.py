"""Data validation and source management endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any

from app.core.database import get_db
from app.services.data_service import DataService
from app.core.validation import validate_artifacts_integrity, get_approved_data_source
from app.core.logging import get_logger

logger = get_logger("data_api")
router = APIRouter()

@router.get("/source", response_model=Dict[str, Any])
async def get_data_source_info():
    """Get information about the current data source configuration."""
    try:
        source_info = get_approved_data_source()
        artifacts_valid, artifacts_errors = validate_artifacts_integrity()
        
        return {
            "status": "success",
            "data_source": source_info,
            "artifacts_status": {
                "valid": artifacts_valid,
                "errors": artifacts_errors if not artifacts_valid else [],
                "validation_enabled": source_info.get('validation_enabled', True)
            }
        }
    except Exception as e:
        logger.error(f"Error getting data source info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get data source info: {str(e)}")

@router.get("/validate", response_model=Dict[str, Any])
async def validate_data_integrity():
    """Validate the integrity of the current data source."""
    try:
        # Validate artifacts integrity
        artifacts_valid, artifacts_errors = validate_artifacts_integrity()
        
        # Get data source configuration
        source_info = get_approved_data_source()
        
        validation_results = {
            "artifacts_validation": {
                "valid": artifacts_valid,
                "errors": artifacts_errors
            },
            "source_configuration": source_info,
            "overall_status": "valid" if artifacts_valid else "invalid"
        }
        
        if not artifacts_valid:
            logger.warning(f"Data validation failed: {artifacts_errors}")
            
        return {
            "status": "success",
            "validation_results": validation_results
        }
        
    except Exception as e:
        logger.error(f"Error validating data integrity: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data validation failed: {str(e)}")

@router.get("/summary", response_model=Dict[str, Any])
async def get_data_summary(db: Session = Depends(get_db)):
    """Get comprehensive data summary including source validation."""
    try:
        data_service = DataService()
        summary = await data_service.get_data_summary(db)
        
        return {
            "status": "success",
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error getting data summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get data summary: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def check_data_health(db: Session = Depends(get_db)):
    """Check the health status of data sources and database."""
    try:
        # Check database connectivity
        try:
            from sqlalchemy import text
            db.execute(text("SELECT 1"))
            db_status = "healthy"
            db_error = None
        except Exception as e:
            db_status = "unhealthy"
            db_error = str(e)
            
        # Check artifacts integrity
        artifacts_valid, artifacts_errors = validate_artifacts_integrity()
        
        # Get data source info
        source_info = get_approved_data_source()
        
        health_status = {
            "database": {
                "status": db_status,
                "error": db_error
            },
            "artifacts": {
                "status": "healthy" if artifacts_valid else "unhealthy",
                "errors": artifacts_errors if not artifacts_valid else []
            },
            "data_source": source_info,
            "overall_health": "healthy" if db_status == "healthy" and artifacts_valid else "unhealthy"
        }
        
        return {
            "status": "success",
            "health": health_status
        }
        
    except Exception as e:
        logger.error(f"Error checking data health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")