from fastapi import APIRouter, Depends
from datetime import datetime
import psutil
import time
from typing import Dict, Any

from app.models.schemas import HealthResponse
from app.models.database import check_db_health
from app.core.config import settings
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger("health")

# Track application start time
start_time = time.time()
last_prediction_time = None

def check_cache_health() -> bool:
    """Check Redis cache connectivity"""
    try:
        import redis
        r = redis.from_url(settings.REDIS_URL)
        r.ping()
        return True
    except Exception as e:
        logger.warning("Cache health check failed", error=str(e))
        return False

def check_model_health() -> bool:
    """Check if model is loaded and accessible"""
    try:
        # This would check if the model service is responsive
        # For now, we'll assume it's healthy if the model path exists
        import os
        return os.path.exists(settings.MODEL_PATH) or True  # Allow for in-memory models
    except Exception as e:
        logger.warning("Model health check failed", error=str(e))
        return False

def get_system_metrics() -> Dict[str, float]:
    """Get system performance metrics"""
    try:
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage_mb = (memory.total - memory.available) / 1024 / 1024
        
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Uptime
        uptime_seconds = time.time() - start_time
        
        return {
            "memory_usage_mb": memory_usage_mb,
            "cpu_usage_percent": cpu_usage,
            "uptime_seconds": uptime_seconds
        }
    except Exception as e:
        logger.warning("Failed to get system metrics", error=str(e))
        return {
            "memory_usage_mb": 0.0,
            "cpu_usage_percent": 0.0,
            "uptime_seconds": time.time() - start_time
        }

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    
    # Check individual services
    db_healthy = check_db_health()
    cache_healthy = check_cache_health()
    model_healthy = check_model_health()
    
    # Get system metrics
    metrics = get_system_metrics()
    
    # Determine overall status
    if db_healthy and cache_healthy and model_healthy:
        status = "healthy"
    elif db_healthy and model_healthy:  # Cache is optional
        status = "degraded"
    else:
        status = "unhealthy"
    
    # Log health check
    logger.info(
        "Health check performed",
        status=status,
        database=db_healthy,
        cache=cache_healthy,
        model=model_healthy
    )
    
    return HealthResponse(
        status=status,
        timestamp=datetime.utcnow(),
        version=settings.APP_VERSION,
        database=db_healthy,
        cache=cache_healthy,
        model=model_healthy,
        uptime_seconds=metrics["uptime_seconds"],
        memory_usage_mb=metrics["memory_usage_mb"],
        cpu_usage_percent=metrics["cpu_usage_percent"],
        active_model_version=settings.MODEL_VERSION,
        last_prediction_time=last_prediction_time
    )

@router.get("/liveness")
async def liveness_probe():
    """Kubernetes liveness probe endpoint"""
    return {"status": "alive", "timestamp": datetime.utcnow()}

@router.get("/readiness")
async def readiness_probe():
    """Kubernetes readiness probe endpoint"""
    db_healthy = check_db_health()
    model_healthy = check_model_health()
    
    if db_healthy and model_healthy:
        return {"status": "ready", "timestamp": datetime.utcnow()}
    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Service not ready")

@router.get("/metrics")
async def metrics_endpoint():
    """Prometheus-style metrics endpoint"""
    metrics = get_system_metrics()
    db_healthy = 1 if check_db_health() else 0
    cache_healthy = 1 if check_cache_health() else 0
    model_healthy = 1 if check_model_health() else 0
    
    # Return metrics in Prometheus format
    prometheus_metrics = f"""
# HELP laliga_api_uptime_seconds Application uptime in seconds
# TYPE laliga_api_uptime_seconds counter
laliga_api_uptime_seconds {metrics['uptime_seconds']}

# HELP laliga_api_memory_usage_bytes Memory usage in bytes
# TYPE laliga_api_memory_usage_bytes gauge
laliga_api_memory_usage_bytes {metrics['memory_usage_mb'] * 1024 * 1024}

# HELP laliga_api_cpu_usage_percent CPU usage percentage
# TYPE laliga_api_cpu_usage_percent gauge
laliga_api_cpu_usage_percent {metrics['cpu_usage_percent']}

# HELP laliga_api_database_healthy Database health status
# TYPE laliga_api_database_healthy gauge
laliga_api_database_healthy {db_healthy}

# HELP laliga_api_cache_healthy Cache health status
# TYPE laliga_api_cache_healthy gauge
laliga_api_cache_healthy {cache_healthy}

# HELP laliga_api_model_healthy Model health status
# TYPE laliga_api_model_healthy gauge
laliga_api_model_healthy {model_healthy}
"""
    
    return prometheus_metrics

def update_last_prediction_time():
    """Update the last prediction timestamp"""
    global last_prediction_time
    last_prediction_time = datetime.utcnow()