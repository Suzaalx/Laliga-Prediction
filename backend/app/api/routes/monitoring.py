from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
import os

from app.core.logging import get_logger

router = APIRouter()
logger = get_logger("monitoring")

@router.get("/dashboard")
async def monitoring_dashboard(days: int = 7) -> Dict[str, Any]:
    """Get monitoring dashboard data
    
    Args:
        days: Number of days to look back for metrics
        
    Returns:
        Comprehensive monitoring dashboard data
    """
    try:
        # Import monitoring system
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../../mlops'))
        
        from monitoring import ModelMonitor
        
        # Initialize monitor
        monitor = ModelMonitor()
        
        # Get dashboard data
        dashboard_data = monitor.get_monitoring_dashboard_data(days=days)
        
        # Add service health check
        services = {
            "backend": "http://localhost:8000/health",
            "frontend": "http://localhost:3000"
        }
        
        health_metrics = monitor.monitor_service_health(services)
        dashboard_data["current_health"] = [{
            "service_name": m.service_name,
            "status": m.status,
            "response_time_ms": m.response_time_ms,
            "error_rate": m.error_rate
        } for m in health_metrics]
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error getting monitoring dashboard data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get monitoring data: {str(e)}"
        )

@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get application metrics for monitoring"""
    import time
    return {
        "uptime": time.time(),
        "version": "1.0.0",
        "status": "operational"
    }