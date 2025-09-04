"""La Liga Prediction API - FastAPI Application

This module implements the main FastAPI application for the La Liga prediction system.
It provides RESTful endpoints for match predictions, team statistics, and model insights
using an enhanced Dixon-Coles model with comprehensive feature engineering.

Key Features:
- Match prediction endpoints with probability distributions
- Team performance analytics and statistics
- Model metadata and performance metrics
- Real-time monitoring and health checks
- Comprehensive error handling and logging
- CORS support for frontend integration
- Redis caching for performance optimization
- Database connection management

API Endpoints:
- /api/predictions/* - Match prediction services
- /api/teams/* - Team statistics and analytics
- /api/model/* - Model information and performance
- /health - Service health monitoring
- /metrics - Application metrics
- /monitoring/dashboard - Monitoring dashboard data

The API is designed for production use with proper error handling,
logging, security middleware, and performance monitoring.
"""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging
import time
import traceback
from typing import Dict, Any
import uvicorn

from app.core.config import settings
from app.core.database import engine, get_db
from app.models.database import Base
from app.api import api_router
from app.services import CacheService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global cache service instance
cache_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global cache_service
    
    # Startup
    logger.info("Starting up La Liga Prediction API...")
    
    try:
        # Create database tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Initialize cache service
        cache_service = CacheService()
        await cache_service.connect()
        logger.info("Cache service connected successfully")
        
        # Store cache service in app state
        app.state.cache_service = cache_service
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down La Liga Prediction API...")
    
    try:
        if cache_service:
            await cache_service.disconnect()
            logger.info("Cache service disconnected successfully")
        
        logger.info("Application shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")
        logger.error(traceback.format_exc())

# Create FastAPI application
app = FastAPI(
    title="La Liga Prediction API",
    description="Advanced football match prediction system using Dixon-Coles model with comprehensive feature engineering",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Add trusted host middleware for security
if not settings.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    
    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'}"
    )
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log response
        logger.info(
            f"Response: {response.status_code} for {request.method} {request.url.path} "
            f"in {process_time:.4f}s"
        )
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Error processing {request.method} {request.url.path} "
            f"after {process_time:.4f}s: {str(e)}"
        )
        logger.error(traceback.format_exc())
        raise

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(
        f"HTTP {exc.status_code} error for {request.method} {request.url.path}: {exc.detail}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_error",
                "code": exc.status_code,
                "message": exc.detail,
                "path": str(request.url.path),
                "method": request.method,
                "timestamp": time.time()
            }
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    logger.warning(
        f"Validation error for {request.method} {request.url.path}: {exc.errors()}"
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "type": "validation_error",
                "code": 422,
                "message": "Request validation failed",
                "details": exc.errors(),
                "path": str(request.url.path),
                "method": request.method,
                "timestamp": time.time()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(
        f"Unhandled exception for {request.method} {request.url.path}: {str(exc)}"
    )
    logger.error(traceback.format_exc())
    
    # Don't expose internal errors in production
    error_detail = str(exc) if settings.DEBUG else "Internal server error"
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "type": "internal_error",
                "code": 500,
                "message": error_detail,
                "path": str(request.url.path),
                "method": request.method,
                "timestamp": time.time()
            }
        }
    )

# Include API router
app.include_router(api_router, prefix="/api")

# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """Root endpoint with API information"""
    return {
        "name": "La Liga Prediction API",
        "version": "1.0.0",
        "description": "Advanced football match prediction system",
        "status": "operational",
        "endpoints": {
            "health": "/api/health/",
            "predictions": "/api/predictions/",
            "teams": "/api/teams/",
            "models": "/api/models/",
            "docs": "/docs" if settings.DEBUG else "disabled",
            "redoc": "/redoc" if settings.DEBUG else "disabled"
        },
        "features": [
            "Dixon-Coles probabilistic model",
            "Advanced feature engineering",
            "Real-time predictions",
            "Team statistics and form analysis",
            "Model performance tracking",
            "Automated retraining",
            "Comprehensive caching",
            "Health monitoring"
        ],
        "timestamp": time.time()
    }

# Health check endpoint (simple version)
@app.get("/health", tags=["Health"])
async def simple_health_check() -> Dict[str, str]:
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": str(time.time())
    }



if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning",
        access_log=settings.DEBUG
    )