import logging
import sys
from typing import Any, Dict
import structlog
from structlog.stdlib import LoggerFactory

from app.core.config import settings

def setup_logging() -> None:
    """Configure structured logging for the application"""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.LOG_FORMAT == "json" else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.upper()),
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance"""
    return structlog.get_logger(name)

class RequestLoggingMiddleware:
    """Middleware for logging HTTP requests"""
    
    def __init__(self, app):
        self.app = app
        self.logger = get_logger("api.requests")
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Extract request info
        method = scope["method"]
        path = scope["path"]
        query_string = scope.get("query_string", b"").decode()
        client_ip = scope.get("client", ["unknown", None])[0]
        
        # Log request start
        self.logger.info(
            "Request started",
            method=method,
            path=path,
            query_string=query_string,
            client_ip=client_ip
        )
        
        # Process request
        await self.app(scope, receive, send)
        
        # Log request completion
        self.logger.info(
            "Request completed",
            method=method,
            path=path,
            client_ip=client_ip
        )

class ModelLoggingMixin:
    """Mixin for model-related logging"""
    
    def __init__(self):
        self.logger = get_logger("model")
    
    def log_prediction(self, match_id: str, home_team: str, away_team: str, 
                      probabilities: Dict[str, float], model_version: str) -> None:
        """Log prediction details"""
        self.logger.info(
            "Prediction generated",
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            home_win_prob=probabilities.get("home_win"),
            draw_prob=probabilities.get("draw"),
            away_win_prob=probabilities.get("away_win"),
            model_version=model_version
        )
    
    def log_model_load(self, model_path: str, model_version: str) -> None:
        """Log model loading"""
        self.logger.info(
            "Model loaded",
            model_path=model_path,
            model_version=model_version
        )
    
    def log_model_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log model errors"""
        self.logger.error(
            "Model error occurred",
            error=str(error),
            error_type=type(error).__name__,
            **context
        )