from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    APP_NAME: str = "LaLiga Predictions API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1", "0.0.0.0"]
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/laliga_predictions"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    # Redis Cache
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_ENABLED: bool = True
    CACHE_TTL: int = 3600  # 1 hour
    
    # Model Configuration
    MODEL_PATH: str = "../artifacts/dixon_coles_model.pkl"
    MODEL_VERSION: str = "1.0.0"
    MIN_TRAIN_MATCHES: int = 380
    TIME_DECAY_FACTOR: float = 0.01
    
    # Data Source Configuration
    DATA_SOURCE_TYPE: str = "artifacts"  # Only allow 'artifacts' for production
    ARTIFACTS_DIR: str = "../artifacts"
    MATCHES_CSV_FILE: str = "matches_clean.csv"
    ALLOWED_DATA_SOURCES: List[str] = ["artifacts"]  # Restrict to artifacts only
    VALIDATE_DATA_SOURCE: bool = True  # Enforce data source validation
    
    # Prediction Settings
    MAX_PREDICTION_DAYS: int = 30
    CONFIDENCE_LEVEL: float = 0.95
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # External APIs (for future enhancements)
    FOOTBALL_API_KEY: Optional[str] = None
    WEATHER_API_KEY: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()

# Environment-specific configurations
class DevelopmentSettings(Settings):
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    ALLOWED_HOSTS: List[str] = ["*"]

class ProductionSettings(Settings):
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 40

class TestingSettings(Settings):
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/laliga_predictions_test"
    REDIS_URL: str = "redis://localhost:6379/1"
    CACHE_TTL: int = 60

def get_settings() -> Settings:
    """Get settings based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()