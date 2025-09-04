"""Model Registry for La Liga Prediction Models.

This module provides functionality to track model versions, performance metrics,
and deployment history for the La Liga prediction system.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class ModelStatus(Enum):
    """Model deployment status."""
    TRAINING = "training"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: float
    brier_score: float
    log_loss: float
    calibration_slope: Optional[float] = None
    calibration_intercept: Optional[float] = None
    n_predictions: int = 0
    validation_period_start: Optional[str] = None
    validation_period_end: Optional[str] = None


@dataclass
class ModelVersion:
    """Model version information."""
    version: str
    created_at: datetime
    status: ModelStatus
    metrics: ModelMetrics
    artifacts_path: Optional[str] = None
    training_data_hash: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    deployed_at: Optional[datetime] = None
    deployed_by: Optional[str] = None


class ModelRegistry:
    """Model registry for tracking model versions and deployments."""
    
    def __init__(self, db_path: str = "mlops/model_registry.db"):
        """Initialize model registry.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    version TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    accuracy REAL,
                    brier_score REAL,
                    log_loss REAL,
                    calibration_slope REAL,
                    calibration_intercept REAL,
                    n_predictions INTEGER,
                    validation_period_start TEXT,
                    validation_period_end TEXT,
                    artifacts_path TEXT,
                    training_data_hash TEXT,
                    hyperparameters TEXT,
                    notes TEXT,
                    deployed_at TEXT,
                    deployed_by TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS deployment_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    deployed_at TEXT NOT NULL,
                    deployed_by TEXT,
                    status TEXT NOT NULL,
                    rollback_version TEXT,
                    notes TEXT,
                    FOREIGN KEY (version) REFERENCES model_versions (version)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_status ON model_versions (status)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_deployment_env ON deployment_history (environment)
            """)
    
    def register_model(
        self,
        version: str,
        metrics: ModelMetrics,
        status: ModelStatus = ModelStatus.TRAINING,
        artifacts_path: Optional[str] = None,
        training_data_hash: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None
    ) -> ModelVersion:
        """Register a new model version.
        
        Args:
            version: Model version identifier
            metrics: Model performance metrics
            status: Model status
            artifacts_path: Path to model artifacts
            training_data_hash: Hash of training data
            hyperparameters: Model hyperparameters
            notes: Additional notes
            
        Returns:
            ModelVersion object
        """
        model_version = ModelVersion(
            version=version,
            created_at=datetime.now(),
            status=status,
            metrics=metrics,
            artifacts_path=artifacts_path,
            training_data_hash=training_data_hash,
            hyperparameters=hyperparameters,
            notes=notes
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO model_versions (
                    version, created_at, status, accuracy, brier_score, log_loss,
                    calibration_slope, calibration_intercept, n_predictions,
                    validation_period_start, validation_period_end,
                    artifacts_path, training_data_hash, hyperparameters, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_version.version,
                model_version.created_at.isoformat(),
                model_version.status.value,
                model_version.metrics.accuracy,
                model_version.metrics.brier_score,
                model_version.metrics.log_loss,
                model_version.metrics.calibration_slope,
                model_version.metrics.calibration_intercept,
                model_version.metrics.n_predictions,
                model_version.metrics.validation_period_start,
                model_version.metrics.validation_period_end,
                model_version.artifacts_path,
                model_version.training_data_hash,
                json.dumps(model_version.hyperparameters) if model_version.hyperparameters else None,
                model_version.notes
            ))
        
        return model_version
    
    def get_model(self, version: str) -> Optional[ModelVersion]:
        """Get model version by version identifier.
        
        Args:
            version: Model version identifier
            
        Returns:
            ModelVersion object or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM model_versions WHERE version = ?",
                (version,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_model_version(row)
    
    def list_models(
        self,
        status: Optional[ModelStatus] = None,
        limit: int = 50
    ) -> List[ModelVersion]:
        """List model versions.
        
        Args:
            status: Filter by model status
            limit: Maximum number of models to return
            
        Returns:
            List of ModelVersion objects
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if status:
                cursor = conn.execute(
                    "SELECT * FROM model_versions WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status.value, limit)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM model_versions ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                )
            
            return [self._row_to_model_version(row) for row in cursor.fetchall()]
    
    def update_model_status(
        self,
        version: str,
        status: ModelStatus,
        notes: Optional[str] = None
    ) -> bool:
        """Update model status.
        
        Args:
            version: Model version identifier
            status: New model status
            notes: Additional notes
            
        Returns:
            True if updated successfully
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE model_versions SET status = ?, notes = ? WHERE version = ?",
                (status.value, notes, version)
            )
            return cursor.rowcount > 0
    
    def deploy_model(
        self,
        version: str,
        environment: str,
        deployed_by: Optional[str] = None,
        notes: Optional[str] = None
    ) -> bool:
        """Deploy model to environment.
        
        Args:
            version: Model version identifier
            environment: Deployment environment (staging, production)
            deployed_by: User who deployed the model
            notes: Deployment notes
            
        Returns:
            True if deployed successfully
        """
        with sqlite3.connect(self.db_path) as conn:
            # Update model version
            conn.execute(
                "UPDATE model_versions SET deployed_at = ?, deployed_by = ?, status = ? WHERE version = ?",
                (datetime.now().isoformat(), deployed_by, ModelStatus.PRODUCTION.value, version)
            )
            
            # Record deployment history
            conn.execute("""
                INSERT INTO deployment_history (
                    version, environment, deployed_at, deployed_by, status, notes
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                version,
                environment,
                datetime.now().isoformat(),
                deployed_by,
                "deployed",
                notes
            ))
            
            return True
    
    def get_production_model(self) -> Optional[ModelVersion]:
        """Get currently deployed production model.
        
        Returns:
            ModelVersion object or None if no production model
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM model_versions WHERE status = ? ORDER BY deployed_at DESC LIMIT 1",
                (ModelStatus.PRODUCTION.value,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_model_version(row)
    
    def get_deployment_history(
        self,
        environment: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get deployment history.
        
        Args:
            environment: Filter by environment
            limit: Maximum number of deployments to return
            
        Returns:
            List of deployment records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if environment:
                cursor = conn.execute(
                    "SELECT * FROM deployment_history WHERE environment = ? ORDER BY deployed_at DESC LIMIT ?",
                    (environment, limit)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM deployment_history ORDER BY deployed_at DESC LIMIT ?",
                    (limit,)
                )
            
            return [dict(row) for row in cursor.fetchall()]
    
    def compare_models(
        self,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare two model versions.
        
        Args:
            version1: First model version
            version2: Second model version
            
        Returns:
            Comparison results
        """
        model1 = self.get_model(version1)
        model2 = self.get_model(version2)
        
        if not model1 or not model2:
            raise ValueError("One or both model versions not found")
        
        return {
            "version1": version1,
            "version2": version2,
            "accuracy_diff": model2.metrics.accuracy - model1.metrics.accuracy,
            "brier_score_diff": model2.metrics.brier_score - model1.metrics.brier_score,
            "log_loss_diff": model2.metrics.log_loss - model1.metrics.log_loss,
            "model1_metrics": asdict(model1.metrics),
            "model2_metrics": asdict(model2.metrics),
            "better_model": version2 if model2.metrics.accuracy > model1.metrics.accuracy else version1
        }
    
    def _row_to_model_version(self, row: sqlite3.Row) -> ModelVersion:
        """Convert database row to ModelVersion object."""
        metrics = ModelMetrics(
            accuracy=row["accuracy"],
            brier_score=row["brier_score"],
            log_loss=row["log_loss"],
            calibration_slope=row["calibration_slope"],
            calibration_intercept=row["calibration_intercept"],
            n_predictions=row["n_predictions"],
            validation_period_start=row["validation_period_start"],
            validation_period_end=row["validation_period_end"]
        )
        
        return ModelVersion(
            version=row["version"],
            created_at=datetime.fromisoformat(row["created_at"]),
            status=ModelStatus(row["status"]),
            metrics=metrics,
            artifacts_path=row["artifacts_path"],
            training_data_hash=row["training_data_hash"],
            hyperparameters=json.loads(row["hyperparameters"]) if row["hyperparameters"] else None,
            notes=row["notes"],
            deployed_at=datetime.fromisoformat(row["deployed_at"]) if row["deployed_at"] else None,
            deployed_by=row["deployed_by"]
        )


def main():
    """Example usage of model registry."""
    registry = ModelRegistry()
    
    # Register a new model
    metrics = ModelMetrics(
        accuracy=0.485,
        brier_score=0.631,
        log_loss=1.245,
        n_predictions=450
    )
    
    model = registry.register_model(
        version="v20240115_120000",
        metrics=metrics,
        status=ModelStatus.TRAINING,
        notes="Enhanced Dixon-Coles model with time decay"
    )
    
    print(f"Registered model: {model.version}")
    
    # List all models
    models = registry.list_models()
    print(f"Total models: {len(models)}")
    
    # Get production model
    prod_model = registry.get_production_model()
    if prod_model:
        print(f"Production model: {prod_model.version}")
    else:
        print("No production model deployed")


if __name__ == "__main__":
    main()