"""Monitoring system for La Liga prediction models.

This module provides comprehensive monitoring for model performance,
feature drift detection, and service health metrics.
"""

import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
import sqlite3
from pathlib import Path

import requests
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss

from model_registry import ModelRegistry, ModelMetrics


@dataclass
class PerformanceAlert:
    """Performance alert data."""
    alert_type: str
    severity: str  # low, medium, high, critical
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    model_version: Optional[str] = None


@dataclass
class DriftAlert:
    """Feature drift alert data."""
    feature_name: str
    drift_score: float
    threshold: float
    p_value: float
    severity: str
    timestamp: datetime
    model_version: Optional[str] = None


@dataclass
class ServiceHealthMetrics:
    """Service health metrics."""
    service_name: str
    status: str  # healthy, degraded, unhealthy
    response_time_ms: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    timestamp: datetime


class ModelMonitor:
    """Model performance and drift monitoring."""
    
    def __init__(self, db_path: str = "monitoring.db"):
        """Initialize monitor.
        
        Args:
            db_path: Path to monitoring database
        """
        self.db_path = db_path
        self.logger = self._setup_logging()
        self._init_database()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("model_monitor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _init_database(self) -> None:
        """Initialize monitoring database."""
        with sqlite3.connect(self.db_path) as conn:
            # Performance metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp DATETIME,
                    environment TEXT
                )
            """)
            
            # Feature drift table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_drift (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT,
                    feature_name TEXT,
                    drift_score REAL,
                    p_value REAL,
                    timestamp DATETIME,
                    environment TEXT
                )
            """)
            
            # Alerts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    model_version TEXT,
                    timestamp DATETIME,
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Service health table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS service_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_name TEXT,
                    status TEXT,
                    response_time_ms REAL,
                    error_rate REAL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    timestamp DATETIME
                )
            """)
            
            conn.commit()
    
    def track_prediction_performance(
        self,
        predictions: List[Dict[str, Any]],
        actuals: List[Dict[str, Any]],
        model_version: str,
        environment: str = "production"
    ) -> ModelMetrics:
        """Track model prediction performance.
        
        Args:
            predictions: List of prediction dictionaries
            actuals: List of actual result dictionaries
            model_version: Model version
            environment: Environment name
            
        Returns:
            Calculated performance metrics
        """
        # Convert to arrays for calculation
        pred_probs = []
        actual_outcomes = []
        
        for pred, actual in zip(predictions, actuals):
            # Extract probabilities and outcomes
            home_prob = pred.get('home_win_prob', 0)
            draw_prob = pred.get('draw_prob', 0)
            away_prob = pred.get('away_win_prob', 0)
            
            # Actual outcome (0=away, 1=draw, 2=home)
            home_goals = actual.get('home_goals', 0)
            away_goals = actual.get('away_goals', 0)
            
            if home_goals > away_goals:
                outcome = 2  # home win
            elif home_goals < away_goals:
                outcome = 0  # away win
            else:
                outcome = 1  # draw
            
            pred_probs.append([away_prob, draw_prob, home_prob])
            actual_outcomes.append(outcome)
        
        pred_probs = np.array(pred_probs)
        actual_outcomes = np.array(actual_outcomes)
        
        # Calculate metrics
        accuracy = self._calculate_accuracy(pred_probs, actual_outcomes)
        brier_score = self._calculate_brier_score(pred_probs, actual_outcomes)
        log_loss_score = self._calculate_log_loss(pred_probs, actual_outcomes)
        calibration_error = self._calculate_calibration_error(pred_probs, actual_outcomes)
        
        metrics = ModelMetrics(
            accuracy=accuracy,
            brier_score=brier_score,
            log_loss=log_loss_score,
            calibration_error=calibration_error
        )
        
        # Store metrics
        self._store_performance_metrics(metrics, model_version, environment)
        
        # Check for performance alerts
        self._check_performance_alerts(metrics, model_version)
        
        return metrics
    
    def _calculate_accuracy(self, pred_probs: np.ndarray, actual_outcomes: np.ndarray) -> float:
        """Calculate prediction accuracy."""
        predicted_outcomes = np.argmax(pred_probs, axis=1)
        return np.mean(predicted_outcomes == actual_outcomes)
    
    def _calculate_brier_score(self, pred_probs: np.ndarray, actual_outcomes: np.ndarray) -> float:
        """Calculate Brier score."""
        # Convert to one-hot encoding
        n_classes = pred_probs.shape[1]
        actual_one_hot = np.zeros((len(actual_outcomes), n_classes))
        actual_one_hot[np.arange(len(actual_outcomes)), actual_outcomes] = 1
        
        return np.mean(np.sum((pred_probs - actual_one_hot) ** 2, axis=1))
    
    def _calculate_log_loss(self, pred_probs: np.ndarray, actual_outcomes: np.ndarray) -> float:
        """Calculate log loss."""
        # Clip probabilities to avoid log(0)
        pred_probs_clipped = np.clip(pred_probs, 1e-15, 1 - 1e-15)
        return log_loss(actual_outcomes, pred_probs_clipped)
    
    def _calculate_calibration_error(self, pred_probs: np.ndarray, actual_outcomes: np.ndarray) -> float:
        """Calculate expected calibration error."""
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Get predictions in this bin
            max_probs = np.max(pred_probs, axis=1)
            in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (np.argmax(pred_probs[in_bin], axis=1) == actual_outcomes[in_bin]).mean()
                avg_confidence_in_bin = max_probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def detect_feature_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        model_version: str,
        environment: str = "production",
        threshold: float = 0.05
    ) -> List[DriftAlert]:
        """Detect feature drift using statistical tests.
        
        Args:
            reference_data: Reference dataset (training data)
            current_data: Current dataset (recent predictions)
            model_version: Model version
            environment: Environment name
            threshold: P-value threshold for drift detection
            
        Returns:
            List of drift alerts
        """
        alerts = []
        
        # Numeric features
        numeric_features = reference_data.select_dtypes(include=[np.number]).columns
        
        for feature in numeric_features:
            if feature in current_data.columns:
                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(
                    reference_data[feature].dropna(),
                    current_data[feature].dropna()
                )
                
                if p_value < threshold:
                    severity = self._get_drift_severity(p_value, threshold)
                    
                    alert = DriftAlert(
                        feature_name=feature,
                        drift_score=ks_stat,
                        threshold=threshold,
                        p_value=p_value,
                        severity=severity,
                        timestamp=datetime.now(),
                        model_version=model_version
                    )
                    
                    alerts.append(alert)
                    
                    # Store drift data
                    self._store_drift_data(alert, environment)
        
        # Categorical features
        categorical_features = reference_data.select_dtypes(include=['object', 'category']).columns
        
        for feature in categorical_features:
            if feature in current_data.columns:
                # Chi-square test
                ref_counts = reference_data[feature].value_counts()
                curr_counts = current_data[feature].value_counts()
                
                # Align categories
                all_categories = set(ref_counts.index) | set(curr_counts.index)
                ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
                
                if sum(ref_aligned) > 0 and sum(curr_aligned) > 0:
                    chi2_stat, p_value = stats.chisquare(curr_aligned, ref_aligned)
                    
                    if p_value < threshold:
                        severity = self._get_drift_severity(p_value, threshold)
                        
                        alert = DriftAlert(
                            feature_name=feature,
                            drift_score=chi2_stat,
                            threshold=threshold,
                            p_value=p_value,
                            severity=severity,
                            timestamp=datetime.now(),
                            model_version=model_version
                        )
                        
                        alerts.append(alert)
                        
                        # Store drift data
                        self._store_drift_data(alert, environment)
        
        return alerts
    
    def _get_drift_severity(self, p_value: float, threshold: float) -> str:
        """Determine drift severity based on p-value."""
        if p_value < threshold * 0.1:
            return "critical"
        elif p_value < threshold * 0.3:
            return "high"
        elif p_value < threshold * 0.6:
            return "medium"
        else:
            return "low"
    
    def monitor_service_health(
        self,
        services: Dict[str, str]
    ) -> List[ServiceHealthMetrics]:
        """Monitor service health metrics.
        
        Args:
            services: Dictionary of service names to health check URLs
            
        Returns:
            List of service health metrics
        """
        health_metrics = []
        
        for service_name, health_url in services.items():
            try:
                start_time = time.time()
                response = requests.get(health_url, timeout=10)
                response_time = (time.time() - start_time) * 1000
                
                # Determine status
                if response.status_code == 200:
                    status = "healthy"
                elif response.status_code < 500:
                    status = "degraded"
                else:
                    status = "unhealthy"
                
                # Mock CPU and memory usage (in production, get from monitoring system)
                cpu_usage = np.random.uniform(10, 80)  # Replace with actual metrics
                memory_usage = np.random.uniform(20, 90)  # Replace with actual metrics
                
                metrics = ServiceHealthMetrics(
                    service_name=service_name,
                    status=status,
                    response_time_ms=response_time,
                    error_rate=0.0 if status == "healthy" else 0.1,
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    timestamp=datetime.now()
                )
                
                health_metrics.append(metrics)
                
                # Store health metrics
                self._store_health_metrics(metrics)
                
            except Exception as e:
                self.logger.error(f"Health check failed for {service_name}: {e}")
                
                metrics = ServiceHealthMetrics(
                    service_name=service_name,
                    status="unhealthy",
                    response_time_ms=10000,  # Timeout
                    error_rate=1.0,
                    cpu_usage=0,
                    memory_usage=0,
                    timestamp=datetime.now()
                )
                
                health_metrics.append(metrics)
                self._store_health_metrics(metrics)
        
        return health_metrics
    
    def _store_performance_metrics(
        self,
        metrics: ModelMetrics,
        model_version: str,
        environment: str
    ) -> None:
        """Store performance metrics in database."""
        with sqlite3.connect(self.db_path) as conn:
            timestamp = datetime.now()
            
            metrics_data = [
                (model_version, "accuracy", metrics.accuracy, timestamp, environment),
                (model_version, "brier_score", metrics.brier_score, timestamp, environment),
                (model_version, "log_loss", metrics.log_loss, timestamp, environment),
                (model_version, "calibration_error", metrics.calibration_error, timestamp, environment)
            ]
            
            conn.executemany(
                "INSERT INTO performance_metrics (model_version, metric_name, metric_value, timestamp, environment) VALUES (?, ?, ?, ?, ?)",
                metrics_data
            )
    
    def _store_drift_data(self, alert: DriftAlert, environment: str) -> None:
        """Store drift data in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO feature_drift (model_version, feature_name, drift_score, p_value, timestamp, environment) VALUES (?, ?, ?, ?, ?, ?)",
                (alert.model_version, alert.feature_name, alert.drift_score, alert.p_value, alert.timestamp, environment)
            )
    
    def _store_health_metrics(self, metrics: ServiceHealthMetrics) -> None:
        """Store health metrics in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO service_health (service_name, status, response_time_ms, error_rate, cpu_usage, memory_usage, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (metrics.service_name, metrics.status, metrics.response_time_ms, metrics.error_rate, metrics.cpu_usage, metrics.memory_usage, metrics.timestamp)
            )
    
    def _check_performance_alerts(
        self,
        metrics: ModelMetrics,
        model_version: str
    ) -> None:
        """Check for performance-based alerts."""
        alerts = []
        
        # Define thresholds
        thresholds = {
            "accuracy": {"min": 0.4, "critical": 0.35},
            "brier_score": {"max": 0.7, "critical": 0.8},
            "log_loss": {"max": 1.2, "critical": 1.5},
            "calibration_error": {"max": 0.15, "critical": 0.25}
        }
        
        # Check accuracy
        if metrics.accuracy < thresholds["accuracy"]["critical"]:
            severity = "critical"
        elif metrics.accuracy < thresholds["accuracy"]["min"]:
            severity = "high"
        else:
            severity = None
        
        if severity:
            alert = PerformanceAlert(
                alert_type="performance",
                severity=severity,
                message=f"Model accuracy dropped to {metrics.accuracy:.3f}",
                metric_name="accuracy",
                current_value=metrics.accuracy,
                threshold=thresholds["accuracy"]["min"],
                timestamp=datetime.now(),
                model_version=model_version
            )
            alerts.append(alert)
        
        # Check Brier score
        if metrics.brier_score > thresholds["brier_score"]["critical"]:
            severity = "critical"
        elif metrics.brier_score > thresholds["brier_score"]["max"]:
            severity = "high"
        else:
            severity = None
        
        if severity:
            alert = PerformanceAlert(
                alert_type="performance",
                severity=severity,
                message=f"Model Brier score increased to {metrics.brier_score:.3f}",
                metric_name="brier_score",
                current_value=metrics.brier_score,
                threshold=thresholds["brier_score"]["max"],
                timestamp=datetime.now(),
                model_version=model_version
            )
            alerts.append(alert)
        
        # Store alerts
        for alert in alerts:
            self._store_alert(alert)
    
    def _store_alert(self, alert: PerformanceAlert) -> None:
        """Store alert in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO alerts (alert_type, severity, message, model_version, timestamp) VALUES (?, ?, ?, ?, ?)",
                (alert.alert_type, alert.severity, alert.message, alert.model_version, alert.timestamp)
            )
    
    def get_monitoring_dashboard_data(self, days: int = 7) -> Dict[str, Any]:
        """Get monitoring dashboard data.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dashboard data dictionary
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Performance metrics
            perf_df = pd.read_sql_query(
                "SELECT * FROM performance_metrics WHERE timestamp > ?",
                conn, params=(cutoff_date,)
            )
            
            # Feature drift
            drift_df = pd.read_sql_query(
                "SELECT * FROM feature_drift WHERE timestamp > ?",
                conn, params=(cutoff_date,)
            )
            
            # Alerts
            alerts_df = pd.read_sql_query(
                "SELECT * FROM alerts WHERE timestamp > ? AND resolved = FALSE",
                conn, params=(cutoff_date,)
            )
            
            # Service health
            health_df = pd.read_sql_query(
                "SELECT * FROM service_health WHERE timestamp > ?",
                conn, params=(cutoff_date,)
            )
        
        return {
            "performance_metrics": perf_df.to_dict("records") if not perf_df.empty else [],
            "feature_drift": drift_df.to_dict("records") if not drift_df.empty else [],
            "active_alerts": alerts_df.to_dict("records") if not alerts_df.empty else [],
            "service_health": health_df.to_dict("records") if not health_df.empty else [],
            "summary": {
                "total_alerts": len(alerts_df),
                "critical_alerts": len(alerts_df[alerts_df["severity"] == "critical"]) if not alerts_df.empty else 0,
                "drift_features": len(drift_df["feature_name"].unique()) if not drift_df.empty else 0,
                "unhealthy_services": len(health_df[health_df["status"] == "unhealthy"]) if not health_df.empty else 0
            }
        }


def main():
    """Example monitoring script."""
    monitor = ModelMonitor()
    
    # Example service health monitoring
    services = {
        "backend": "http://localhost:8000/health",
        "frontend": "http://localhost:3000"
    }
    
    health_metrics = monitor.monitor_service_health(services)
    
    for metrics in health_metrics:
        print(f"{metrics.service_name}: {metrics.status} ({metrics.response_time_ms:.1f}ms)")
    
    # Get dashboard data
    dashboard_data = monitor.get_monitoring_dashboard_data()
    print(json.dumps(dashboard_data["summary"], indent=2))


if __name__ == "__main__":
    main()