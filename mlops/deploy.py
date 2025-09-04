"""Deployment automation for La Liga prediction models.

This module provides automated deployment functionality with health checks,
rollback capabilities, and integration with the model registry.
"""

import os
import time
import json
import subprocess
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from model_registry import ModelRegistry, ModelStatus, ModelMetrics


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    environment: str
    backend_url: str
    frontend_url: str
    health_check_timeout: int = 300
    health_check_interval: int = 10
    rollback_on_failure: bool = True
    notification_webhook: Optional[str] = None


class DeploymentError(Exception):
    """Deployment error."""
    pass


class ModelDeployer:
    """Automated model deployment with health checks and rollback."""
    
    def __init__(self, config: DeploymentConfig, registry: ModelRegistry):
        """Initialize deployer.
        
        Args:
            config: Deployment configuration
            registry: Model registry instance
        """
        self.config = config
        self.registry = registry
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(f"deployer_{self.config.environment}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def deploy_model(
        self,
        version: str,
        deployed_by: Optional[str] = None,
        force: bool = False
    ) -> bool:
        """Deploy model version to environment.
        
        Args:
            version: Model version to deploy
            deployed_by: User deploying the model
            force: Force deployment even if health checks fail
            
        Returns:
            True if deployment successful
        """
        self.logger.info(f"Starting deployment of model {version} to {self.config.environment}")
        
        # Get model from registry
        model = self.registry.get_model(version)
        if not model:
            raise DeploymentError(f"Model version {version} not found in registry")
        
        # Check if model is ready for deployment
        if model.status not in [ModelStatus.STAGING, ModelStatus.VALIDATION] and not force:
            raise DeploymentError(f"Model {version} status is {model.status.value}, not ready for deployment")
        
        # Get current production model for rollback
        current_model = self.registry.get_production_model()
        
        try:
            # Deploy backend
            self.logger.info("Deploying backend service...")
            self._deploy_backend(version)
            
            # Deploy frontend
            self.logger.info("Deploying frontend service...")
            self._deploy_frontend(version)
            
            # Health checks
            self.logger.info("Running health checks...")
            if not self._run_health_checks() and not force:
                raise DeploymentError("Health checks failed")
            
            # Update model registry
            self.registry.deploy_model(
                version=version,
                environment=self.config.environment,
                deployed_by=deployed_by,
                notes=f"Deployed to {self.config.environment}"
            )
            
            self.logger.info(f"Successfully deployed model {version}")
            self._send_notification(f"✅ Model {version} deployed successfully to {self.config.environment}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            
            if self.config.rollback_on_failure and current_model:
                self.logger.info(f"Rolling back to model {current_model.version}")
                self._rollback(current_model.version)
            
            self._send_notification(f"❌ Model {version} deployment failed: {e}")
            raise
    
    def _deploy_backend(self, version: str) -> None:
        """Deploy backend service.
        
        Args:
            version: Model version to deploy
        """
        if self.config.environment == "production":
            # Production deployment using Docker/Kubernetes
            self._deploy_with_kubernetes("backend", version)
        else:
            # Staging deployment
            self._deploy_with_docker_compose("backend", version)
    
    def _deploy_frontend(self, version: str) -> None:
        """Deploy frontend service.
        
        Args:
            version: Model version to deploy
        """
        if self.config.environment == "production":
            # Production deployment using Docker/Kubernetes
            self._deploy_with_kubernetes("frontend", version)
        else:
            # Staging deployment
            self._deploy_with_docker_compose("frontend", version)
    
    def _deploy_with_kubernetes(self, service: str, version: str) -> None:
        """Deploy service using Kubernetes.
        
        Args:
            service: Service name (backend/frontend)
            version: Model version
        """
        # Update Kubernetes deployment
        image_tag = f"ghcr.io/your-org/laliga-{service}:{version}"
        
        cmd = [
            "kubectl", "set", "image",
            f"deployment/laliga-{service}",
            f"{service}={image_tag}",
            "-n", self.config.environment
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise DeploymentError(f"Kubernetes deployment failed: {result.stderr}")
        
        # Wait for rollout to complete
        cmd = ["kubectl", "rollout", "status", f"deployment/laliga-{service}", "-n", self.config.environment]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise DeploymentError(f"Kubernetes rollout failed: {result.stderr}")
    
    def _deploy_with_docker_compose(self, service: str, version: str) -> None:
        """Deploy service using Docker Compose.
        
        Args:
            service: Service name (backend/frontend)
            version: Model version
        """
        # Update docker-compose.yml with new image tag
        compose_file = Path("docker-compose.yml")
        if not compose_file.exists():
            raise DeploymentError("docker-compose.yml not found")
        
        # Set environment variable for image tag
        os.environ[f"{service.upper()}_IMAGE_TAG"] = version
        
        # Deploy with docker-compose
        cmd = ["docker-compose", "up", "-d", service]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise DeploymentError(f"Docker Compose deployment failed: {result.stderr}")
    
    def _run_health_checks(self) -> bool:
        """Run health checks on deployed services.
        
        Returns:
            True if all health checks pass
        """
        start_time = time.time()
        
        while time.time() - start_time < self.config.health_check_timeout:
            try:
                # Check backend health
                backend_healthy = self._check_backend_health()
                
                # Check frontend health
                frontend_healthy = self._check_frontend_health()
                
                if backend_healthy and frontend_healthy:
                    self.logger.info("All health checks passed")
                    return True
                
                self.logger.info(f"Health check failed - Backend: {backend_healthy}, Frontend: {frontend_healthy}")
                
            except Exception as e:
                self.logger.warning(f"Health check error: {e}")
            
            time.sleep(self.config.health_check_interval)
        
        self.logger.error("Health checks timed out")
        return False
    
    def _check_backend_health(self) -> bool:
        """Check backend service health.
        
        Returns:
            True if backend is healthy
        """
        try:
            response = requests.get(f"{self.config.backend_url}/health", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def _check_frontend_health(self) -> bool:
        """Check frontend service health.
        
        Returns:
            True if frontend is healthy
        """
        try:
            response = requests.get(self.config.frontend_url, timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def _rollback(self, version: str) -> None:
        """Rollback to previous model version.
        
        Args:
            version: Version to rollback to
        """
        try:
            self.logger.info(f"Rolling back to version {version}")
            
            # Deploy previous version
            self._deploy_backend(version)
            self._deploy_frontend(version)
            
            # Update registry
            self.registry.deploy_model(
                version=version,
                environment=self.config.environment,
                notes=f"Rollback deployment to {self.config.environment}"
            )
            
            self.logger.info(f"Rollback to {version} completed")
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            raise DeploymentError(f"Rollback failed: {e}")
    
    def _send_notification(self, message: str) -> None:
        """Send deployment notification.
        
        Args:
            message: Notification message
        """
        if not self.config.notification_webhook:
            return
        
        try:
            payload = {
                "text": message,
                "environment": self.config.environment,
                "timestamp": datetime.now().isoformat()
            }
            
            requests.post(
                self.config.notification_webhook,
                json=payload,
                timeout=10
            )
        except Exception as e:
            self.logger.warning(f"Failed to send notification: {e}")
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status.
        
        Returns:
            Deployment status information
        """
        current_model = self.registry.get_production_model()
        
        status = {
            "environment": self.config.environment,
            "current_model": current_model.version if current_model else None,
            "deployed_at": current_model.deployed_at.isoformat() if current_model and current_model.deployed_at else None,
            "deployed_by": current_model.deployed_by if current_model else None,
            "backend_healthy": self._check_backend_health(),
            "frontend_healthy": self._check_frontend_health(),
            "last_check": datetime.now().isoformat()
        }
        
        if current_model:
            status["model_metrics"] = {
                "accuracy": current_model.metrics.accuracy,
                "brier_score": current_model.metrics.brier_score,
                "log_loss": current_model.metrics.log_loss
            }
        
        return status


def create_deployment_configs() -> Dict[str, DeploymentConfig]:
    """Create deployment configurations for different environments.
    
    Returns:
        Dictionary of deployment configurations
    """
    return {
        "staging": DeploymentConfig(
            environment="staging",
            backend_url="http://staging-api.laliga-predictions.com",
            frontend_url="http://staging.laliga-predictions.com",
            health_check_timeout=180,
            rollback_on_failure=True
        ),
        "production": DeploymentConfig(
            environment="production",
            backend_url="http://api.laliga-predictions.com",
            frontend_url="http://laliga-predictions.com",
            health_check_timeout=300,
            rollback_on_failure=True,
            notification_webhook=os.getenv("SLACK_WEBHOOK_URL")
        )
    }


def main():
    """Example deployment script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy La Liga prediction model")
    parser.add_argument("version", help="Model version to deploy")
    parser.add_argument("--environment", choices=["staging", "production"], default="staging")
    parser.add_argument("--deployed-by", help="User deploying the model")
    parser.add_argument("--force", action="store_true", help="Force deployment")
    
    args = parser.parse_args()
    
    # Setup
    registry = ModelRegistry()
    configs = create_deployment_configs()
    config = configs[args.environment]
    
    deployer = ModelDeployer(config, registry)
    
    try:
        # Deploy model
        success = deployer.deploy_model(
            version=args.version,
            deployed_by=args.deployed_by,
            force=args.force
        )
        
        if success:
            print(f"✅ Model {args.version} deployed successfully to {args.environment}")
            
            # Show deployment status
            status = deployer.get_deployment_status()
            print(json.dumps(status, indent=2))
        else:
            print(f"❌ Deployment failed")
            exit(1)
            
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        exit(1)


if __name__ == "__main__":
    main()