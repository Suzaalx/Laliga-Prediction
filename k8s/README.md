# Kubernetes Deployment for La Liga Predictions

This directory contains Kubernetes manifests and deployment scripts for the La Liga Predictions application.

## Architecture

The application consists of the following components:

- **PostgreSQL**: Primary database for storing match data, predictions, and model metadata
- **Redis**: Caching layer for improved performance
- **Backend API**: FastAPI application serving the prediction API
- **Frontend**: Next.js web application providing the user interface
- **Ingress**: NGINX ingress controller for external access

## Prerequisites

1. **Kubernetes Cluster**: A running Kubernetes cluster (local or cloud)
2. **kubectl**: Kubernetes command-line tool configured to access your cluster
3. **NGINX Ingress Controller**: For external access (install if not present)
4. **Docker Images**: Backend and frontend images built and available

### Installing NGINX Ingress Controller

```bash
# For local development (minikube, kind, etc.)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# Wait for the ingress controller to be ready
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=120s
```

## Building Docker Images

Before deploying to Kubernetes, build the Docker images:

```bash
# Build backend image
docker build -t laliga-backend:latest ./backend

# Build frontend image
docker build -t laliga-frontend:latest ./frontend

# For cloud deployment, tag and push to your registry
# docker tag laliga-backend:latest your-registry/laliga-backend:latest
# docker push your-registry/laliga-backend:latest
```

## Deployment

### Quick Deployment

Use the provided deployment script for a one-command deployment:

```bash
./deploy.sh
```

### Manual Deployment

Deploy components step by step:

```bash
# 1. Create namespace
kubectl apply -f namespace.yaml

# 2. Deploy PostgreSQL
kubectl apply -f postgres.yaml

# 3. Deploy Redis
kubectl apply -f redis.yaml

# 4. Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n laliga-predictions --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n laliga-predictions --timeout=300s

# 5. Deploy backend
kubectl apply -f backend.yaml

# 6. Wait for backend to be ready
kubectl wait --for=condition=ready pod -l app=backend -n laliga-predictions --timeout=300s

# 7. Deploy frontend
kubectl apply -f frontend.yaml

# 8. Deploy ingress
kubectl apply -f ingress.yaml
```

## Accessing the Application

After deployment, the application will be available at:

- **Frontend**: http://laliga-predictions.local
- **Backend API**: http://laliga-predictions.local/api

### Local Development Setup

For local development, add the following to your `/etc/hosts` file:

```bash
echo '127.0.0.1 laliga-predictions.local' | sudo tee -a /etc/hosts
```

For cloud deployments, replace `127.0.0.1` with your ingress controller's external IP.

## Monitoring and Management

### Check Deployment Status

```bash
# Check all pods
kubectl get pods -n laliga-predictions

# Check services
kubectl get services -n laliga-predictions

# Check ingress
kubectl get ingress -n laliga-predictions

# Check horizontal pod autoscalers
kubectl get hpa -n laliga-predictions
```

### View Logs

```bash
# Backend logs
kubectl logs -l app=backend -n laliga-predictions -f

# Frontend logs
kubectl logs -l app=frontend -n laliga-predictions -f

# PostgreSQL logs
kubectl logs -l app=postgres -n laliga-predictions -f
```

### Scaling

The application includes Horizontal Pod Autoscalers (HPA) for automatic scaling based on CPU and memory usage. You can also manually scale:

```bash
# Scale backend
kubectl scale deployment backend --replicas=3 -n laliga-predictions

# Scale frontend
kubectl scale deployment frontend --replicas=3 -n laliga-predictions
```

## Configuration

### Environment Variables

Configuration is managed through ConfigMaps:

- `backend-config`: Backend application configuration
- `frontend-config`: Frontend application configuration
- `postgres-config`: PostgreSQL configuration
- `redis-config`: Redis configuration

### Persistent Storage

The deployment uses PersistentVolumeClaims for data persistence:

- `postgres-pvc`: PostgreSQL data (10Gi)
- `redis-pvc`: Redis data (5Gi)

## Security Considerations

1. **Secrets**: Replace hardcoded passwords with Kubernetes Secrets in production
2. **RBAC**: Implement Role-Based Access Control
3. **Network Policies**: Restrict network traffic between pods
4. **TLS**: Enable HTTPS with cert-manager for production

### Example Secret Creation

```bash
# Create database secret
kubectl create secret generic postgres-secret \
  --from-literal=username=laliga_user \
  --from-literal=password=your-secure-password \
  -n laliga-predictions

# Create Redis secret
kubectl create secret generic redis-secret \
  --from-literal=password=your-redis-password \
  -n laliga-predictions
```

## Cleanup

To remove the entire deployment:

```bash
./cleanup.sh
```

Or manually:

```bash
kubectl delete namespace laliga-predictions
```

## Troubleshooting

### Common Issues

1. **Pods stuck in Pending**: Check resource requests and cluster capacity
2. **ImagePullBackOff**: Ensure Docker images are built and accessible
3. **CrashLoopBackOff**: Check pod logs for application errors
4. **Ingress not working**: Verify NGINX ingress controller is installed

### Debug Commands

```bash
# Describe pod for detailed information
kubectl describe pod <pod-name> -n laliga-predictions

# Get events
kubectl get events -n laliga-predictions --sort-by='.lastTimestamp'

# Check resource usage
kubectl top pods -n laliga-predictions
kubectl top nodes
```

## Production Considerations

1. **Resource Limits**: Adjust CPU and memory limits based on your workload
2. **Monitoring**: Integrate with Prometheus and Grafana
3. **Backup**: Implement database backup strategies
4. **High Availability**: Deploy across multiple nodes/zones
5. **CI/CD**: Integrate with your deployment pipeline

## Files Description

- `namespace.yaml`: Kubernetes namespace definition
- `postgres.yaml`: PostgreSQL deployment, service, and PVC
- `redis.yaml`: Redis deployment, service, and PVC
- `backend.yaml`: Backend API deployment, service, and HPA
- `frontend.yaml`: Frontend deployment, service, and HPA
- `ingress.yaml`: Ingress configuration for external access
- `deploy.sh`: Automated deployment script
- `cleanup.sh`: Automated cleanup script
- `README.md`: This documentation file