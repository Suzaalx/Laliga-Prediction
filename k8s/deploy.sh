#!/bin/bash

# La Liga Predictions Kubernetes Deployment Script
# This script deploys the entire application stack to Kubernetes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Check if we can connect to Kubernetes cluster
if ! kubectl cluster-info &> /dev/null; then
    print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
    exit 1
fi

print_status "Starting La Liga Predictions deployment..."

# Create namespace
print_status "Creating namespace..."
kubectl apply -f namespace.yaml

# Deploy PostgreSQL
print_status "Deploying PostgreSQL..."
kubectl apply -f postgres.yaml

# Deploy Redis
print_status "Deploying Redis..."
kubectl apply -f redis.yaml

# Wait for database services to be ready
print_status "Waiting for database services to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n laliga-predictions --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n laliga-predictions --timeout=300s

# Deploy backend
print_status "Deploying backend..."
kubectl apply -f backend.yaml

# Wait for backend to be ready
print_status "Waiting for backend to be ready..."
kubectl wait --for=condition=ready pod -l app=backend -n laliga-predictions --timeout=300s

# Deploy frontend
print_status "Deploying frontend..."
kubectl apply -f frontend.yaml

# Wait for frontend to be ready
print_status "Waiting for frontend to be ready..."
kubectl wait --for=condition=ready pod -l app=frontend -n laliga-predictions --timeout=300s

# Deploy ingress
print_status "Deploying ingress..."
kubectl apply -f ingress.yaml

print_success "Deployment completed successfully!"

# Display deployment status
print_status "Deployment status:"
kubectl get pods -n laliga-predictions
echo
kubectl get services -n laliga-predictions
echo
kubectl get ingress -n laliga-predictions

print_status "Application URLs:"
echo "Frontend: http://laliga-predictions.local"
echo "Backend API: http://laliga-predictions.local/api"
echo
print_warning "Note: Add 'laliga-predictions.local' to your /etc/hosts file pointing to your ingress controller IP"
print_warning "Example: echo '127.0.0.1 laliga-predictions.local' | sudo tee -a /etc/hosts"