#!/bin/bash

# La Liga Predictions Kubernetes Cleanup Script
# This script removes the entire application stack from Kubernetes

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

print_warning "This will delete the entire La Liga Predictions application from Kubernetes."
read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_status "Cleanup cancelled."
    exit 0
fi

print_status "Starting cleanup..."

# Delete ingress
print_status "Removing ingress..."
kubectl delete -f ingress.yaml --ignore-not-found=true

# Delete frontend
print_status "Removing frontend..."
kubectl delete -f frontend.yaml --ignore-not-found=true

# Delete backend
print_status "Removing backend..."
kubectl delete -f backend.yaml --ignore-not-found=true

# Delete Redis
print_status "Removing Redis..."
kubectl delete -f redis.yaml --ignore-not-found=true

# Delete PostgreSQL
print_status "Removing PostgreSQL..."
kubectl delete -f postgres.yaml --ignore-not-found=true

# Delete namespace (this will also delete any remaining resources)
print_status "Removing namespace..."
kubectl delete -f namespace.yaml --ignore-not-found=true

print_success "Cleanup completed successfully!"

# Check if any resources remain
print_status "Checking for remaining resources..."
REMAINING=$(kubectl get all -n laliga-predictions 2>/dev/null | wc -l)
if [ $REMAINING -gt 0 ]; then
    print_warning "Some resources may still be terminating. Run 'kubectl get all -n laliga-predictions' to check."
else
    print_success "All resources have been removed."
fi