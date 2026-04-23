#!/bin/bash
set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$PROJECT_ROOT"

echo "Building pharma-serving:local..."
docker build -f docker/Dockerfile.serving -t pharma-serving:local .

echo "Loading image into k3d cluster..."
k3d image import pharma-serving:local -c pharma-mlops

echo "Applying Kubernetes manifests..."
kubectl apply -f kubernetes/serving/namespace.yaml
kubectl apply -f kubernetes/serving/configmap.yaml
kubectl apply -f kubernetes/serving/deployment.yaml
kubectl apply -f kubernetes/serving/service.yaml
kubectl apply -f kubernetes/serving/hpa.yaml

echo "Waiting for rollout..."
kubectl rollout status deployment/serving -n pharma-prod --timeout=120s

echo ""
echo -e "${GREEN}Serving API deployed to k3s!${NC}"
echo "  kubectl get pods -n pharma-prod"
echo "  kubectl logs -n pharma-prod -l app=serving"