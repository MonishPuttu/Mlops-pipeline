#!/bin/bash
# infra/install_k3d.sh
# Installs k3d and creates the pharma-mlops k3s cluster.
# k3d runs k3s inside Docker — no Linux VM needed on macOS.

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

CLUSTER_NAME="pharma-mlops"

echo "=============================="
echo "  Installing k3d + k3s cluster"
echo "=============================="
echo ""

# Install k3d via Homebrew if not already installed
if ! command -v k3d &>/dev/null; then
  echo "Installing k3d..."
  brew install k3d
else
  echo -e "${GREEN}k3d already installed: $(k3d version | head -1)${NC}"
fi

# Verify Docker is running
if ! docker info &>/dev/null; then
  echo -e "${RED}Docker Desktop is not running. Start it first.${NC}"
  exit 1
fi

# Create cluster if it doesn't exist
if k3d cluster list 2>/dev/null | grep -q "$CLUSTER_NAME"; then
  echo -e "${YELLOW}Cluster '$CLUSTER_NAME' already exists.${NC}"
  echo "Starting it if stopped..."
  k3d cluster start "$CLUSTER_NAME" 2>/dev/null || true
else
  echo "Creating k3s cluster '$CLUSTER_NAME'..."
  k3d cluster create "$CLUSTER_NAME" \
    --agents 1 \
    --k3s-arg "--disable=traefik@server:0" \
    --port "80:80@loadbalancer" \
    --port "443:443@loadbalancer" \
    --wait
fi

# Configure kubectl
echo ""
echo "Configuring kubectl..."
k3d kubeconfig merge "$CLUSTER_NAME" --kubeconfig-merge-default
kubectl config use-context "k3d-$CLUSTER_NAME"

# Verify
echo ""
echo "=============================="
echo "  Cluster status"
echo "=============================="
kubectl get nodes
echo ""
echo -e "${GREEN}k3s is running via k3d!${NC}"
echo ""
echo "Useful commands:"
echo "  k3d cluster list                      — list clusters"
echo "  k3d cluster stop $CLUSTER_NAME        — stop (saves state)"
echo "  k3d cluster start $CLUSTER_NAME       — start again"
