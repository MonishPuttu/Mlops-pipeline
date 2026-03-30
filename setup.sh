#!/bin/bash
# setup.sh - Complete local environment setup for Pharma MLOps Pipeline
# Run: bash setup.sh

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║      Pharma MLOps Pipeline — Local Environment Setup             ║" 
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "→ Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "  Virtual environment created and activated."

# Upgrade pip
echo ""
echo "→ Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo ""
echo "→ Installing dependencies (this may take 2-5 minutes)..."
pip install -r requirements.txt

echo ""
echo "→ Creating directory structure..."
mkdir -p data/{raw,processed,features}
mkdir -p models/{trained,registry}
mkdir -p logs audit monitoring

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Setup complete!                                                 ║"
echo "║                                                                  ║"
echo "║  Run the pipeline:                                               ║"
echo "║    source venv/bin/activate                                      ║"
echo "║    python run.py                                                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
