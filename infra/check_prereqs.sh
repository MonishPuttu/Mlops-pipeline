#!/bin/bash
# infra/check_prereqs.sh
# macOS Apple Silicon version

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0

check() {
  local name="$1"
  local cmd="$2"
  if eval "$cmd" &>/dev/null; then
    echo -e "${GREEN}[PASS]${NC} $name"
    PASS=$((PASS+1))
  else
    echo -e "${RED}[FAIL]${NC} $name"
    FAIL=$((FAIL+1))
  fi
}

echo "=============================="
echo "  Day 1 — Prerequisites Check"
echo "  macOS Apple Silicon"
echo "=============================="
echo ""

check "Homebrew installed"       "brew --version"
check "Docker Desktop installed" "docker --version"
check "Docker daemon running"    "docker info"
check "Docker Compose v2"        "docker compose version"
check "kubectl installed"        "kubectl version --client"
check "Helm 3 installed"         "helm version"
check "k3d installed"            "k3d version"
check "Python 3.11+"             "python3 --version | grep -E '3\.(11|12|13)'"
check "pip installed"            "pip3 --version"
check "git installed"            "git --version"
check "make installed"           "make --version"

echo ""
echo "── Hardware ──────────────────────────"
check "Apple Silicon (arm64)" "uname -m | grep arm64"

RAM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
if [ "$RAM_GB" -ge 8 ]; then
  echo -e "${GREEN}[PASS]${NC} RAM: ${RAM_GB}GB"
  PASS=$((PASS+1))
else
  echo -e "${RED}[FAIL]${NC} RAM: ${RAM_GB}GB — need at least 8GB"
  FAIL=$((FAIL+1))
fi

DISK_GB=$(df -g ~ | awk 'NR==2{print $4}')
if [ "$DISK_GB" -ge 20 ]; then
  echo -e "${GREEN}[PASS]${NC} Free disk: ${DISK_GB}GB"
  PASS=$((PASS+1))
else
  echo -e "${RED}[FAIL]${NC} Free disk: ${DISK_GB}GB — need at least 20GB"
  FAIL=$((FAIL+1))
fi

echo ""
echo "=============================="
printf "  Results: ${GREEN}%d passed${NC}, ${RED}%d failed${NC}\n" "$PASS" "$FAIL"
echo "=============================="

if [ $FAIL -gt 0 ]; then
  echo ""
  echo -e "${YELLOW}Fix failures then re-run: make prereqs${NC}"
  echo ""
  echo "Missing tool install commands:"
  echo "  brew install kubectl helm k3d make"
  echo "  Docker Desktop: https://www.docker.com/products/docker-desktop/"
  exit 1
fi

echo ""
echo -e "${GREEN}All checks passed! Ready for Day 1.${NC}"
