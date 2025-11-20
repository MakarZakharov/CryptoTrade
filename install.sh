#!/bin/bash
###############################################################################
# Automated Installation Script for CryptoTrade DRL Environment
#
# This script automates the installation of all dependencies required for
# the cryptocurrency trading reinforcement learning environment.
#
# Usage:
#   bash install.sh [minimal|full|dev]
#
# Options:
#   minimal - Install only core dependencies (default)
#   full    - Install all dependencies including optional packages
#   dev     - Install full + development tools
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Installation mode (default: minimal)
MODE="${1:-minimal}"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   CryptoTrade DRL Environment - Installation Script        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}[1/6]${NC} Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION detected (>= 3.8 required)"
else
    echo -e "${RED}✗${NC} Python $PYTHON_VERSION is too old. Please upgrade to Python 3.8 or higher."
    exit 1
fi

# Check if virtual environment exists
echo ""
echo -e "${YELLOW}[2/6]${NC} Checking virtual environment..."
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠${NC}  Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created: venv/"
else
    echo -e "${GREEN}✓${NC} Virtual environment exists: venv/"
fi

# Activate virtual environment
echo ""
echo -e "${YELLOW}[3/6]${NC} Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Upgrade pip
echo ""
echo -e "${YELLOW}[4/6]${NC} Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo -e "${GREEN}✓${NC} pip upgraded to $(pip --version | awk '{print $2}')"

# Install dependencies based on mode
echo ""
echo -e "${YELLOW}[5/6]${NC} Installing dependencies (mode: ${BLUE}${MODE}${NC})..."

case "$MODE" in
    minimal)
        echo -e "  Installing minimal dependencies..."
        pip install -r requirements-minimal.txt
        ;;
    full)
        echo -e "  Installing full dependencies..."
        pip install -r requirements.txt
        ;;
    dev)
        echo -e "  Installing full + development dependencies..."
        pip install -r requirements.txt
        pip install black flake8 mypy jupyter ipywidgets notebook
        ;;
    *)
        echo -e "${RED}✗${NC} Unknown mode: $MODE"
        echo -e "  Valid options: minimal, full, dev"
        exit 1
        ;;
esac

echo -e "${GREEN}✓${NC} Dependencies installed successfully"

# Verify installation
echo ""
echo -e "${YELLOW}[6/6]${NC} Verifying installation..."

# Check critical imports
python3 << EOF
import sys
errors = []

try:
    import gymnasium
    print("  ✓ gymnasium")
except ImportError as e:
    errors.append(f"gymnasium: {e}")

try:
    import numpy
    print("  ✓ numpy")
except ImportError as e:
    errors.append(f"numpy: {e}")

try:
    import pandas
    print("  ✓ pandas")
except ImportError as e:
    errors.append(f"pandas: {e}")

try:
    import pyarrow
    print("  ✓ pyarrow")
except ImportError as e:
    errors.append(f"pyarrow: {e}")

try:
    import stable_baselines3
    print("  ✓ stable-baselines3")
except ImportError as e:
    errors.append(f"stable-baselines3: {e}")

try:
    import torch
    print("  ✓ torch")
except ImportError as e:
    errors.append(f"torch: {e}")

if errors:
    print("\n❌ Import errors detected:")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)
else:
    print("\n✅ All critical packages verified!")
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Installation verification successful"
else
    echo -e "${RED}✗${NC} Installation verification failed"
    exit 1
fi

# Display next steps
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                Installation Complete! 🎉                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo -e "  1. Activate the environment:"
echo -e "     ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo -e "  2. Test the environment:"
echo -e "     ${YELLOW}python DRL/Environment/examples/basic_usage.py${NC}"
echo ""
echo -e "  3. Run unit tests:"
echo -e "     ${YELLOW}pytest DRL/Environment/tests/${NC}"
echo ""
echo -e "  4. Start training an agent:"
echo -e "     ${YELLOW}python DRL/Environment/examples/train_sb3.py${NC}"
echo ""
echo -e "${BLUE}Documentation:${NC} See DRL/Environment/README.md"
echo -e "${BLUE}Examples:${NC} See DRL/Environment/examples/"
echo ""
