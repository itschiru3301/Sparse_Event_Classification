#!/bin/bash
################################################################################
# Sparse Autoencoder Setup Script
# 
# This script automates the setup of the sparse autoencoder project.
# It creates the necessary directory structure and installs dependencies.
################################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "  Sparse Autoencoder - Setup Script"
echo "================================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Step 1: Check Python version
echo "Step 1: Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_status "Python 3 found: version $PYTHON_VERSION"
echo ""

# Step 2: Create directory structure
echo "Step 2: Creating directory structure..."
mkdir -p src
mkdir -p models
mkdir -p images
mkdir -p logs
mkdir -p setup_docs
print_status "Directory structure created"
echo ""

# Step 3: Check if src module files exist
echo "Step 3: Verifying module files..."
REQUIRED_FILES=("src/__init__.py" "src/configs.py" "src/models.py" "src/datasets.py" "src/utils.py" "src/train.py" "src/finetune.py")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_status "$file exists"
    else
        MISSING_FILES+=("$file")
        print_error "$file missing"
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    print_error "Some module files are missing. Please ensure all files are in place."
    exit 1
fi
echo ""

# Step 4: Install Python dependencies
echo "Step 4: Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    print_info "Installing packages from requirements.txt..."
    python3 -m pip install --upgrade pip --quiet
    python3 -m pip install -r requirements.txt --quiet
    print_status "Dependencies installed successfully"
else
    print_error "requirements.txt not found"
    exit 1
fi
echo ""

# Step 5: Verify CUDA (if available)
echo "Step 5: Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA is available')
    print(f'  Device: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA Version: {torch.version.cuda}')
else:
    print('ℹ CUDA is not available (CPU-only mode)')
" || print_info "PyTorch not yet fully installed"
echo ""

# Step 6: Verify imports
echo "Step 6: Verifying module imports..."
python3 << 'EOF'
import sys
try:
    from src import (
        configs,
        SparseAutoencoder,
        SparseClassifier,
        UnlabelledJetDataset,
        LabelledJetDataset,
        dense_to_sparse,
        apply_pruning,
    )
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_status "All modules imported successfully"
else
    print_error "Module import failed"
    exit 1
fi
echo ""

# Step 7: Summary
echo "================================================================================"
echo "  Setup Complete!"
echo "================================================================================"
echo ""
print_info "Next steps:"
echo "  1. Review configuration in src/configs.py"
echo "  2. Update data paths if necessary"
echo "  3. Run full training pipeline:"
echo "     python3 -m src.train"
echo "  4. Or run fine-tuning only:"
echo "     python3 -m src.finetune"
echo "  5. View detailed setup guide:"
echo "     cat setup_docs/SETUP.md"
echo ""
print_status "Setup successful!"
echo ""
