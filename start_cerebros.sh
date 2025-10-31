#!/bin/bash

# CEREBROS "NotGPT" MVP - Environment Setup Script
# Initializes NFS, Postgres, and MLflow directories and validates environment

set -e  # Exit on any error

echo "🚀 CEREBROS NotGPT MVP - Environment Setup"
echo "=========================================="

# Configuration
PROJECT_ROOT="$(pwd)"
NFS_ROOT="/mnt/data/cerebros"
PRIV_NFS="$PROJECT_ROOT/priv/nfs"
MLRUNS_DIR="$PROJECT_ROOT/mlruns"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Phase 1: Check Dependencies
echo
log_info "Phase 1: Checking Dependencies"
echo "--------------------------------"

# Check Python version
if command_exists python3; then
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [ "$(echo "$PYTHON_VERSION >= 3.10" | bc -l)" -eq 1 ] 2>/dev/null || python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
        log_info "✓ Python $PYTHON_VERSION detected (>= 3.10 required)"
    else
        log_error "✗ Python >= 3.10 required, found $PYTHON_VERSION"
        exit 1
    fi
else
    log_error "✗ Python 3 not found"
    exit 1
fi

# Check CUDA
if command_exists nvidia-smi; then
    log_info "✓ NVIDIA GPU drivers detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
else
    log_warn "⚠ NVIDIA GPU drivers not detected (CPU-only mode)"
fi

# Check key Python packages
log_info "Checking Python packages..."
python3 -c "
import sys
required_packages = ['torch', 'transformers', 'llama_cpp', 'mlflow', 'fastapi', 'pandas', 'numpy']
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f'  ✓ {pkg}')
    except ImportError:
        print(f'  ✗ {pkg}')
        missing.append(pkg)
        
if missing:
    print(f'\\nMissing packages: {missing}')
    print('Run: pip install -r requirements.txt')
    sys.exit(1)
else:
    print('  All required packages found!')
"

# Phase 2: Create Directory Structure
echo
log_info "Phase 2: Creating Directory Structure"
echo "------------------------------------"

# Create NFS-style directory structure
log_info "Creating NFS directories..."
mkdir -p "$PRIV_NFS"/{demo,agents,uploads,datasets}
mkdir -p "$PRIV_NFS/demo"/{checkpoints,datasets,logs,processed}
mkdir -p "$PRIV_NFS/agents"
mkdir -p "$PRIV_NFS/uploads"
mkdir -p "$PRIV_NFS/datasets"

# Create MLflow directory
mkdir -p "$MLRUNS_DIR"

# Create server directory
mkdir -p "$PROJECT_ROOT/server"

# Create scripts directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/scripts"

# Create docs directory
mkdir -p "$PROJECT_ROOT/docs"

log_info "✓ Directory structure created:"
tree "$PRIV_NFS" 2>/dev/null || find "$PRIV_NFS" -type d | sed 's/^/  /'

# Phase 3: Initialize MLflow
echo
log_info "Phase 3: Initializing MLflow"
echo "-----------------------------"

export MLFLOW_TRACKING_URI="file://$MLRUNS_DIR"
log_info "MLflow tracking URI: $MLFLOW_TRACKING_URI"

# Start MLflow server in background if not running
if ! pgrep -f "mlflow server" > /dev/null; then
    log_info "Starting MLflow tracking server..."
    mlflow server --backend-store-uri "$MLRUNS_DIR" --default-artifact-root "$MLRUNS_DIR" --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &
    MLFLOW_PID=$!
    echo $MLFLOW_PID > mlflow.pid
    sleep 3
    if ps -p $MLFLOW_PID > /dev/null; then
        log_info "✓ MLflow server started (PID: $MLFLOW_PID)"
        log_info "  Access at: http://localhost:5000"
    else
        log_warn "⚠ MLflow server may have failed to start (check mlflow.log)"
    fi
else
    log_info "✓ MLflow server already running"
fi

# Phase 4: Database Setup (Mock Postgres tables)
echo
log_info "Phase 4: Database Setup (Mock)"
echo "------------------------------"

# Create mock database files to simulate Postgres tables
DB_DIR="$PRIV_NFS/database"
mkdir -p "$DB_DIR"

# Create mock table files
cat > "$DB_DIR/assistants.sql" << 'EOF'
-- Mock Postgres table: assistants
CREATE TABLE assistants (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'training',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
EOF

cat > "$DB_DIR/training_samples.sql" << 'EOF'
-- Mock Postgres table: training_samples
CREATE TABLE training_samples (
    id UUID PRIMARY KEY,
    assistant_id UUID REFERENCES assistants(id),
    content TEXT NOT NULL,
    sample_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
EOF

cat > "$DB_DIR/training_jobs.sql" << 'EOF'
-- Mock Postgres table: training_jobs
CREATE TABLE training_jobs (
    id UUID PRIMARY KEY,
    assistant_id UUID REFERENCES assistants(id),
    status VARCHAR(50) DEFAULT 'pending',
    stage INTEGER DEFAULT 1,
    metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
EOF

log_info "✓ Mock database tables created in $DB_DIR"

# Phase 5: GPU Model Validation
echo
log_info "Phase 5: GPU Model Validation"
echo "-----------------------------"

log_info "Testing GPU model loading (this may take a few minutes)..."

# Check if the model is available
MODEL_PATH="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF/blobs/4b78837bbec5ee248e4a5642bf608b6793721af41b92589e40c8da0bce58b907"

if [ ! -f "$MODEL_PATH" ]; then
    log_warn "⚠ Qwen 3 30B model not found at expected location"
    log_info "  Attempting to run process_gutenberg_local.py (will download if needed)..."
fi

# Test model loading
if python3 process_gutenberg_local.py --min_index 0 --max_index 1 2>&1 | grep -q "Qwen 3 30B loaded"; then
    log_info "✓ Qwen 3 30B loaded successfully"
else
    log_error "✗ Failed to load Qwen 3 30B model"
    log_info "  This may be due to insufficient GPU memory or missing model files"
    log_info "  The system will continue with CPU-only mode"
fi

# Phase 6: Environment Summary
echo
log_info "Phase 6: Environment Summary"
echo "----------------------------"

cat << EOF
🎉 CEREBROS Environment Setup Complete!

Directories Created:
  📁 $PRIV_NFS/
    ├── demo/ (for demo assistant)
    ├── agents/ (for user assistants)
    ├── uploads/ (for user data)
    ├── datasets/ (for training data)
    └── database/ (mock Postgres tables)
  
  📁 $MLRUNS_DIR/ (MLflow tracking)
  📁 $PROJECT_ROOT/server/ (FastAPI app)
  📁 $PROJECT_ROOT/scripts/ (processing scripts)

Services:
  🔬 MLflow: http://localhost:5000
  
Next Steps:
  1. Run: python3 scripts/process_user_samples.py
  2. Run: python3 scripts/multi_stage_trainer.py demo demo
  3. Run: python3 server/app.py
  4. Open web_demo/ in browser

Environment Variables:
  export MLFLOW_TRACKING_URI="file://$MLRUNS_DIR"
  export CEREBROS_NFS_PATH="$PRIV_NFS"

EOF

# Export environment variables for current session
export MLFLOW_TRACKING_URI="file://$MLRUNS_DIR"
export CEREBROS_NFS_PATH="$PRIV_NFS"

# Create .env file for persistence
cat > "$PROJECT_ROOT/.env" << EOF
MLFLOW_TRACKING_URI=file://$MLRUNS_DIR
CEREBROS_NFS_PATH=$PRIV_NFS
PYTHONPATH=$PROJECT_ROOT
EOF

log_info "✅ Setup complete! Environment ready for CEREBROS MVP development."

echo
echo "🚀 You can now proceed with the development phases!"
echo "================================================="