# CEREBROS NotGPT - Demo Runbook

**Version:** 1.0  
**Last Updated:** 2025-10-31  
**Estimated Time:** 15 minutes for complete setup and demo

---

## 📋 Prerequisites

### System Requirements
- **OS:** Linux (Ubuntu 20.04+ recommended)
- **Python:** ≥ 3.10
- **GPU:** NVIDIA GPU with CUDA support (optional, falls back to CPU)
- **RAM:** 8GB minimum, 16GB+ recommended
- **Disk:** 10GB free space

### Required Software
```bash
# Python packages
pip install torch transformers llama-cpp-python mlflow fastapi uvicorn pandas numpy

# Node.js (for UI)
node --version  # Should be ≥ 16.x
npm --version   # Should be ≥ 8.x
```

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Clone and Setup
```bash
git clone <repository-url> cerebros-core-algorithm-alpha
cd cerebros-core-algorithm-alpha

# Run setup script
chmod +x start_cerebros.sh
./start_cerebros.sh
```

**Expected Output:**
```
🚀 CEREBROS NotGPT MVP - Environment Setup
==========================================
[INFO] ✓ Python 3.13 detected
[INFO] ✓ NVIDIA GPU drivers detected
[INFO] ✓ All required packages found!
[INFO] ✓ Directory structure created
[INFO] ✓ MLflow server started (PID: xxxxx)
[INFO] ✅ Setup complete!
```

### Step 2: Load Environment
```bash
source .env
```

This sets:
- `MLFLOW_TRACKING_URI` - MLflow tracking location
- `CEREBROS_NFS_PATH` - Data storage path
- `PYTHONPATH` - Project root

### Step 3: Process Sample Data
```bash
python3 scripts/process_user_samples.py --assistant_id demo
```

**Expected Output:**
```
11:11:35 [Stage 1] Processing Stage 1: Work Products
11:11:35 [Stage 1] ✓ Stage 1 complete: 15 samples
11:11:35 [Stage 2] Processing Stage 2: Qa Examples
11:11:35 [Stage 2] ✓ Stage 2 complete: 15 samples
11:11:35 [Stage 3] Processing Stage 3: Threads
11:11:35 [Stage 3] ✓ Stage 3 complete: 15 samples
11:11:35 [Stage 4] Processing Stage 4: Reference Docs
11:11:35 [Stage 4] ✓ Stage 4 complete: 15 samples
🎉 Data processing pipeline completed successfully!
📊 Total samples generated: 60
```

**Generated Files:**
```
priv/nfs/demo/datasets/
├── training_stage1.csv  # Work products
├── training_stage2.csv  # Q&A examples
├── training_stage3.csv  # Communication threads
└── training_stage4.csv  # Reference docs
```

### Step 4: Train the Model
```bash
python3 multi_stage_trainer.py demo "Demo Assistant" priv/nfs
```

**Expected Output:**
```
2025-10-31 11:11:49 [Stage 1] Starting Stage 1: Initial Foundation Training
2025-10-31 11:11:50 [Stage 1] ✓ Stage 1 complete!
2025-10-31 11:11:52 [Stage 2] ✓ Stage 2 complete!
2025-10-31 11:11:54 [Stage 3] ✓ Stage 3 complete!
2025-10-31 11:11:55 [Stage 4] ✓ Stage 4 complete!
2025-10-31 11:11:58 [Stage 5] ✓ Stage 5 complete! Model ready for deployment!
🎉 All 5 stages completed successfully!
```

**Generated Files:**
```
priv/nfs/agents/demo/checkpoints/
├── stage_1_checkpoint.keras
├── stage_2_checkpoint.keras
├── stage_3_checkpoint.keras
├── stage_4_checkpoint.keras
├── stage_5_checkpoint.keras
└── model_metadata.json
```

### Step 5: Start API Server
```bash
CEREBROS_API_PORT=8080 python3 server/app.py
```

**Expected Output:**
```
============================================================
🚀 CEREBROS NotGPT API Server
============================================================
📡 Starting server on http://0.0.0.0:8080
📖 API docs at http://0.0.0.0:8080/docs
============================================================
INFO:     Uvicorn running on http://0.0.0.0:8080
```

Keep this terminal open. The server is now running!

---

## 🧪 Testing the API

### Open a New Terminal
```bash
cd cerebros-core-algorithm-alpha
source .env
```

### Test 1: Health Check
```bash
curl http://localhost:8080/
```

**Expected Response:**
```json
{
  "service": "CEREBROS NotGPT API",
  "version": "1.0.0",
  "status": "running"
}
```

### Test 2: List Assistants
```bash
curl http://localhost:8080/assistants
```

**Expected Response:**
```json
{
  "assistants": [
    {
      "assistant_id": "demo",
      "name": "Demo Assistant",
      "status": "ready",
      "deployment_ready": true
    }
  ],
  "count": 1
}
```

### Test 3: Query Assistant
```bash
curl -X POST http://localhost:8080/assistants/demo/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I reset my password?",
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

**Expected Response:**
```json
{
  "response": "Based on my training, here's what I understand...",
  "assistant_id": "demo",
  "timestamp": "2025-10-31T11:15:30.123456",
  "metadata": {
    "query_length": 28,
    "response_length": 156
  }
}
```

### Test 4: Get Assistant Status
```bash
curl http://localhost:8080/assistants/demo/status
```

**Expected Response:**
```json
{
  "assistant_id": "demo",
  "name": "Demo Assistant",
  "status": "ready",
  "model_path": "priv/nfs/agents/demo/checkpoints/stage_5_checkpoint.keras",
  "created_at": "2025-10-31T11:11:58.123456"
}
```

---

## 🎨 UI Demo (Optional - Requires Frontend Squad Completion)

### Step 1: Install UI Dependencies
```bash
cd web_demo
npm install
```

### Step 2: Start UI Development Server
```bash
npm run dev
```

**Expected Output:**
```
  VITE v5.2.0  ready in 234 ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: http://192.168.1.x:3000/
```

### Step 3: Open Browser
Navigate to `http://localhost:3000`

**Expected Pages:**
- **Dashboard (/)** - List of assistants and their status
- **Upload Wizard (/new)** - Upload files for new assistant
- **Chat Interface (/assistants/demo)** - Chat with demo assistant

---

## 📊 Monitoring with MLflow

### Access MLflow UI
```bash
# MLflow server should already be running from start_cerebros.sh
# If not, start it manually:
mlflow server --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```

Open browser: `http://localhost:5000`

**What You'll See:**
- **Experiments:** `cerebros_data_processing`
- **Runs:** Individual processing runs with metrics
- **Metrics:**
  - `total_samples_generated`
  - `stage_1_samples`, `stage_2_samples`, etc.
  - `processing_time_seconds`
  - `llm_available`

---

## 🔍 Output Verification

After completing the steps above, validate that all outputs match the expected results defined in [`docs/QA_VALIDATION_REPORT.md`](QA_VALIDATION_REPORT.md).

### Steps for Verification
| Checkpoint | Command | Expected Output |
|-------------|----------|----------------|
| Data Processing | `python3 scripts/process_user_samples.py --assistant_id demo` | 60 processed samples saved under `priv/nfs/demo/datasets/` |
| Model Training | `python3 multi_stage_trainer.py demo "Demo Assistant" priv/nfs` | 5 successful checkpoints under `priv/nfs/agents/demo/checkpoints/` |
| FastAPI | `curl http://localhost:8080/assistants` | JSON object listing `Demo Assistant` with `status: ready` |
| Frontend | open `http://localhost:3000` | Dashboard, Upload Wizard, and Chat interfaces visible |
| MLflow | open `http://localhost:5000` | Experiments containing recent demo runs |

Results should correspond with the QA results section **“Demo Validation — Expected Outputs”** in [`QA_VALIDATION_REPORT.md`](QA_VALIDATION_REPORT.md).

---

### Check Directory Structure
```bash
tree -L 3 priv/nfs/
```

**Expected Output:**
```
priv/nfs/
├── agents
│   └── demo
│       ├── checkpoints
│       └── model_metadata.json
├── database
│   ├── assistants.sql
│   ├── training_jobs.sql
│   └── training_samples.sql
├── datasets
├── demo
│   ├── datasets
│   ├── logs
│   └── processed
└── uploads
```

### Check Generated Files
```bash
ls -lh priv/nfs/demo/datasets/
ls -lh priv/nfs/agents/demo/checkpoints/
```

### Check MLflow Logs
```bash
ls -lh mlruns/
cat mlflow.log
```

---

## 🐛 Troubleshooting

### Issue: "Permission denied: '/mlruns'"
**Solution:** Ensure you've sourced the `.env` file:
```bash
source .env
echo $MLFLOW_TRACKING_URI  # Should show: file:///path/to/mlruns
```

### Issue: "Port 8080 already in use"
**Solution:** Use a different port:
```bash
CEREBROS_API_PORT=8081 python3 server/app.py
```

### Issue: "LLM initialization failed"
**Solution:** This is expected for demo. The system falls back to mock data generation which works fine for testing.

### Issue: "Module not found"
**Solution:** Ensure PYTHONPATH is set:
```bash
export PYTHONPATH=$(pwd)
```

### Issue: "CUDA out of memory"
**Solution:** Reduce GPU layers in `process_gutenberg_local.py`:
```python
GPU_LAYERS = 10  # Reduce from 25
```

---

## 📁 File Locations Reference

### Configuration Files
- `.env` - Environment variables
- `requirements.txt` - Python dependencies
- `start_cerebros.sh` - Setup script

### Core Scripts
- `scripts/process_user_samples.py` - Data ingestion (4 stages)
- `multi_stage_trainer.py` - Model training (5 stages)
- `server/app.py` - FastAPI inference server

### Data Directories
- `priv/nfs/{assistant_id}/datasets/` - Training CSV files
- `priv/nfs/agents/{assistant_id}/checkpoints/` - Model checkpoints
- `priv/nfs/uploads/` - User uploaded files
- `mlruns/` - MLflow experiment tracking

### UI (when complete)
- `web_demo/` - React frontend application
- `web_demo/src/` - React components and pages

---

## 🎬 Complete Demo Script

Run this for a full end-to-end demo:

```bash
#!/bin/bash
# CEREBROS NotGPT - Complete Demo Script

echo "🚀 CEREBROS NotGPT Demo Starting..."
echo

# Step 1: Setup
echo "Step 1: Environment Setup"
./start_cerebros.sh
source .env
sleep 2

# Step 2: Process Data
echo
echo "Step 2: Processing User Data"
python3 scripts/process_user_samples.py --assistant_id demo
sleep 2

# Step 3: Train Model
echo
echo "Step 3: Training Multi-Stage Model"
python3 multi_stage_trainer.py demo "Demo Assistant" priv/nfs
sleep 2

# Step 4: Start API
echo
echo "Step 4: Starting API Server"
CEREBROS_API_PORT=8080 python3 server/app.py &
API_PID=$!
sleep 5

# Step 5: Test API
echo
echo "Step 5: Testing API Endpoints"
echo
echo "5a. Health Check:"
curl -s http://localhost:8080/ | jq .
echo
echo
echo "5b. List Assistants:"
curl -s http://localhost:8080/assistants | jq .
echo
echo
echo "5c. Query Assistant:"
curl -s -X POST http://localhost:8080/assistants/demo/query \
  -H "Content-Type: application/json" \
  -d '{"query":"How do I reset my password?"}' | jq .
echo
echo

# Step 6: Summary
echo "✅ Demo Complete!"
echo
echo "🌐 API Server: http://localhost:8080/docs"
echo "📊 MLflow: http://localhost:5000"
echo "📁 Data: priv/nfs/"
echo
echo "To stop API server: kill $API_PID"
```

Save as `demo.sh`, make executable, and run:
```bash
chmod +x demo.sh
./demo.sh
```

---

## 📞 Support & Next Steps

### Getting Help
- Check `docs/TEAM_COORDINATION.md` for squad assignments
- Review API documentation at `http://localhost:8080/docs`
- Check MLflow for metrics at `http://localhost:5000`

### Next Steps for Development
1. **Frontend Squad:** Integrate UI from UIREFERENCE
2. **Backend Squad:** Add real LLM model loading
3. **QA Squad:** Write comprehensive tests
4. **Docs Squad:** Add screenshots and video demo

### Production Deployment
- Add authentication/authorization
- Configure proper CORS origins
- Set up reverse proxy (nginx)
- Add monitoring and logging
- Containerize with Docker
- Set up CI/CD pipeline

---

**🎉 Congratulations! You've successfully set up and tested CEREBROS NotGPT MVP.**