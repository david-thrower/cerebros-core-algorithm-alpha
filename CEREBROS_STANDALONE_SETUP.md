# Cerebros Multi-Stage Training System

Complete standalone system for training personalized AI assistants through a 5-stage pipeline.

## Architecture

This system consists of three components:

1. **Python Training Backend** - Multi-stage training pipeline
2. **Flask API Server** - REST API for training management
3. **React UI** - User interface for the wizard

## Setup Instructions

### 1. Quick Start (Recommended)

Use the provided startup script that handles virtual environment setup:

```bash
cd /home/mo/thunderline
./start_cerebros.sh
```

This will:
- Create a Python virtual environment (if needed)
- Install all dependencies
- Start the API server on `http://localhost:5000`

### 2. Manual Setup (Alternative)

If you prefer to set up manually:

```bash
cd /home/mo/thunderline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install flask flask-cors pandas numpy

# Start API server
python cerebros-core-algorithm-alpha/training_api_server.py
```

Server will start on `http://localhost:5000`

### 3. Start the React UI

```bash
cd /home/mo/thunderline/cerebros-core-algorithm-alpha/UI\ REFERENCE
npm install
npm run dev
```

UI will start on `http://localhost:5173`

## 5-Stage Training Pipeline

### Stage 1: Initial Foundation
- Base model training with foundational patterns
- Outputs: `stage_1_checkpoint.keras`

### Stage 2: Domain Adaptation  
- Merges relevant and general domain data
- Loads Stage 1 checkpoint
- Outputs: `stage_2_checkpoint.keras`

### Stage 3: Knowledge Integration
- Merges relevant data, general data, and reference knowledge base
- Loads Stage 2 checkpoint  
- Outputs: `stage_3_checkpoint.keras`

### Stage 4: Style Refinement
- Merges relevant and general style data
- Loads Stage 3 checkpoint
- Outputs: `stage_4_checkpoint.keras`

### Stage 5: Personalization Fine-Tuning
- Merges work products, prompts, and communications
- Loads Stage 4 checkpoint
- Outputs: Final model `stage_5_checkpoint.keras`

## API Endpoints

### Health Check
```bash
GET /health
```

### Create Agent
```bash
POST /api/agents
Body: {"name": "My Assistant"}
```

### Upload Documents
```bash
POST /api/agents/{agent_id}/documents
Form Data: files[], type=work_product|communication|reference
```

### Start Training
```bash
POST /api/agents/{agent_id}/train
Body: {"agent_name": "My Assistant"}
```

### Get Training Status
```bash
GET /api/agents/{agent_id}/status
```

### Get Training Results
```bash
GET /api/agents/{agent_id}/results
```

### Deploy Agent
```bash
POST /api/agents/{agent_id}/deploy
```

### Chat with Agent
```bash
POST /api/agents/{agent_id}/chat
Body: {"message": "Hello"}
```

## File Structure

```
cerebros-core-algorithm-alpha/
├── multi_stage_trainer.py      # 5-stage training pipeline
├── training_api_server.py      # Flask API server
└── UI REFERENCE/
    └── src/
        └── components/
            └── MultiStageWizard.tsx  # React wizard UI

priv/nfs/agents/
└── {agent_id}/
    ├── agent_metadata.json
    ├── model_metadata.json
    ├── work_product/
    ├── communication/
    ├── reference/
    ├── processed/
    └── checkpoints/
        ├── stage_1_checkpoint.keras
        ├── stage_2_checkpoint.keras
        ├── stage_3_checkpoint.keras
        ├── stage_4_checkpoint.keras
        ├── stage_5_checkpoint.keras
        └── stage_X_metadata.json
```

## Testing the System

### 1. Test API Server

```bash
# Health check
curl http://localhost:5000/health

# Create agent
curl -X POST http://localhost:5000/api/agents \
  -H "Content-Type: application/json" \
  -d '{"name":"Test Assistant"}'
```

### 2. Test Training Pipeline Directly

```bash
# Run training for an agent
python3 cerebros-core-algorithm-alpha/multi_stage_trainer.py \
  "test-agent-123" \
  "Test Assistant"
```

### 3. Test UI

1. Open browser to `http://localhost:5173`
2. Follow the 5-step wizard
3. Upload files in each step
4. Click "Start Training" in Step 5
5. Watch real-time progress through all 5 stages

## Training Progress Indicators

The UI shows real-time progress for each stage:

- ⚪ Pending - Stage not yet started
- 🔵 Training - Stage currently executing  
- ✅ Complete - Stage finished successfully
- ❌ Error - Stage failed

Each stage shows:
- Progress percentage (0-100%)
- Training metrics (loss, accuracy, perplexity)
- Status indicators

## Advanced Settings (Post-Deployment)

After training completes, the deployed assistant supports:

- **top_p**: 0.6 to 0.995
- **repetition_penalty**: 1 to 1.5
- **presence_penalty**: 1 to 1.5
- **frequency_penalty**: 1 to 1.5

These settings can be adjusted in the chat interface.

## Development Notes

- The training pipeline currently uses simulation for demo purposes
- Replace `simulate_training()` with actual Cerebros NAS calls for production
- MLflow integration for metrics tracking (TODO)
- Model deployment automation (TODO)
- Database table creation for conversations (TODO)

## Troubleshooting

### Port already in use
```bash
# Kill existing Flask server
pkill -f training_api_server.py

# Or change port in training_api_server.py
app.run(host='0.0.0.0', port=5001, debug=True)
```

### React build errors
```bash
cd cerebros-core-algorithm-alpha/UI\ REFERENCE
rm -rf node_modules
npm install
```

### Training fails
```bash
# Check logs
tail -f /tmp/training.log

# Check agent directory
ls -la priv/nfs/agents/{agent_id}/
```

## Next Steps

1. ✅ Multi-stage training pipeline
2. ✅ Flask API server
3. ✅ React wizard UI
4. ⏳ Integrate actual Cerebros NAS
5. ⏳ MLflow metrics tracking
6. ⏳ Model deployment automation
7. ⏳ Chat interface with advanced settings
8. ⏳ Database setup for conversations

## License

MIT
