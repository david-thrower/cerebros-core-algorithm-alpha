# Cerebros NotGPT — AI Colleague Dashboard

Standalone dashboard for the 7-step AI Colleague onboarding workflow. Upload data, run preprocessing pipelines, review synthetic samples, trigger training.

## Quick Start (No Redis, No GPU)

```bash
pip install reflex pdfplumber python-docx sqlalchemy pendulum aiofiles huey redis
cd cerebros-core-algorithm-alpha
reflex run --env dev
# → http://localhost:3000
```

This runs the dashboard with deterministic generators (no LLM) and inline task execution (no Redis). Good for UI development and demo.

## Full Stack (Redis + Worker Pods)

### 1. Start Redis

```bash
# Docker (recommended)
docker run -d --name cerebros-redis -p 6379:6379 redis:7-alpine

# Or install locally
# Fedora: sudo dnf install redis && sudo systemctl start redis
# Mac: brew install redis && brew services start redis
```

### 2. Start the Huey Worker

```bash
cd cerebros-core-algorithm-alpha

# 2 worker threads, process-based (for CPU tasks)
huey_consumer notgpt.orchestration.queue.huey -w 2 -k process

# For GPU tasks, run a separate worker with GPU access:
# huey_consumer notgpt.orchestration.queue.huey -w 1 -k process
```

### 3. Start the Dashboard

```bash
reflex run --env dev
# → http://localhost:3000
```

### 4. Test the Pipeline (No UI)

```bash
# Run all tests (immediate mode, no Redis needed)
HUEY_IMMEDIATE=true python -m pytest notgpt/tests/ -v

# Dispatch a task via Python
python -c "
from notgpt.orchestration.client import dispatch, get_result
task_id = dispatch('preprocess_work_products', colleague_id=1, num_samples=3)
print(f'Task dispatched: {task_id}')
"
```

## Environment Variables

### Redis / Queue
| Variable | Default | Description |
|---|---|---|
| `REDIS_HOST` | `localhost` | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_DB` | `0` | Redis database number |
| `HUEY_IMMEDIATE` | `false` | Run tasks inline (no Redis, for testing) |

### LLM / Generators
| Variable | Default | Description |
|---|---|---|
| `USE_LLM_GENERATORS` | `true` | Use real Qwen LLM vs deterministic fallback |
| `QWEN_MODEL` | `Qwen/Qwen3.5-0.8B` | HuggingFace model ID |
| `MAX_SEQ_LEN` | `500` | Max tokens per generation |
| `TARGET_SEQUENCE_LEN` | `500` | Target length for prompt engineering |
| `GEN_TEMPERATURE` | `0.7` | Generation temperature |
| `GEN_TOP_P` | `0.95` | Top-p sampling |
| `GEN_TOP_K` | `50` | Top-k sampling |

### MLflow (Optional)
| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | MLflow server |
| `MLFLOW_S3_ENDPOINT_URL` | (unset) | SeaweedFS/S3 endpoint |

### Storage
| Variable | Default | Description |
|---|---|---|
| `NOTGPT_DB_PATH` | `notgpt/data/notgpt.db` | SQLite database path |

## Architecture

```
Dashboard (Reflex)
    ↓ dispatch()
Redis (Huey Queue)
    ↓ KEDA monitors queue length
Worker Pods (huey_consumer)
    ├── preprocess_work_products    (CPU)
    ├── preprocess_qa               (CPU)
    ├── preprocess_comm_threads     (CPU)
    ├── preprocess_references       (CPU)
    ├── train_stage_2               (GPU)
    ├── train_stage_3               (GPU)
    ├── train_stage_4               (GPU)
    └── run_full_pipeline           (chains all above)
```

## Kubernetes / KEDA

The worker scales to zero when idle. KEDA watches Redis queue length.

```yaml
# keda-scaledobject.yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: cerebros-worker
spec:
  scaleTargetRef:
    name: cerebros-worker
  minReplicaCount: 0
  maxReplicaCount: 10
  triggers:
    - type: redis
      metadata:
        address: redis:6379
        listName: "cerebros_pipeline"
        listLength: "1"
```

## File Structure

```
notgpt/
  app.py                          # NiceGUI prototype (legacy)
  storage/
    models.py                     # SQLAlchemy: Colleague, Document, QA, Sample
    db.py                         # Engine + session management
  pipeline/
    generators.py                 # LLM + tokenizer-based fallback
    text_extract.py               # PDF/DOCX/TXT extraction
    work_products.py              # Pipeline #1
    qa_upsampling.py              # Pipeline #2
    comm_threads.py               # Pipeline #3
    references.py                 # Pipeline #4
  orchestration/
    queue.py                      # RedisHuey config
    tasks.py                      # 8 Huey task definitions
    client.py                     # RPC dispatch + result polling
  tests/                          # 17 tests passing

cerebros_dashboard/
  cerebros_dashboard.py           # Reflex app (splash + 7-step wizard)
  state.py                        # WizardState + AppState

assets/
  cerebros-logo.png               # Brain logo
  style.css                       # Brand tokens (cyan→pink gradient)

rxconfig.py                       # Reflex + Tailwind config
```

## Tests

```bash
# All 17 tests, no Redis needed
HUEY_IMMEDIATE=true python -m pytest notgpt/tests/ -v
```
