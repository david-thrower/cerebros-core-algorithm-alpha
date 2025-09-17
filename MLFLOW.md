# Cerebros PoC MLflow Tracking

This document explains how to run the generative PoC with MLflow tracking locally (no Automat0 integration yet).

## Overview
We wrapped Optuna trials with MLflow runs:
- Parent run started in `main()` (one per invocation)
- Nested run per Optuna trial (trial_* numbering)
- Logged artifacts (future: model or oracle CSV) TBD
- Metrics:
  - objective: value returned by `objective()`
  - trial_duration_seconds
- Params:
  - Sampled hyperparameters (with `_sampled` suffix)
  - Best trial params (prefixed with `best_` in parent run)
  - n_trials executed

Environment toggles:
- `CEREBROS_FAST=1` reduces `n_trials` default to 3 (vs 20)
- `CEREBROS_N_TRIALS` overrides trial count
- `MLFLOW_EXPERIMENT_NAME` sets experiment (default `cerebros_poc`)
- `MLFLOW_PARENT_RUN_NAME` sets the parent run name
- `MLFLOW_TRACKING_URI` sets tracking backend (default local ./mlruns if unset)

## Prerequisites
Python 3.10+ suggested.

Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start (Local File Store)
If `MLFLOW_TRACKING_URI` is not set, MLflow will create `./mlruns` directory.

Run fast smoke:
```bash
export CEREBROS_FAST=1
python generative-proof-of-concept-CPU-preprocessing-in-memory.py
```

Run full (20 trials):
```bash
unset CEREBROS_FAST
python generative-proof-of-concept-CPU-preprocessing-in-memory.py
```

Override number of trials:
```bash
export CEREBROS_N_TRIALS=5
python generative-proof-of-concept-CPU-preprocessing-in-memory.py
```

Set experiment & remote tracking (example using local backend + artifact dir):
```bash
export MLFLOW_TRACKING_URI="file:$(pwd)/mlruns"  # explicit
export MLFLOW_EXPERIMENT_NAME=cerebros_poc_experiment
python generative-proof-of-concept-CPU-preprocessing-in-memory.py
```

## Launch MLflow UI
```bash
mlflow ui --backend-store-uri "file:$(pwd)/mlruns" --host 0.0.0.0 --port 5000
```
Visit: http://localhost:5000

## Expected Run Structure
Parent Run (cerebros_poc_parent)
- Params: n_trials, best_* params
- Metrics: best_value
Nested Trial Runs (trial_<n>)
- Params: sampled hyperparameters
- Metrics: objective, trial_duration_seconds

## Future Enhancements (TODO)
- Log generated model artifact (.keras) when enabled
- Log oracle / results CSV as an artifact
- Stream intermediate epoch metrics if training loop exposed
- Attach system metrics (CPU/RAM) via MLflow system metrics plugin or custom logging
- Integrate with Automat0 event bus once bridge implemented

## Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| No runs appear | Different working directory | Run from repo root or check mlruns path |
| Permission error | Artifact path unwritable | Set `MLFLOW_TRACKING_URI` to writable location |
| Optuna fails import | Not installed | Re-run `pip install -r requirements.txt` |
| GPU OOM / slow | Large model or batch size | Use `CEREBROS_FAST=1` or reduce trials |

## Clean Up
Remove local runs:
```bash
rm -rf mlruns
```

## Minimal Programmatic Access Example
```python
import mlflow
for run in mlflow.search_runs(experiment_names=["cerebros_poc"], filter_string="tags.phase = 'poc'"):
    print(run.info.run_id, run.data.metrics.get("objective"))
```

---
Maintained as part of Q4 2025 Steel Thread (Single Worker + MLflow).

## R&D Localhost Minimal MLflow (Isolation from MVP)
For pure R&D (fast iteration, disposable runs) keep a totally local, isolated MLflow instance. This prevents cluttering any shared or future MVP tracking backends and keeps experiment semantics loose.

### Why Separate?
- R&D = high-churn, exploratory, many failed/partial runs.
- MVP / Product = curated, reproducible, governance and retention rules.
- Separation avoids polluting metrics lineage and lets you freely mutate schema/params.

### Minimal Pattern
No DB, no S3: just the file store created automatically by mlflow.

```
project-root/
  cerebros-worker-branch/
    mlruns/            # auto-created (gitignored if desired)
    generative-proof-of-concept-CPU-preprocessing-in-memory.py
    run_local_mlflow.sh
```

### One-Time Script (created separately as run_local_mlflow.sh)
```
#!/usr/bin/env bash
set -euo pipefail
export MLFLOW_TRACKING_URI="file:$(pwd)/mlruns"
echo "MLflow tracking dir: $MLFLOW_TRACKING_URI"
echo "Starting MLflow UI on http://127.0.0.1:5000"
mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI" --host 127.0.0.1 --port 5000
```
Make executable:
```bash
chmod +x run_local_mlflow.sh
```

In another terminal run the PoC (fast mode):
```bash
export MLFLOW_TRACKING_URI="file:$(pwd)/mlruns"  # redundant but explicit
export CEREBROS_FAST=1
python generative-proof-of-concept-CPU-preprocessing-in-memory.py
```

### Clean Slate R&D Reset
```bash
rm -rf mlruns
```

### Git Hygiene
Optionally add to a local-only .gitignore snippet (if not already ignored):
```
mlruns/
```

### Promotion Path
If an R&D run matters:
1. Capture its best params from the parent run.
2. Re-run in a controlled (future MVP) environment with pinned seed & code hash.
3. Tag the promoted run (e.g. tag: `source_run_id=<original_run_id>`).

### Future: Dual Wiring
Later we can add an ENV gate, e.g.:
```
if [ "$MLFLOW_MODE" = "mvp" ]; then
  export MLFLOW_TRACKING_URI=https://mlflow.azere.net
else
  export MLFLOW_TRACKING_URI="file:$(pwd)/mlruns"
fi
```
