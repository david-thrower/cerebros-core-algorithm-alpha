# Tokenize-first Phishing Email Classification

This doc explains the new prepare → train workflow for `phishing_email_detection_gpt2.py`.

## Prepare tokens

Input can be a CSV with columns `Email Text, Email Type` (mapped to labels) or a generic CSV with `text,label`, or a JSONL with `{"text": ..., "label": 0|1}`.

Example (small, CPU-safe):

```bash
python phishing_email_detection_gpt2.py --mode prepare \
  --in Phishing_Email.csv \
  --out data/train_tokens.npz \
  --max_len 128 \
  --tokenizer_checkpoint HuggingFaceTB/SmolLM3-3B
```

## Train from cache

```bash
python phishing_email_detection_gpt2.py --mode train \
  --cache data/train_tokens.npz \
  --epochs 1 --batch 8 --print-score-only
```

If `MLFLOW_TRACKING_URI` is set, params/metrics and the model artifact are logged to MLflow.

## Docker (GPU-ready)

```bash
# Build
docker build -t thunder/poc:tf2.19 .

# Prepare
docker run --rm -it -v "$PWD":/app --gpus all \
  thunder/poc:tf2.19 \
  python phishing_email_detection_gpt2.py --mode prepare \
  --in Phishing_Email.csv --out data/train_tokens.npz --max_len 128 \
  --tokenizer_checkpoint HuggingFaceTB/SmolLM3-3B

# Train
docker run --rm -it -v "$PWD":/app --gpus all \
  thunder/poc:tf2.19 \
  python phishing_email_detection_gpt2.py --mode train \
  --cache data/train_tokens.npz --epochs 1 --batch 8 --print-score-only
```

Speed tip: mount your HF cache: `-v $HOME/.cache/huggingface:/root/.cache/huggingface`.

## MLflow backed by Postgres (docker-compose)

Spin up Postgres + MLflow locally (persistent volumes included):

```bash
cd infra/mlflow-postgres
docker compose up -d --build
```

Set your client env and run the scripts:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Stop the stack when done:

```bash
docker compose down
```
