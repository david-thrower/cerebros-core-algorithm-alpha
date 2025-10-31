"""
CEREBROS Multi-Stage Trainer
----------------------------

Chains together staged model training based on datasets created by process_user_samples.py:
- Reads training_stage{1..4}.csv from NFS datasets directory.
- Performs progressive fine-tuning for each stage using reduced Qwen7B.
- Logs metrics (loss, accuracy, tokens/s) to MLflow.
- Saves checkpoints to /priv/nfs/{assistant_id}/checkpoints/stage*.keras.

Usage:
    python3 scripts/multi_stage_trainer.py <agent_id> <agent_name>
"""

import os, time, argparse, json, gc, mlflow
from pathlib import Path
import numpy as np
from typing import Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

CHECKPOINT_PATH = "priv/nfs/{agent_id}/checkpoints"
DATASETS_PATH = "priv/nfs/{agent_id}/datasets"
TOKENIZER_CHECKPOINT = "Qwen/Qwen2.5-7B-Instruct"
MAX_LEN = 512

def load_dataset(file_path: str, tokenizer):
    """Creates a line dataset for LM training."""
    from datasets import Dataset
    import pandas as pd
    df = pd.read_csv(file_path)
    texts = (df["prompt"] + "\n" + df["think"] + "\n" + df["response"]).tolist()
    tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LEN)
    return Dataset.from_dict(tokenized)

def train_stage(stage_id: int, model, tokenizer, dataset, output_dir: Path):
    """Train for one stage and log metrics."""
    with mlflow.start_run(run_name=f"stage{stage_id}"):
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=1,
            per_device_train_batch_size=1,
            save_total_limit=1,
            logging_steps=10,
            learning_rate=2e-5,
            report_to=[],
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=data_collator)
        start = time.time()
        trainer.train()
        duration = time.time() - start

        metrics = {"stage": stage_id, "duration_s": duration, "samples": len(dataset)}
        mlflow.log_metrics(metrics)

        # Save model
        output_ckpt = output_dir / f"stage{stage_id}.keras"
        model.save_pretrained(output_ckpt)
        print(f"✓ Stage {stage_id} completed → {output_ckpt}")

def main(agent_id: str, agent_name: str):
    base_datasets = Path(DATASETS_PATH.format(agent_id=agent_id))
    ckpt_dir = Path(CHECKPOINT_PATH.format(agent_id=agent_id))
    os.makedirs(ckpt_dir, exist_ok=True)
    mlflow.set_experiment(f"cerebros_multistage_training_{agent_id}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CHECKPOINT, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(TOKENIZER_CHECKPOINT)

    stages = sorted([p for p in base_datasets.glob("training_stage*.csv")])
    if not stages:
        print("✗ No dataset CSVs found.")
        return

    print(f"Found {len(stages)} stage datasets for {agent_name}")
    for i, stage_file in enumerate(stages, start=1):
        dataset = load_dataset(stage_file, tokenizer)
        train_stage(i, model, tokenizer, dataset, ckpt_dir)
        gc.collect()

    print(f"✓ All stages complete. Final checkpoint: {ckpt_dir / 'stage5.keras'} (if present)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-stage trainer for CEREBROS NotGPT")
    parser.add_argument("agent_id", help="Agent ID / assistant ID")
    parser.add_argument("agent_name", help="Agent Display name")
    args = parser.parse_args()
    main(args.agent_id, args.agent_name)