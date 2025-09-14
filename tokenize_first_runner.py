#!/usr/bin/env python3
"""
Tokenize-First Runner for Cerebros Phishing Detection
Implements the tokenize → cache → train workflow for K8s deployment
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import mlflow

def tokenize_texts(texts, tokenizer, max_seq_length):
    """Tokenize texts using HuggingFace tokenizer"""
    tokenized = []
    texts_as_list = [str(s) for s in texts.tolist()]
    for text in texts_as_list:
        tokens = tokenizer(
            text,
            max_length=max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        tokenized.append(tokens['input_ids'][0])
    return np.array(tokenized)

def _load_dataframe_from_input(input_path: str) -> pd.DataFrame:
    """Load a DataFrame from CSV or JSONL with common schemas.

    Supported formats:
      - CSV with columns: Email Text, Email Type (label mapped to 0/1)
      - CSV with columns: text, label
      - JSONL with objects: {"text": str, "label": int}
    """
    if input_path.lower().endswith(".jsonl"):
        rows = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append({
                    "text": str(obj.get("text", "")),
                    "label": int(obj.get("label", 0))
                })
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("Input JSONL parsed to empty DataFrame")
        return df

    # CSV
    df = pd.read_csv(input_path)
    # Schema 1: Phishing_Email.csv
    if {"Email Text", "Email Type"}.issubset(df.columns):
        df = df[df["Email Text"].apply(lambda x: isinstance(x, str))].copy()
        label_mapping = {"Safe Email": 0, "Phishing Email": 1}
        df["text"] = df["Email Text"].astype(str)
        df["label"] = df["Email Type"].map(label_mapping).astype(int)
        return df[["text", "label"]]

    # Schema 2: generic text,label
    if {"text", "label"}.issubset(df.columns):
        df = df[["text", "label"]].copy()
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(int)
        return df

    raise ValueError("Unsupported CSV schema. Expected columns: [Email Text, Email Type] or [text, label]")


def prepare_tokens(args):
    """Prepare and cache tokenized data"""
    print(f"🔄 Preparing tokens: {args.input} → {args.output}")

    # Determine input path: explicit --input wins; else fallback to Phishing_Email.csv
    input_path = args.input if args.input and os.path.exists(args.input) else None
    if input_path is None and os.path.exists("Phishing_Email.csv"):
        input_path = "Phishing_Email.csv"

    if not input_path:
        print("❌ Error: No input found. Provide --input or include Phishing_Email.csv in working directory.")
        return False

    try:
        df = _load_dataframe_from_input(input_path)
    except Exception as e:
        print(f"❌ Failed to load input '{input_path}': {e}")
        return False

    # Basic cleaning
    df = df[df["text"].apply(lambda x: isinstance(x, str))].copy()
    df["text"] = df["text"].astype(str)
    df.reset_index(drop=True, inplace=True)

    X = df["text"].to_numpy()
    y = df["label"].to_numpy()
    X, y = shuffle(X, y)

    # 85% test split as in original
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.85, shuffle=False)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"📏 Tokenizing with max_length={args.max_len}")
    print(f"🧠 Using tokenizer: {args.tokenizer_checkpoint}")
    print(f"📊 Vocab size: {len(tokenizer)}")

    # Tokenize
    train_tokens = tokenize_texts(X_train, tokenizer, args.max_len)
    test_tokens = tokenize_texts(X_test, tokenizer, args.max_len)

    # Save to npz
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    np.savez_compressed(
        args.output,
        train_tokens=train_tokens,
        train_labels=y_train,
        test_tokens=test_tokens,
        test_labels=y_test,
        vocab_size=len(tokenizer),
        max_len=args.max_len
    )

    print(f"✅ Tokens cached to {args.output}")
    print(f"📈 Train samples: {len(train_tokens)}, Test samples: {len(test_tokens)}")
    return True

def train_from_cache(args):
    """Train model from cached tokens"""
    print(f"🚀 Training from cache: {args.cache}")

    if not os.path.exists(args.cache):
        print(f"❌ Error: Cache file {args.cache} not found")
        return False

    # Load cached data
    data = np.load(args.cache)
    train_tokens = data['train_tokens']
    train_labels = data['train_labels']
    test_tokens = data['test_tokens']
    test_labels = data['test_labels']
    vocab_size = int(data['vocab_size'])
    max_len = int(data['max_len'])

    print(f"📊 Loaded: train={len(train_tokens)}, test={len(test_tokens)}, vocab={vocab_size}")

    # Clear any existing models from memory
    tf.keras.backend.clear_session()

    # Build simple model for smoke test
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32),
        tf.keras.layers.Embedding(vocab_size, 128, mask_zero=True),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    print("🏗️ Model compiled")

    # Train
    with mlflow.start_run():
        mlflow.log_params({
            "vocab_size": vocab_size,
            "max_len": max_len,
            "epochs": args.epochs,
            "batch_size": args.batch_size
        })

        history = model.fit(
            train_tokens,
            train_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=0.2,
            verbose=1 if not args.print_score_only else 0
        )

        # Test evaluation
        test_loss, test_acc, test_prec, test_rec = model.evaluate(
            test_tokens, test_labels, verbose=0
        )

        # Log metrics
        mlflow.log_metrics({
            "test_accuracy": test_acc,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_loss": test_loss
        })

        # Save model artifact
        model_path = "cerebros-tokenized-model.keras"
        model.save(model_path)
        mlflow.log_artifact(model_path)

        # Print model size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"💾 Model size: {model_size_mb:.2f} MB")

        if args.print_score_only:
            print(f"{test_acc:.4f}")  # Single scalar for orchestration
        else:
            print(f"🎯 Test Accuracy: {test_acc:.4f}")
            print(f"🎯 Test Precision: {test_prec:.4f}")
            print(f"🎯 Test Recall: {test_rec:.4f}")

    # Clean up
    tf.keras.backend.clear_session()
    return True

def main():
    parser = argparse.ArgumentParser(description="Tokenize-First Cerebros Runner")
    parser.add_argument("--mode", choices=["prepare", "train"], required=True,
                       help="Mode: prepare tokens or train from cache")

    # Prepare mode args
    parser.add_argument("--input", "--in", default="data/train.jsonl",
                       help="Input data file (for prepare mode)")
    parser.add_argument("--output", "--out", default="data/train_tokens.npz",
                       help="Output token cache file (for prepare mode)")
    parser.add_argument("--max_len", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--tokenizer_checkpoint", default="HuggingFaceTB/SmolLM3-3B",
                       help="HuggingFace tokenizer checkpoint")

    # Train mode args
    parser.add_argument("--cache", default="data/train_tokens.npz",
                       help="Cached token file (for train mode)")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Training epochs")
    parser.add_argument("--batch", "--batch_size", type=int, default=8, dest="batch_size",
                       help="Batch size")
    parser.add_argument("--print-score-only", action="store_true",
                       help="Print only final scalar score")

    args = parser.parse_args()

    if args.mode == "prepare":
        success = prepare_tokens(args)
    elif args.mode == "train":
        success = train_from_cache(args)
    else:
        print("❌ Invalid mode")
        success = False

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()