#!/usr/bin/env python3
"""
Cerebros Training Wrapper for Thunderline
Receives training data from Thunderline Oban worker and initiates Cerebros training.
"""

import sys
import json
import os
from pathlib import Path
import pandas as pd
from datetime import datetime


def load_training_config(payload_file):
    """Load training configuration from JSON payload"""
    with open(payload_file, 'r') as f:
        return json.load(f)


def prepare_training_data(csv_path):
    """Load and prepare CSV training data"""
    print(f"Loading training data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} training samples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Concatenate all text chunks for training
    training_text = "\n\n".join(df['text'].astype(str).tolist())
    
    return {
        'text': training_text,
        'chunk_count': len(df),
        'total_chars': len(training_text)
    }


def train_cerebros_model(config, training_data):
    """
    Main training function - calls Cerebros NAS for text generation
    
    For the demo, this is a placeholder that will:
    1. Save training data to expected format
    2. Call the actual Cerebros training script
    3. Return training results
    """
    agent_id = config['agent_id']
    agent_name = config['agent_name']
    model_config = config['model_config']
    
    print(f"\n{'='*60}")
    print(f"Starting Cerebros Training")
    print(f"{'='*60}")
    print(f"Agent ID: {agent_id}")
    print(f"Agent Name: {agent_name}")
    print(f"Model Type: {model_config['model_type']}")
    print(f"Training samples: {training_data['chunk_count']}")
    print(f"Total characters: {training_data['total_chars']}")
    print(f"{'='*60}\n")
    
    # Create output directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"mlruns/agent_{agent_id}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training text to file
    training_text_file = output_dir / "training_text.txt"
    with open(training_text_file, 'w') as f:
        f.write(training_data['text'])
    
    print(f"Saved training text to {training_text_file}")
    
    # For now, return success with placeholder results
    # In production, this would call the actual Cerebros training
    results = {
        'status': 'training_initiated',
        'agent_id': agent_id,
        'output_dir': str(output_dir),
        'training_text_file': str(training_text_file),
        'chunk_count': training_data['chunk_count'],
        'total_chars': training_data['total_chars'],
        'model_config': model_config,
        'timestamp': timestamp
    }
    
    # TODO: Call actual Cerebros training
    # This would involve:
    # 1. Tokenizing the text
    # 2. Running Cerebros NAS with SimpleCerebrosRandomSearch
    # 3. Training the generated model
    # 4. Saving the trained model
    # 5. Returning training metrics
    
    print("\n" + "="*60)
    print("Training initiated successfully!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Training text file: {training_text_file}")
    print("="*60 + "\n")
    
    return results


def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python train_model_wrapper.py <payload_json_file>")
        sys.exit(1)
    
    payload_file = sys.argv[1]
    
    try:
        # Load configuration
        config = load_training_config(payload_file)
        
        # Load training data
        training_data = prepare_training_data(config['csv_path'])
        
        # Train model
        results = train_cerebros_model(config, training_data)
        
        # Output results as JSON
        print("\n" + "="*60)
        print("TRAINING RESULTS:")
        print(json.dumps(results, indent=2))
        print("="*60)
        
        sys.exit(0)
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(f"ERROR: {error_msg}", file=sys.stderr)
        print(json.dumps({'status': 'error', 'error': error_msg}))
        sys.exit(1)


if __name__ == "__main__":
    main()
