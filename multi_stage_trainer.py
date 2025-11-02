#!/usr/bin/env python3
"""
Cerebros Multi-Stage Training Pipeline
Implements the full 5-stage training process for personalized AI assistants
Using full CEREBROS architecture search with embeddings, tokenization, and real training
"""

import sys
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import time
from typing import Dict, List, Optional
from gc import collect

import tensorflow as tf
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.models import Model

# Import Cerebros components
from cerebros.simplecerebrosrandomsearch.simple_cerebros_random_search import SimpleCerebrosRandomSearch
from cerebrosllmutils.llm_utils import (
    prepare_data,
    InterleavedRoPE,
    Perplexity,
    CerebrosNotGPTConfig,
    CerebrosNotGPT
)
from cerebros.units.units import DenseUnit
from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component import (
    zero_7_exp_decay,
    zero_95_exp_decay,
    simple_sigmoid
)
import pendulum


class MultiStageTrainer:
    """Handles the 5-stage training pipeline with full CEREBROS architecture"""
    
    def __init__(self, agent_id: str, agent_name: str, nfs_path: str = "priv/nfs"):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.nfs_path = Path(nfs_path)
        self.agent_path = self.nfs_path / "agents" / agent_id
        self.checkpoints_path = self.agent_path / "checkpoints"
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)
        
        # Stage configurations
        self.stages = {
            1: "Initial Foundation",
            2: "Domain Adaptation", 
            3: "Knowledge Integration",
            4: "Style Refinement",
            5: "Personalization"
        }
        
        # Initialize hyperparameters (from generative-proof-of-concept)
        self.MAX_SEQ_LENGTH = 40
        self.PROMPT_LENGTH = 1
        self.EMBEDDING_N = 6
        self.EMBEDDING_DIM = int(self.EMBEDDING_N * 2)
        self.PROJECTION_N = 1
        self.POSITIONAL_EMBEDDING_DROPOUT = 0.75
        
        # Architecture search params
        self.minimum_levels = 2
        self.maximum_levels = 2
        self.minimum_units_per_level = 2
        self.maximum_units_per_level = 3
        self.minimum_neurons_per_unit = 1
        self.maximum_neurons_per_unit = 2
        self.moities_to_try = 3
        self.tries_per_moity = 1
        
        # Training hyperparameters
        self.activation = 'relu'
        self.learning_rate = 0.004
        self.epochs = 40
        self.batch_size = 5
        self.gradient_accumulation_steps = 3
        
        # Connection params
        self.predecessor_level_connection_affinity_factor_first = 20.0
        self.predecessor_level_connection_affinity_factor_main = 15.0
        self.max_consecutive_lateral_connections = 4
        self.p_lateral_connection = 0.2
        self.num_lateral_connection_tries_per_unit = 20
        
        # Initialize tokenizer
        tokenizer_checkpoint = "HuggingFaceTB/SmolLM3-3B"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
        special_tokens = {
            "additional_special_tokens": ["<prompt>", "</prompt>", "<response>", "</response>"]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.VOCABULARY_SIZE = len(self.tokenizer)
        
        # Base model will be created on first use
        self.cerebros_base_model = None
        self.current_model = None
        self.generator = None
        
    def log(self, message: str, stage: Optional[int] = None):
        """Log training progress"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stage_prefix = f"[Stage {stage}]" if stage else "[SYSTEM]"
        print(f"{timestamp} {stage_prefix} {message}", flush=True)
    
    def create_base_model(self):
        """Create the embedding base model (like in generative-proof-of-concept)"""
        if self.cerebros_base_model is not None:
            return self.cerebros_base_model
        
        self.log("Creating embedding base model with InterleavedRoPE")
        
        inp = tf.keras.layers.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32)
        
        embedded = tf.keras.layers.Embedding(
            input_dim=self.VOCABULARY_SIZE,
            output_dim=self.EMBEDDING_DIM,
            input_length=self.MAX_SEQ_LENGTH,
            mask_zero=False
        )(inp)
        
        position_embedding = InterleavedRoPE(
            dim=self.EMBEDDING_DIM,
            max_seq_len=self.MAX_SEQ_LENGTH,
        )(embedded)
        
        x = tf.keras.layers.Concatenate()([embedded, position_embedding])
        x = tf.keras.layers.Dropout(self.POSITIONAL_EMBEDDING_DROPOUT)(x)
        flattened = tf.keras.layers.Flatten()(x)
        projected = tf.keras.layers.Dense(self.EMBEDDING_DIM * self.PROJECTION_N)(flattened)
        
        self.cerebros_base_model = tf.keras.Model(
            inputs=inp,
            outputs=projected
        )
        
        self.log("Base model created successfully")
        return self.cerebros_base_model
    
    def load_training_data(self, data_type: str) -> List[str]:
        """Load training text samples from CSV files"""
        data_path = self.agent_path / "processed" / f"{data_type}.csv"
        if data_path.exists():
            self.log(f"Loading {data_type} data from {data_path}")
            df = pd.read_csv(data_path)
            # Extract text from prompt/response columns
            texts = []
            if 'prompt' in df.columns and 'response' in df.columns:
                for _, row in df.iterrows():
                    texts.append(f"<prompt>{row['prompt']}</prompt><response>{row['response']}</response>")
            elif 'text' in df.columns:
                texts = df['text'].tolist()
            else:
                # Fallback: concatenate all string columns
                texts = df.astype(str).agg(' '.join, axis=1).tolist()
            return texts
        else:
            self.log(f"Warning: {data_type} data not found at {data_path}", None)
            return []
    
    def merge_text_samples(self, *text_lists: List[str]) -> List[str]:
        """Merge multiple text sample lists"""
        merged = []
        for text_list in text_lists:
            if text_list:
                merged.extend(text_list)
        return merged
    
    def save_checkpoint(self, stage: int, metrics: Dict, model: Model = None) -> str:
        """Save model checkpoint"""
        checkpoint_name = f"stage_{stage}_checkpoint.keras"
        checkpoint_path = self.checkpoints_path / checkpoint_name
        
        # Save the actual Keras model
        if model is not None:
            model.save(str(checkpoint_path))
            self.log(f"Keras model saved: {checkpoint_path}", stage)
        elif self.generator is not None:
            self.generator.save(str(checkpoint_path))
            self.log(f"Generator model saved: {checkpoint_path}", stage)
        else:
            self.log(f"Warning: No model to save for stage {stage}", stage)
        
        # Save checkpoint metadata
        metadata = {
            "stage": stage,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "checkpoint_file": str(checkpoint_path)
        }
        
        metadata_path = self.checkpoints_path / f"stage_{stage}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, stage: int) -> Optional[Model]:
        """Load previous stage checkpoint"""
        checkpoint_name = f"stage_{stage}_checkpoint.keras"
        checkpoint_path = self.checkpoints_path / checkpoint_name
        
        if checkpoint_path.exists():
            self.log(f"Loading checkpoint: {checkpoint_path}", stage)
            try:
                model = tf.keras.models.load_model(str(checkpoint_path))
                return model
            except Exception as e:
                self.log(f"Error loading checkpoint: {e}", stage)
                return None
        return None
    
    def train_with_cerebros(self, stage: int, text_samples: List[str], is_foundation: bool = False) -> Dict:
        """
        Train using full CEREBROS architecture search (like generative-proof-of-concept)
        
        Args:
            stage: Current stage number
            text_samples: List of training text samples
            is_foundation: If True, run full architecture search; if False, fine-tune existing model
        """
        if not text_samples:
            self.log(f"No training samples for stage {stage}, skipping", stage)
            return {"status": "skipped", "reason": "no_data"}
        
        self.log(f"Preparing {len(text_samples)} samples for training", stage)
        
        # Prepare data with tokenizer
        x, y, vocab_size = prepare_data(
            data_0=text_samples,
            tokenizer_0=self.tokenizer,
            max_seq_length=self.MAX_SEQ_LENGTH,
            prompt_length=self.PROMPT_LENGTH
        )
        
        self.log(f"Tokenized data: {len(x)} samples, vocab size: {vocab_size}", stage)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.15, shuffle=False
        )
        
        # Convert to TensorFlow tensors
        x_train_tf = tf.constant(X_train, tf.int32)
        y_train_tf = tf.constant(y_train, tf.float32)
        x_test_tf = tf.constant(X_test, tf.int32)
        y_test_tf = tf.constant(y_test, tf.float32)
        
        x_train_packaged = [x_train_tf]
        y_train_packaged = [y_train_tf]
        x_test_packaged = [x_test_tf]
        y_test_packaged = [y_test_tf]
        
        INPUT_SHAPES = [(self.MAX_SEQ_LENGTH,)]
        OUTPUT_SHAPES = [(self.VOCABULARY_SIZE,)]
        
        if is_foundation or self.current_model is None:
            # Stage 1: Full architecture search
            self.log("Running CEREBROS architecture search", stage)
            
            # Create base embedding model
            base_model = self.create_base_model()
            
            # Setup CEREBROS AutoML
            TIME = pendulum.now(tz='America/New_York').__str__()[:16]\
                .replace('T', '_')\
                .replace(':', '_')\
                .replace('-', '_')
            PROJECT_NAME = f'{TIME}_cerebros_{self.agent_name}_stage_{stage}'
            
            perplexity_metric = Perplexity()
            
            cerebros_automl = SimpleCerebrosRandomSearch(
                unit_type=DenseUnit,
                input_shapes=INPUT_SHAPES,
                output_shapes=OUTPUT_SHAPES,
                training_data=x_train_packaged,
                labels=y_train_packaged,
                validation_split=0.2,
                direction='minimize',
                metric_to_rank_by="perplexity",
                minimum_levels=self.minimum_levels,
                maximum_levels=self.maximum_levels,
                minimum_units_per_level=self.minimum_units_per_level,
                maximum_units_per_level=self.maximum_units_per_level,
                minimum_neurons_per_unit=self.minimum_neurons_per_unit,
                maximum_neurons_per_unit=self.maximum_neurons_per_unit,
                activation=self.activation,
                final_activation='softmax',
                number_of_architecture_moities_to_try=self.moities_to_try,
                number_of_tries_per_architecture_moity=self.tries_per_moity,
                minimum_skip_connection_depth=1,
                maximum_skip_connection_depth=7,
                predecessor_level_connection_affinity_factor_first=self.predecessor_level_connection_affinity_factor_first,
                predecessor_level_connection_affinity_factor_first_rounding_rule='ceil',
                predecessor_level_connection_affinity_factor_main=self.predecessor_level_connection_affinity_factor_main,
                predecessor_level_connection_affinity_factor_main_rounding_rule='ceil',
                predecessor_level_connection_affinity_factor_decay_main=zero_7_exp_decay,
                seed=8675309,
                max_consecutive_lateral_connections=self.max_consecutive_lateral_connections,
                gate_after_n_lateral_connections=3,
                gate_activation_function=simple_sigmoid,
                p_lateral_connection=self.p_lateral_connection,
                p_lateral_connection_decay=zero_95_exp_decay,
                num_lateral_connection_tries_per_unit=self.num_lateral_connection_tries_per_unit,
                learning_rate=self.learning_rate,
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy(), perplexity_metric],
                epochs=self.epochs,
                project_name=PROJECT_NAME,
                model_graphs='model_graphs',
                batch_size=self.batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                meta_trial_number=stage,
                base_models=[base_model],
                train_data_dtype=tf.int32
            )
            
            t0 = time.time()
            result = cerebros_automl.run_random_search()
            t1 = time.time()
            
            self.log(f"Architecture search completed in {(t1-t0)/60:.2f} min, perplexity: {result:.4f}", stage)
            
            # Get best model
            best_model = cerebros_automl.get_best_model(purge_model_storage_files='slate')
            self.current_model = best_model
            
            # Create generator
            config = CerebrosNotGPTConfig(
                max_sequence_length=self.MAX_SEQ_LENGTH,
                padding_token=self.tokenizer.pad_token_id
            )
            self.generator = CerebrosNotGPT(config, model=best_model)
            
            # Build the model
            input_tokens = [self.tokenizer.pad_token_id] * self.MAX_SEQ_LENGTH
            input_tensor = tf.constant([input_tokens], dtype=tf.int32)
            _ = self.generator(input_tensor)
            
            metrics = {
                "perplexity": float(result),
                "training_time_minutes": (t1-t0)/60
            }
            
            del cerebros_automl
            collect()
            
        else:
            # Fine-tune existing model
            self.log("Fine-tuning existing model", stage)
            
            # Recompile with updated learning rate
            fine_tune_lr = self.learning_rate * 0.5  # Lower LR for fine-tuning
            perplexity_metric = Perplexity(name=f"perplexity_stage_{stage}")
            
            self.generator.model.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy(), perplexity_metric],
                optimizer=AdamW(
                    learning_rate=fine_tune_lr,
                    weight_decay=0.01,
                    gradient_accumulation_steps=self.gradient_accumulation_steps
                ),
                jit_compile=True
            )
            
            # Prepare dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train_tf, y_train_tf))
            train_dataset = train_dataset.batch(self.batch_size)
            
            val_dataset = tf.data.Dataset.from_tensor_slices((x_test_tf, y_test_tf))
            val_dataset = val_dataset.batch(self.batch_size)
            
            # Train
            t0 = time.time()
            history = self.generator.model.fit(
                x=train_dataset,
                validation_data=val_dataset,
                epochs=self.epochs
            )
            t1 = time.time()
            
            history_df = pd.DataFrame(history.history)
            final_perplexity = float(history_df[f'perplexity_stage_{stage}'].min())
            
            self.log(f"Fine-tuning completed in {(t1-t0)/60:.2f} min, perplexity: {final_perplexity:.4f}", stage)
            
            metrics = {
                "perplexity": final_perplexity,
                "training_time_minutes": (t1-t0)/60,
                "history": history.history
            }
            
            collect()
        
        return {
            "status": "success",
            "metrics": metrics
        }
    
    def stage_1_foundation(self) -> Dict:
        """Stage 1: Initial Foundation Training with CEREBROS architecture search"""
        self.log("=" * 60, 1)
        self.log("Starting Stage 1: Initial Foundation Training", 1)
        self.log("=" * 60, 1)
        
        # Load base training data
        stage1_texts = self.load_training_data("stage1_base")
        
        if not stage1_texts:
            self.log("Warning: No stage1 data found, using fallback samples", 1)
            stage1_texts = [
                "This is foundational training data for language understanding.",
                "Learning basic patterns and structures in text.",
                "Understanding context and semantic relationships."
            ] * 5
        
        self.log(f"Training with {len(stage1_texts)} text samples", 1)
        
        # Run CEREBROS architecture search
        result = self.train_with_cerebros(stage=1, text_samples=stage1_texts, is_foundation=True)
        
        if result["status"] == "success":
            metrics = result["metrics"]
            # Save checkpoint
            checkpoint_path = self.save_checkpoint(1, metrics)
            self.log("Stage 1 complete!", 1)
            return {"checkpoint": checkpoint_path, "metrics": metrics}
        else:
            return result
    
    def stage_2_domain_adaptation(self) -> Dict:
        """Stage 2: Domain Adaptation with fine-tuning"""
        self.log("=" * 60, 2)
        self.log("Starting Stage 2: Domain Adaptation", 2)
        self.log("=" * 60, 2)
        
        # Load Stage 1 checkpoint (model already loaded in self.generator)
        if self.generator is None:
            self.log("Loading Stage 1 checkpoint...", 2)
            loaded_model = self.load_checkpoint(1)
            if loaded_model is None:
                raise ValueError("Stage 1 checkpoint not found")
            config = CerebrosNotGPTConfig(
                max_sequence_length=self.MAX_SEQ_LENGTH,
                padding_token=self.tokenizer.pad_token_id
            )
            self.generator = CerebrosNotGPT(config, model=loaded_model)
            self.current_model = loaded_model
        
        # Load domain-specific data
        stage2_relevant = self.load_training_data("stage2_relevant")
        stage2_general = self.load_training_data("stage2_general")
        
        # Merge
        merged_texts = self.merge_text_samples(stage2_relevant, stage2_general)
        self.log(f"Training with {len(merged_texts)} merged samples", 2)
        
        # Fine-tune
        result = self.train_with_cerebros(stage=2, text_samples=merged_texts, is_foundation=False)
        
        if result["status"] == "success":
            metrics = result["metrics"]
            checkpoint_path = self.save_checkpoint(2, metrics)
            self.log("Stage 2 complete!", 2)
            return {"checkpoint": checkpoint_path, "metrics": metrics}
        else:
            return result
    
    def stage_3_knowledge_integration(self) -> Dict:
        """Stage 3: Knowledge Integration"""
        self.log("=" * 60, 3)
        self.log("Starting Stage 3: Knowledge Integration", 3)
        self.log("=" * 60, 3)
        
        # Load training data
        stage3_relevant = self.load_training_data("stage3_relevant")
        stage3_general = self.load_training_data("stage3_general")
        reference_data = self.load_training_data("reference_knowledge_base")
        
        # Merge all datasets
        merged_texts = self.merge_text_samples(stage3_relevant, stage3_general, reference_data)
        self.log(f"Training with {len(merged_texts)} merged samples", 3)
        
        # Fine-tune
        result = self.train_with_cerebros(stage=3, text_samples=merged_texts, is_foundation=False)
        
        if result["status"] == "success":
            metrics = result["metrics"]
            checkpoint_path = self.save_checkpoint(3, metrics)
            self.log("Stage 3 complete!", 3)
            return {"checkpoint": checkpoint_path, "metrics": metrics}
        else:
            return result
    
    def stage_4_style_refinement(self) -> Dict:
        """Stage 4: Style Refinement"""
        self.log("=" * 60, 4)
        self.log("Starting Stage 4: Style Refinement", 4)
        self.log("=" * 60, 4)
        
        # Load training data
        stage4_relevant = self.load_training_data("stage4_relevant")
        stage4_general = self.load_training_data("stage4_general")
        
        # Merge
        merged_texts = self.merge_text_samples(stage4_relevant, stage4_general)
        self.log(f"Training with {len(merged_texts)} merged samples", 4)
        
        # Fine-tune
        result = self.train_with_cerebros(stage=4, text_samples=merged_texts, is_foundation=False)
        
        if result["status"] == "success":
            metrics = result["metrics"]
            checkpoint_path = self.save_checkpoint(4, metrics)
            self.log("Stage 4 complete!", 4)
            return {"checkpoint": checkpoint_path, "metrics": metrics}
        else:
            return result
    
    def stage_5_personalization(self) -> Dict:
        """Stage 5: Personalization Fine-Tuning"""
        self.log("=" * 60, 5)
        self.log("Starting Stage 5: Personalization Fine-Tuning", 5)
        self.log("=" * 60, 5)
        
        # Load user-specific data
        work_products = self.load_training_data("work_products_augmented")
        prompts = self.load_training_data("prompts_responses_augmented")
        communications = self.load_training_data("communications_augmented")
        
        # Merge all user data
        merged_texts = self.merge_text_samples(work_products, prompts, communications)
        self.log(f"Training with {len(merged_texts)} personalization samples", 5)
        
        # Fine-tune with more epochs for personalization
        original_epochs = self.epochs
        self.epochs = max(50, self.epochs)  # More epochs for final personalization
        
        result = self.train_with_cerebros(stage=5, text_samples=merged_texts, is_foundation=False)
        
        self.epochs = original_epochs  # Restore
        
        if result["status"] == "success":
            metrics = result["metrics"]
            checkpoint_path = self.save_checkpoint(5, metrics)
            
            # Save final model metadata
            model_metadata = {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "status": "ready",
                "final_checkpoint": checkpoint_path,
                "deployment_ready": True,
                "created_at": datetime.now().isoformat(),
                "metrics": metrics,
                "tokenizer_vocab_size": self.VOCABULARY_SIZE,
                "max_seq_length": self.MAX_SEQ_LENGTH
            }
            
            metadata_path = self.agent_path / "model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            # Save tokenizer
            tokenizer_path = self.agent_path / "tokenizer"
            self.tokenizer.save_pretrained(str(tokenizer_path))
            self.log(f"Tokenizer saved to {tokenizer_path}", 5)
            
            self.log("Stage 5 complete! Model ready for deployment!", 5)
            return {
                "checkpoint": checkpoint_path,
                "metrics": metrics,
                "status": "ready_for_deployment"
            }
        else:
            return result
    
    def run_full_pipeline(self) -> Dict:
        """Execute the complete 5-stage training pipeline"""
        self.log("=" * 80)
        self.log(f"Starting Multi-Stage Training Pipeline for Agent: {self.agent_name}")
        self.log(f"Agent ID: {self.agent_id}")
        self.log("=" * 80)
        
        results = {}
        
        try:
            # Stage 1: Foundation
            results['stage_1'] = self.stage_1_foundation()
            
            # Stage 2: Domain Adaptation
            results['stage_2'] = self.stage_2_domain_adaptation()
            
            # Stage 3: Knowledge Integration
            results['stage_3'] = self.stage_3_knowledge_integration()
            
            # Stage 4: Style Refinement
            results['stage_4'] = self.stage_4_style_refinement()
            
            # Stage 5: Personalization
            results['stage_5'] = self.stage_5_personalization()
            
            self.log("=" * 80)
            self.log("🎉 All 5 stages completed successfully!")
            self.log("=" * 80)
            
            # Find the last successfully completed stage checkpoint
            final_checkpoint = None
            for stage_num in [5, 4, 3, 2, 1]:
                stage_key = f'stage_{stage_num}'
                if results.get(stage_key, {}).get('checkpoint'):
                    final_checkpoint = results[stage_key]['checkpoint']
                    break
            
            return {
                "status": "success",
                "agent_id": self.agent_id,
                "stages": results,
                "final_model": final_checkpoint
            }
            
        except Exception as e:
            self.log(f"❌ Pipeline failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "completed_stages": results
            }


def main():
    """Main entry point"""
    if len(sys.argv) < 3:
        print("Usage: python multi_stage_trainer.py <agent_id> <agent_name> [nfs_path]")
        sys.exit(1)
    
    agent_id = sys.argv[1]
    agent_name = sys.argv[2]
    nfs_path = sys.argv[3] if len(sys.argv) > 3 else "priv/nfs"
    
    trainer = MultiStageTrainer(agent_id, agent_name, nfs_path)
    results = trainer.run_full_pipeline()
    
    # Output final results as JSON
    print("\n" + "=" * 80)
    print("FINAL RESULTS:")
    print(json.dumps(results, indent=2))
    print("=" * 80)
    
    sys.exit(0 if results["status"] == "success" else 1)


if __name__ == "__main__":
    main()
