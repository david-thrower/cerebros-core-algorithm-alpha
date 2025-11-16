
from ast import literal_eval
import time
from gc import collect
import re


import tensorflow as tf
import pandas as pd
import numpy as np
import pendulum


from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from cerebros.units.units import DenseUnit
from cerebros.simplecerebrosrandomsearch.simple_cerebros_random_search\
    import SimpleCerebrosRandomSearch
from cerebrosllmutils.llm_utils import prepare_data, \
                                       InterleavedRoPE, \
                                       Perplexity, \
                                       CerebrosNotGPTConfig, \
                                       CerebrosNotGPT
from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component\
    import zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid

from vanilladatasets.web_english_bible import samples as bible   
    
    
PHASE_I_A_SAMPLES_TO_CREATE = 10 # 681
PHASE_I_B_SAMPLES_TO_CREATE = 20
PHASE_I_B_VAL_SPLIT = 0.15  # Validation split for phase I-b (0.0 to 1.0)

PHASE_I_B_SAMPLE_EXPANSION_BATCH_SIZE = 10

# How many tokens to provide before expecting the next token to be predicted. 
# Half this = double RAM  (inversely proportional to RAM requirement)
PROMPT_LENGTH = 1
    
# Text encoding / embedding related constants
    
    
MAX_SEQ_LENGTH = 40 # 1536 (Linear and directly proportional to RAM requirement)

#
# Cerebros [non-HP-tunable] configurables (Parameters to Optimize continued)
#

    
moities_to_try = 3 # ++ Accuracy, linear increase in computation time (Raise this before resorting to raising the next one)
tries_per_moity = 1 # ++ Modest ++ Accuracy, quadratic increase in computation time 

## Generation time configurables: ##########

GENERATION_PROMPT_LEN = 25
MAX_NEW_TOKENS = MAX_SEQ_LENGTH - GENERATION_PROMPT_LEN

# Tunable parameters:

POSITIONAL_EMBEDDING_DROPOUT = trial.suggest_float('POSITIONAL_EMBEDDING_DROPOUT', 0.72, 0.8)

activation = trial.suggest_categorical('activation', ['relu', 'gelu', 'swish', 'softsign', 'softplus'])

predecessor_level_connection_affinity_factor_first = trial.suggest_float('predecessor_level_connection_affinity_factor_first', 10.0, 30.0)

predecessor_level_connection_affinity_factor_main = trial.suggest_float('predecessor_level_connection_affinity_factor_main', 10.0, 25.0)

max_consecutive_lateral_connections = trial.suggest_int('max_consecutive_lateral_connections', 2, 7)

p_lateral_connection = trial.suggest_float('p_lateral_connection', 0.12, 0.35)

num_lateral_connection_tries_per_unit = trial.suggest_int('num_lateral_connection_tries_per_unit', 10, 35)
    
learning_rate = trial.suggest_float('learning_rate', 0.003, 0.006) # log=True)
phase_i_b_learning_rate = trial.suggest_float('phase_i_b_learning_rate', 0.0001, 0.006)
    
    
epochs = trial.suggest_int('epochs', 30, 75)
phase_i_b_epochs =  trial.suggest_int('phase_i_b_epochs', 40, 60)
    
batch_size = 5 # trial.suggest_int('batch_size', 5, 10)

gradient_accumulation_steps = trial.suggest_int('gradient_accumulation_steps', 1, 7)

phase_i_b_gradient_accumulation_steps = trial.suggest_int("phase_i_b_gradient_accumulation_steps", 2, 20)
phase_i_b_weight_decay = trial.suggest_float("phase_i_b_weight_decay", 0.004, 0.1)
    
# Level constraints - ensure max >= min by setting min of max to value of min
minimum_levels = 2 # trial.suggest_int('minimum_levels', 1, 3)
maximum_levels = 2 # trial.suggest_int('maximum_levels', minimum_levels, 3)
    
# Units per level - ensure max >= min by setting min of max to value of min
minimum_units_per_level = trial.suggest_int('minimum_units_per_level', 2, 3)
maximum_units_per_level = trial.suggest_int('maximum_units_per_level', minimum_units_per_level, 3)
    
# Neurons per unit - ensure max >= min by setting min of max to value of min
minimum_neurons_per_unit = trial.suggest_int('minimum_neurons_per_unit', 1, 2)
maximum_neurons_per_unit = trial.suggest_int('maximum_neurons_per_unit', minimum_neurons_per_unit, 2)

    
tokenizer_checkpoint = "HuggingFaceTB/SmolLM3-3B" # "HuggingFaceTB/SmolLM2-1.7B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    
# Step 1: Add special tokens
special_tokens = {
    "additional_special_tokens": ["<prompt>", "</prompt>", "<response>", "</response>"]
}
tokenizer.add_special_tokens(special_tokens)
    
VOCABULARY_SIZE = len(tokenizer)
    
# For interleaved Rotary Positional Embedding (iRoPE), the 
# embedding output dim must be an even number
# Maximize EMBEDDING_N based on available RAM and CPU / GPU
    
EMBEDDING_N = 6 # trial.suggest_int('embedding_n',6, 9) # 12
EMBEDDING_DIM = int(EMBEDDING_N * 2)
    
PROJECTION_N = 1 # Punatuve increase of ram, leaving this as 1 until we are running on HPC

## Get training data:

non_instruct_samples = bible[:PHASE_I_A_SAMPLES_TO_CREATE]
phase_i_b_samples = bible[PHASE_I_A_SAMPLES_TO_CREATE:PHASE_I_B_SAMPLES_TO_CREATE + PHASE_I_A_SAMPLES_TO_CREATE] 
print(f"Samples from KJV bible consisting of {len(non_instruct_samples)} look like this (sub-sample of 3): {non_instruct_samples[:3]}")

# Split the phase I-b data set for training and validation:

phase_i_b_train_samples, phase_i_b_val_samples = train_test_split(
        phase_i_b_samples, 
        test_size=PHASE_I_B_VAL_SPLIT, 
        shuffle=False
)


# 
        
X_train, X_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.85, shuffle=False)
        
INPUT_SHAPES = [(MAX_SEQ_LENGTH,)]
OUTPUT_SHAPES = [(VOCABULARY_SIZE)]
        
x_train_tf = tf.constant(X_train, tf.int32)
y_train_tf = tf.constant(y_train, tf.float32)
        
x_train_packaged = [x_train_tf]
y_train_packaged = [y_train_tf]
        
x_test_tf = tf.constant(X_test, tf.int32)
y_test_tf = tf.constant(y_test, tf.float32)
        
x_test_packaged = [x_test_tf] 
y_test_packaged = [y_test_tf]
        



    
    

