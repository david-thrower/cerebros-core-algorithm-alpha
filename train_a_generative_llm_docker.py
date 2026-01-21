import os
import subprocess
import time
from gc import collect
from os import getenv
from pathlib import Path
from time import sleep
from contextlib import nullcontext

import tensorflow as tf
import pandas as pd
import pendulum
import mlflow

from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Cerebros NAS components
from cerebros.units.units import DenseUnit
from cerebros.simplecerebrosrandomsearch.simple_cerebros_random_search \
    import SimpleCerebrosRandomSearch

# LLM Components
from cerebrosllmutils.llm_utils import (
    prepare_data,
    InterleavedRoPE,
    SparsePerplexity,
    GatedMergeLayer,
    ManifoldHyperConnect,
    ChunkedAttentionBlock,
    MambaBlock,
    VoxelBlock,
    LinformerBlock,
    AdapterBlock,
    CerebrosNotGPTConfig,
    CerebrosNotGPT,
    WarmupCosineDecayRestarts
)

from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component \
    import zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid

# Platform engineering constants and variables:


ARTIFACTS_FOLDER = "/opt/artifacts"

OWNER = getenv("OWNER", "cerebros")


# Create the directory if it doesn't exist
Path(ARTIFACTS_FOLDER).mkdir(parents=True, exist_ok=True)



#
# Project metadata
#
TIME = pendulum.now(tz='America/New_York').__str__()[:16] \
    .replace('T', '_') \
    .replace(':', '_') \
    .replace('-', '_')
TIME_HYPHENATED = TIME.replace('_','-').replace(" ","--")
PROJECT_NAME = f'{TIME}_cerebros_not-gpt'
meta_trial_number = 42  # irrelevant unless in distributed training

EXPERIMENT_FOLDER = f"{ARTIFACTS_FOLDER}/{TIME_HYPHENATED}-{OWNER}"


keras_models_folder = f"{EXPERIMENT_FOLDER}/keras_models-{meta_trial_number}"
Path(keras_models_folder).mkdir(parents=True, exist_ok=True)

# File paths to save the model and toeknizer to:
MODEL_SAVE_PATH = f"{keras_models_folder}/model_tr_{meta_trial_number}_1_b.keras"
TOKENIZER_SAVE_PATH = f"{keras_models_folder}/tokenizer-tr-{meta_trial_number}-i-b"

# Sanity check
print(f"Model will be save to: '{MODEL_SAVE_PATH}'. Toeknizer is being saved to: '{TOKENIZER_SAVE_PATH}' ")

## Dataset Selection
# Assumes:
# 1. Is a huggingface dataset of the structure ...
# 2. Has a key ['train']['text']
# 3. The key duck types as a List[str]
# 4. The samples tokenize consistent with the MAX_SEQUENCE_LENGTH

DATASET_TO_RUN = str(os.getenv("DATASET_TO_RUN",  "david-thrower/tiny-stories-mini-96-seq-len-50000-samples"))

######################### here ######################

# Samples to use for the neural architecture seaerch stage
PHASE_I_A_SAMPLES_TO_CREATE = int(getenv("PHASE_I_A_SAMPLES_TO_CREATE", "300"))

# Samples to use for the main training stage
PHASE_I_B_SAMPLES_TO_CREATE = int(getenv("PHASE_I_B_SAMPLES_TO_CREATE", "200"))
PHASE_I_B_VAL_SPLIT = float(getenv("PHASE_I_B_VAL_SPLIT", "0.15"))


MLFLOW_PORT = int(os.getenv("MLFLOW_PORT", 7777))

# If you don't want Mlflow, just add `-e MLFLOW_PORT=0` to `docker run`
if MLFLOW_PORT != 0:
    # Enable system metrics
    mlflow.enable_system_metrics_logging()

    # Folder where artifacts will be logged.
    mlflow_artifacts_path = f"{EXPERIMENT_FOLDER}/mlflow-artifacts-{meta_trial_number}"
    Path(mlflow_artifacts_path).mkdir(parents=True, exist_ok=True)

    # Directory for MlFlow database for this experiment:
    mlflow_db_dir= f"{EXPERIMENT_FOLDER}/mlruns-{meta_trial_number}"
    Path(mlflow_db_dir).mkdir(parents=True, exist_ok=True)

    # File name for MlFlow DB file:
    mlflow_db_path = f"{mlflow_db_dir}/mlflow-tr-{meta_trial_number}.db"

    print(f"Logging artifacts to: {mlflow_artifacts_path} and the databse file is at: {mlflow_db_path} ")

    cmd = "".join([
        "mlflow server ",
        "--host 0.0.0.0 ",
        f"--port {str(MLFLOW_PORT)} ",
        f"--default-artifact-root {mlflow_artifacts_path} ",
        f"--backend-store-uri sqlite:///{mlflow_db_path} &"
    ])

    # Debug
    print(f"cmd: {cmd}")
    # / Debug

    answer = subprocess.run(cmd, shell=True)
    time.sleep(30)
    print(answer.stdout)


    # Set up MlFlow experiment

    ds_root_name = DATASET_TO_RUN.split('/')[-1]
    MLFLOW_EXPERIMENT_NAME = f"{TIME_HYPHENATED}--llm-training--{ds_root_name}-" +\
                      f"ia-{PHASE_I_A_SAMPLES_TO_CREATE}-ib-{PHASE_I_B_SAMPLES_TO_CREATE}-a"

    mlflow.set_tracking_uri(uri=f"http://127.0.0.1:{MLFLOW_PORT}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# This is a single head model. It only returns the next token. For this reason,
# a training text sample is "expanded" into several sub - samples at training time.
# The first token becomes the first sub-sample. The second token expressed as a one
# hot representationn of the vocabulary becomes the first sub-label. The first and
# second token become the second sub sample,  and the 3rd token one hot encoded
# becomes the second sub-label. ... until a padding token becomes the label.
#
# In training Stage I-a, this is done in memory for all samples (a streaming dataset object)
# is not yet supported for the NAS algorithm. It is on the road map, with some
# challenges to implementation.
#
# For training Stage I-b: At scale, we will be training on a large number of samples. The
# sample expansion process turns dozens of KBs of text into GBs of tensors. For this reason,
# we must preprocess in batches, otherwise we would need a lot of RAM to hold the preprocessed
# tokens. The parameter 'PHASE_I_B_SAMPLE_EXPANSION_BATCH_SIZE' bleow controls how many to
# process at a time. If you are training at scale, I would raise this to a few hunded,
# depending on your RAM and GPU RAM.


PHASE_I_B_SAMPLE_EXPANSION_BATCH_SIZE = int(getenv("PHASE_I_B_SAMPLE_EXPANSION_BATCH_SIZE", "100"))

# How many tokens to provide before expecting the next token to be predicted.
# It is recommended to keep this as 1. Raising it may reduce RAM pressure at
# the expense of poor accuracy when responding to short prompts.
#
PROMPT_LENGTH = 1

# Text encoding / embedding related constants

# A minimum of 1536 is recommended for any model meant for typical use.
# (MAX_SEQ_LENGTH has a linear and directly proportional to RAM and
# CPU requirement for Cerebros NotGPT. This is a subquadratic NLP
# algo)

MAX_SEQ_LENGTH = int(getenv("MAX_SEQ_LENGTH", "96"))

#
# Cerebros [non-HP-tunable] configurables (Parameters to Optimize continued)
# (Parameters to Stage I-a / Neural Architecture Search stage)
#

# How many permutations of layers to try, basically.
moities_to_try = 1  # ++ Accuracy, linear increase in computation time (Raise this before resorting to raising the next one)

# How many different topologies between the same permutation of
# layers to try, basically (Multiplies the number of models to be tried:
# number of models it will try =  moities_to_try * tries_per_moity)
tries_per_moity = 1  # ++ Modest ++ Accuracy, quadratic increase in computation time

# Main tunable hyperparameters:

POSITIONAL_EMBEDDING_DROPOUT = 0.0819834183890946
activation = 'softplus'

# Directly proportional to the connectivity density between the Input layer
# (output of the text embedding) and the first Dense layer.
predecessor_level_connection_affinity_factor_first = 28.2975136

# Directly propertional to the connectivity density between hidden layers
# and upstream layers.
predecessor_level_connection_affinity_factor_main = 12.45

# Cerebros arranges a grid of Dense layers (Units) on rows (Levels). They connect both
# laterally with Dense layers on the same row as well as verticly with layers on other
# rows. A limit to the number of consecutive connections on the same row.
max_consecutive_lateral_connections = 9

# Basically the density of lateral comnnectiosn approximately
# equals p_lateral_connection * num_lateral_connection_tries_per_unit
p_lateral_connection = 0.628396083507019

num_lateral_connection_tries_per_unit = 24

# The learning rate for Srage I-a
learning_rate = 0.000474

# Number of epochs for Training Stage I-a
epochs = 113

# Batch size for both stages.
batch_size = 20  # When training at scale, use a higher batch size.

# In the Neural architecture search, if set to 1, it will omit the gradient_accumulation_steps.
# It allows '1' to be selected because we want to use it in hyperparameter tuning and not raise
# an error if 1 is called...

gradient_accumulation_steps = 4

# How many hidden "Levels" or rows of Dense layers:
minimum_levels = 2
maximum_levels = 2

# Number of hidden Dense layers per row (Level):
minimum_units_per_level = 2
maximum_units_per_level = 2

# Number of units in each Dense layer:
minimum_neurons_per_unit = 2
maximum_neurons_per_unit = 3

## Training Stage I-b parameters: ###

# LR Scheduler for training stage I-b
INITIAL_LR_STAGE_I_B = 0.000685075852792669

# A fixed number for the initial warmup
WARMUP_EPOCHS_STAGE_I_B = 7
WARMUP_STEPS = 1140  # Generally between 500 and 2000
FIRST_DECAY_STEPS_STAGE_I_B = 1900

phase_i_b_epochs = 97

phase_i_b_gradient_accumulation_steps = 4

phase_i_b_weight_decay = 0.0032205875070735815

## Generation time configurables: ##########

GENERATION_PROMPT_LEN = 25
MAX_NEW_TOKENS = MAX_SEQ_LENGTH - GENERATION_PROMPT_LEN

# Tokenization

tokenizer_checkpoint = "HuggingFaceTB/SmolLM3-3B"  # "HuggingFaceTB/SmolLM2-1.7B-Instruct"
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

EMBEDDING_N = 7
EMBEDDING_DIM = int(EMBEDDING_N * 2)

# Size of the projection layer bet
PROJECTION_N = 1  # Punitive increase of ram, leaving this as 1 until we are running on HPC

##### Attention blocks' and attention mimetic blocks' constants: #######

# --- SingleHeadChunkedAttention Block Constants ---
K_PROJ_CHUNKED = 6
DFF_CHUNKED = 11
DROPOUT_RATE_CHUNKED = 0.05258

# --- MAMBA Block Constants ---
MAMBA_D_STATE = 24
MAMBA_D_CONV = 4
MAMBA_EXPAND = 4
MAMBA_DROPOUT = 0.0765

# --- VoxelAttentionLayer Constants ---
VOXEL_MAX_GRID_SIZE = 7
VOXEL_CA_STEPS = 2
VOXEL_DROPOUT = 0.03270228784722243

# --- Linformer Block Constants (Adjusted for tiny model) ---
LINFORMER_K_PROJ = 11
LINFORMER_DFF = 42
LINFORMER_DROPOUT = 0.21111718785990585
LINFORMER_FFN_DROPOUT = 0.2505778

# --- Adapter Block Constants ---
ADAPTER_DROPOUT = 0.0903144117533307


# Package Parameters for logging

PARAMS = {
    "POSITIONAL_EMBEDDING_DROPOUT": POSITIONAL_EMBEDDING_DROPOUT,
    "activation": activation,
    "predecessor_level_connection_affinity_factor_first": predecessor_level_connection_affinity_factor_first,
    "predecessor_level_connection_affinity_factor_main": predecessor_level_connection_affinity_factor_main,
    "max_consecutive_lateral_connections": max_consecutive_lateral_connections,
    "p_lateral_connection": p_lateral_connection,
    "num_lateral_connection_tries_per_unit": num_lateral_connection_tries_per_unit,
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "minimum_levels": minimum_levels,
    "maximum_levels": maximum_levels,
    "minimum_units_per_level": minimum_units_per_level,  # Fixed
    "maximum_units_per_level": maximum_units_per_level,
    "minimum_neurons_per_unit": minimum_neurons_per_unit,  # Fixed
    "maximum_neurons_per_unit": maximum_neurons_per_unit,
    "INITIAL_LR_STAGE_I_B": 0.0039295722955565125,  # Fixed
    # "WARMUP_STEPS": WARMUP_STEPS,
    # "FIRST_DECAY_STEPS_STAGE_I_B": FIRST_DECAY_STEPS_STAGE_I_B,
    "phase_i_b_epochs": phase_i_b_epochs,
    "phase_i_b_gradient_accumulation_steps": phase_i_b_gradient_accumulation_steps,
    "phase_i_b_weight_decay": phase_i_b_weight_decay,
    "tokenizer_checkpoint": tokenizer_checkpoint,
    "EMBEDDING_N": EMBEDDING_N,
    "EMBEDDING_DIM": EMBEDDING_DIM,
    "K_PROJ_CHUNKED": K_PROJ_CHUNKED,
    "DFF_CHUNKED": DFF_CHUNKED,
    "DROPOUT_RATE_CHUNKED": DROPOUT_RATE_CHUNKED,
    "MAMBA_D_STATE": MAMBA_D_STATE,
    "MAMBA_D_CONV": MAMBA_D_CONV,
    "MAMBA_EXPAND": MAMBA_EXPAND,
    "MAMBA_DROPOUT": MAMBA_DROPOUT,
    "VOXEL_MAX_GRID_SIZE": VOXEL_MAX_GRID_SIZE,
    "VOXEL_CA_STEPS": VOXEL_CA_STEPS,
    "VOXEL_DROPOUT": VOXEL_DROPOUT,
    "LINFORMER_K_PROJ": LINFORMER_K_PROJ,
    "LINFORMER_DFF": LINFORMER_DFF,
    "LINFORMER_DROPOUT": LINFORMER_DROPOUT,
    "LINFORMER_FFN_DROPOUT": LINFORMER_FFN_DROPOUT,
    "ADAPTER_DROPOUT": ADAPTER_DROPOUT,
    "VOCABULARY_SIZE": VOCABULARY_SIZE,
    "PROJECTION_N": PROJECTION_N
}


## Get training data:

ds = load_dataset(DATASET_TO_RUN)
ds_text_column = ds['train']['text']
x_list = list(ds_text_column)

non_instruct_samples = x_list[:PHASE_I_A_SAMPLES_TO_CREATE]
phase_i_b_samples = x_list[PHASE_I_A_SAMPLES_TO_CREATE:PHASE_I_B_SAMPLES_TO_CREATE + PHASE_I_A_SAMPLES_TO_CREATE]
print(
    f"Samples from Tiny Stories consisting of {len(non_instruct_samples)} look like this (sub-sample of 3): {non_instruct_samples[:3]}")


# Preprocess data for Stage I-a training
x, y, vocab_size = prepare_data(data_0=non_instruct_samples, tokenizer_0=tokenizer, max_seq_length=MAX_SEQ_LENGTH,
                                prompt_length=PROMPT_LENGTH)  # Preprocess data for Stage I-a training

X_train, X_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.15, shuffle=False)

print("Debug info")

print(f"Data: {X_train[:50]}")
print(f"Labels: {y_train[:50]}")

print("\n\n\nSanity check for correct preprocessing:")
print(f"Shape of X_train: {len(X_train[0])}")
print(f"Shape of y_train: {len(y_train)}")
print(f"Shape of X_test: {len(X_test[0])}")
print(f"Shape of y_test: {len(y_test)}")

x_train_tf = tf.constant(X_train, tf.int32)
print(x_train_tf)
y_train_tf = tf.constant(y_train, tf.int32)
print(y_train_tf)

x_train_packaged = [x_train_tf]
y_train_packaged = [y_train_tf]

x_test_tf = tf.constant(X_test, tf.int32)
y_test_tf = tf.constant(y_test, tf.int32)

x_test_packaged = [x_test_tf]
y_test_packaged = [y_test_tf]

# Important parameters for the training run
INPUT_SHAPES = [(MAX_SEQ_LENGTH,)]
OUTPUT_SHAPES = [(VOCABULARY_SIZE)]

# Split the phase I-b data set for training and validation:

phase_i_b_train_samples, phase_i_b_val_samples = train_test_split(
    phase_i_b_samples,
    test_size=PHASE_I_B_VAL_SPLIT,
    shuffle=False
)

####### Text embedding base model #####################

# 1. Input Layer
inp = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32)

# 2. Embedding & Initial Processing
embedded = tf.keras.layers.Embedding(
    input_dim=VOCABULARY_SIZE,
    output_dim=EMBEDDING_DIM,
    input_length=MAX_SEQ_LENGTH,
    mask_zero=False,
    name="base_embedding"
)(inp)

# iRoPE Stream
position_embedding = InterleavedRoPE(
    dim=EMBEDDING_DIM,
    max_seq_len=MAX_SEQ_LENGTH,
    name="interleaved_rope"
)(embedded)

# Skip Connection Stream is the `embedded` tensor itself

# Stream Merging: Use a GatedMergeLayer for optimal combination

# initial_merge = GatedMergeLayer(d_model=EMBEDDING_DIM, name="initial_stream_merge")
initial_merge = ManifoldHyperConnect(name="initial_stream_merge_mhc")

x = initial_merge([embedded, position_embedding])
x = tf.keras.layers.Dropout(POSITIONAL_EMBEDDING_DROPOUT, name="initial_dropout")(x)

# 3. Core Attention/Processing Stack (Sequential Order)

# --- Block 1: SingleHeadChunkedAttention ---
x = ChunkedAttentionBlock(
    d_model=EMBEDDING_DIM,
    k_proj=K_PROJ_CHUNKED,
    dff=DFF_CHUNKED,
    dropout_rate=DROPOUT_RATE_CHUNKED,
    name="chunked_attention_block"
)(x)

# --- Block 2: MAMBA Layer ---
x = MambaBlock(
    d_model=EMBEDDING_DIM,
    d_state=MAMBA_D_STATE,
    d_conv=MAMBA_D_CONV,
    expand=MAMBA_EXPAND,
    dropout_rate=MAMBA_DROPOUT,
    name="mamba_block"
)(x)

# --- Block 3: VoxelAttentionLayer ---
x = VoxelBlock(
    d_model=EMBEDDING_DIM,
    dropout_rate=VOXEL_DROPOUT,
    max_voxel_grid_size=VOXEL_MAX_GRID_SIZE,
    ca_steps=VOXEL_CA_STEPS,
    name="voxel_block"
)(x)

# --- Block 4: Linformer Layer ---
x = LinformerBlock(
    d_model=EMBEDDING_DIM,
    k_proj=LINFORMER_K_PROJ,
    dff=LINFORMER_DFF,
    dropout_rate=LINFORMER_DROPOUT,
    ffn_dropout_rate=LINFORMER_FFN_DROPOUT,
    name="linformer_block"
)(x)

# 4. Adapter Block
# Reduces dimension from (BATCH, SEQ_LEN, EMBEDDING_DIM) to (BATCH, SEQ_LEN)
# using the new custom AdapterBlock layer.
flattened_output = AdapterBlock(
    d_model=EMBEDDING_DIM,
    dropout_rate=ADAPTER_DROPOUT,
    name="adapter_block"
)(x)

# 5. Final Model Assembly
cerebros_base_model = tf.keras.Model(
    inputs=inp,
    outputs=flattened_output,  # Output shape is now (BATCH_SIZE, MAX_SEQ_LENGTH)
    name="cerebros_base_model"
)

# Display the model summary to verify the architecture
cerebros_base_model.summary()

# DEBUG <--------------<<<<       ###################

cerebros_base_model.compile()

print("--- Inspecting Model Trainable Weights (Corrected) ---")
all_weights_valid = True
for i, weight in enumerate(cerebros_base_model.trainable_weights):
    # The correct type in Keras 3 is keras.src.backend.Variable
    if not isinstance(weight, tf.keras.Variable):
        print(f"!!! CRITICAL ERROR: Found an unexpected weight type at index {i}.")
        print(f"!!! Actual Type:   {type(weight)}")
        all_weights_valid = False
        break

if all_weights_valid:
    print("!!! SUCCESS: All trainable weights are valid Keras Variables. !!!")
    # To get the parameter count, use model.count_params(), not len()
    print(f"Total trainable params: {cerebros_base_model.count_params():,}")
else:
    print("--- End of Inspection ---")

# end debug


######## Cerebros Neural Architecture Search #######



# Custom metric: Perplexity:
sparse_perplexity_metric = SparsePerplexity()

cerebros_automl = SimpleCerebrosRandomSearch(
    unit_type=DenseUnit,
    input_shapes=INPUT_SHAPES,
    output_shapes=OUTPUT_SHAPES,
    training_data=x_train_packaged,
    labels=y_train_packaged,
    validation_split=0.2,
    direction='minimize',
    metric_to_rank_by="perplexity",
    minimum_levels=minimum_levels,
    maximum_levels=maximum_levels,
    minimum_units_per_level=minimum_units_per_level,
    maximum_units_per_level=maximum_units_per_level,
    minimum_neurons_per_unit=minimum_neurons_per_unit,
    maximum_neurons_per_unit=maximum_neurons_per_unit,
    activation=activation,
    final_activation=None,
    number_of_architecture_moities_to_try=moities_to_try,
    number_of_tries_per_architecture_moity=tries_per_moity,
    minimum_skip_connection_depth=1,
    maximum_skip_connection_depth=7,
    predecessor_level_connection_affinity_factor_first=predecessor_level_connection_affinity_factor_first,
    predecessor_level_connection_affinity_factor_first_rounding_rule='ceil',
    predecessor_level_connection_affinity_factor_main=predecessor_level_connection_affinity_factor_main,
    predecessor_level_connection_affinity_factor_main_rounding_rule='ceil',
    predecessor_level_connection_affinity_factor_decay_main=zero_7_exp_decay,
    seed=8675309,
    max_consecutive_lateral_connections=max_consecutive_lateral_connections,
    gate_after_n_lateral_connections=3,
    gate_activation_function=simple_sigmoid,
    p_lateral_connection=p_lateral_connection,
    p_lateral_connection_decay=zero_95_exp_decay,
    num_lateral_connection_tries_per_unit=num_lateral_connection_tries_per_unit,
    learning_rate=learning_rate,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
             sparse_perplexity_metric,
             # tf.keras.metrics.Accuracy()
             ],
    epochs=epochs,
    project_name=f"{PROJECT_NAME}_meta_{meta_trial_number}",
    model_graphs='model_graphs',
    batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    meta_trial_number=meta_trial_number,
    base_models=[cerebros_base_model],
    train_data_dtype=tf.int32,
    merging_strategy='concatenate')  # "mhc")



ctx = mlflow.start_run() if MLFLOW_PORT else nullcontext()

with ctx: # experiment_id=experiment_id):
    if MLFLOW_PORT:
        mlflow.log_params(PARAMS)

    cerebros_t0 = time.time()
    phase_i_a_result_0 = cerebros_automl.run_random_search()
    phase_i_a_result = float(phase_i_a_result_0)  # Deep copy that survives del() of parent object ...
    cerebros_t1 = time.time()
    cerebros_time_all_models_min = (cerebros_t1 - cerebros_t0) / 60
    models_tried = moities_to_try * tries_per_moity
    cerebros_time_per_model = cerebros_time_all_models_min / models_tried

    if MLFLOW_PORT:
        mlflow.log_metric("perplexity_stage_i_a", phase_i_a_result)

    print(
        f"Cerebros trained {models_tried} models FROM A COLD START in ONLY {cerebros_time_all_models_min} min. Cerebros took only {cerebros_time_per_model} minutes on average per model.")
    print(f'Cerebros best perplexity achieved in Phase I-a is {phase_i_a_result}')

    MODEL_FILE_NAME = "cerebros-foundation-model.keras"

    best_model_found = cerebros_automl.get_best_model(purge_model_storage_files='slate')

    # Create config and generative model
    config = CerebrosNotGPTConfig(
        max_sequence_length=MAX_SEQ_LENGTH,
        padding_token=tokenizer.pad_token_id
    )
    # Instantiate generative model
    generator = CerebrosNotGPT(config, model=best_model_found)

    text = "This is a test ..."

    PADDING_TOKEN = tokenizer.pad_token_id

    input_ids = tokenizer(
        text,
        add_special_tokens=False
    )['input_ids']
    current_tokens = input_ids.copy()

    # Pad (Had been advised by AI when writing .generate() that this
    # should be done manually, not using the tokenizer ...)
    if len(current_tokens) > MAX_SEQ_LENGTH:
        input_tokens = current_tokens[-MAX_SEQ_LENGTH:]
    else:
        padding_needed = MAX_SEQ_LENGTH - len(current_tokens)
        input_tokens = current_tokens + [PADDING_TOKEN] * padding_needed

    # Convert to tensor and get model prediction
    input_tensor = tf.constant([input_tokens], dtype=tf.int32)

    try:
        _ = generator(input_tensor)
        print("✅ Building LLM Model Successful!")
    except Exception as exc:
        error_message = f"❌ Building model returned the error: {exc}"


    # Utility function to generate text from greedy sampling:
    def complete_text_greedy(text: str, max_new_tokens: int = 10) -> str:
        input_ids = tokenizer(
            text,
            add_special_tokens=False
        )['input_ids']

        generated_tokens = generator.generate(
            token_ids=input_ids,  # Just the actual tokens, no padding
            do_sample=False,
            max_new_tokens=max_new_tokens
        )
        generated_text = \
            tokenizer.decode(generated_tokens).replace(text, "")
        return generated_text


    # Utility function to generate text from beam sampling:
    def complete_text_beam(text: str,
                           max_new_tokens: int = 10,
                           temperature: float = 0.75,
                           top_k: int = 75,
                           top_p: float = 0.98,
                           repetition_penalty: float = None,
                           presence_penalty: float = 1.3,
                           frequency_penalty: float = 1.4) -> str:
        input_ids = tokenizer(
            text,
            add_special_tokens=False
        )['input_ids']

        generated_tokens = generator.generate(
            token_ids=input_ids,  # Just the actual tokens, no padding
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            # repetition_penalty=1.2,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty
        )
        generated_text = \
            tokenizer.decode(generated_tokens).replace(text, "")
        return generated_text


    trial_number = meta_trial_number


    def test_text(test_prompt: str, max_new_tokens: int, result_cutoff: float, trial_id: int,
                  test_sample_number: int, result_0: float) -> None:
        """
        If the result_0 < result_cutoff, this will run a matrix of different sampling values and print out the resulting text for human subjective evaluation.

        Parameters:
            - test_prompt: a string to prompt generation
            - max_new_tokens: int, number of tokens to generate unless we generate a stop token.
            - sample_number: Metadata for sample...
            - result_0: Perplexity score from this run
            - result_cutoff: Perplexity score that would be expected to indicate a trial worth running this pn

        """
        if result_0 < result_cutoff:
            generation_param_permutations = [
                # #3
                {
                    'max_new_tokens': max_new_tokens,
                    'temperature': 0.6,
                    'top_k': 75,
                    'top_p': 0.98,
                    'repetition_penalty': None,
                    'presence_penalty': 1.3,
                    'frequency_penalty': 1.4
                },
                # #4
                {
                    'max_new_tokens': max_new_tokens,
                    'temperature': 0.7,
                    'top_k': 75,
                    'top_p': 0.98,
                    'repetition_penalty': None,
                    'presence_penalty': 1.3,
                    'frequency_penalty': 1.4
                },
                # #5
                {
                    'max_new_tokens': max_new_tokens,
                    'temperature': 0.7,
                    'top_k': 75,
                    'top_p': 0.97,
                    'repetition_penalty': None,
                    'presence_penalty': 1.3,
                    'frequency_penalty': 1.4},
                # #6
                {
                    'max_new_tokens': max_new_tokens,
                    'temperature': 0.75,
                    'top_k': 75,
                    'top_p': 0.98,
                    'repetition_penalty': None,
                    'presence_penalty': 1.4,
                    'frequency_penalty': 1.4},
                # #7
                {
                    'max_new_tokens': max_new_tokens,
                    'temperature': 0.7,
                    'top_k': 75,
                    'top_p': 0.98,
                    'repetition_penalty': None,
                    'presence_penalty': 1.4,
                    'frequency_penalty': 1.4},
                # #8
                {
                    'max_new_tokens': max_new_tokens,
                    'temperature': 0.6,
                    'top_k': 75,
                    'top_p': 0.98,
                    'repetition_penalty': None,
                    'presence_penalty': 1.4,
                    'frequency_penalty': 1.4
                },
                {
                    'max_new_tokens': max_new_tokens,
                    'temperature': 0.6,
                    'top_k': 40,
                    'top_p': 0.96,
                    'repetition_penalty': None,
                    'presence_penalty': 1.4,
                    'frequency_penalty': 1.4
                },
                {
                    'max_new_tokens': max_new_tokens,
                    'temperature': 0.7,
                    'top_k': 45,
                    'top_p': 0.97,
                    'repetition_penalty': None,
                    'presence_penalty': 1.4,
                    'frequency_penalty': 1.3
                },  #
                {
                    'max_new_tokens': max_new_tokens,
                    'temperature': 0.6,
                    'top_k': 75,
                    'top_p': 0.99,
                    'repetition_penalty': None,
                    'presence_penalty': 1.4,
                    'frequency_penalty': 1.4
                },
                {
                    'max_new_tokens': max_new_tokens,
                    'temperature': 0.65,
                    'top_k': 75,
                    'top_p': 0.985,
                    'repetition_penalty': None,
                    'presence_penalty': 1.4,
                    'frequency_penalty': 1.4
                },
                {
                    'max_new_tokens': max_new_tokens,
                    'temperature': 0.8,
                    'top_k': 75,
                    'top_p': 0.99,
                    'repetition_penalty': None,
                    'presence_penalty': 0.7,
                    'frequency_penalty': 0.7
                },
                {
                    'max_new_tokens': max_new_tokens,
                    'temperature': 0.8,
                    'top_k': 75,
                    'top_p': 0.99,
                    'repetition_penalty': 1.4,
                    'presence_penalty': None,
                    'frequency_penalty': None
                }
            ]
            # Default cases, no params
            response_1 = complete_text_greedy(text=test_prompt, max_new_tokens=max_new_tokens)
            print(
                f"Trial #: {trial_id} Text Sample #: {test_sample_number} Perplexity: {result_0}  GENERATE SAMPLING PARAMS: Greedy max_new_tokens=10 otherwise - N/A: PROMPT: '{test_prompt}' RESPONSE: '{response_1}'")
            # print(f"Sample {sample_number}: I ask the generator (greedy): {test_prompt}... It responds: '{response_1}'.")
            response_2 = complete_text_beam(text=test_prompt, max_new_tokens=max_new_tokens)
            print(
                f"Trial #: {trial_id} Text Sample #: {test_sample_number} Perplexity: {result_0} GENERATE PARAMS: Beam Default - max_new_tokens = 10, temperature=0.75, top_k=75,  top_p=0.98, repetition_penalty=None, presence_penalty=1.3, frequency_penalty=1.4: PROMPT: '{test_prompt}' RESPONSE: '{response_2}'.")
            # print(f"Sample {sample_number}: I ask the generator (Beam defaults - max_new_tokens: 10,  temperature: 0.75, top_k: 75, top_p: 0.98, repetition_penalty: None, presence_penalty: 1.3, frequency_penalty: 1.4): {test_prompt}... It responds: '{response_2}'.")

            for perm_0 in generation_param_permutations:
                response_0 = complete_text_beam(text=test_prompt,
                                                max_new_tokens=max_new_tokens,
                                                temperature=perm_0['temperature'],
                                                top_k=perm_0['top_k'],
                                                top_p=perm_0['top_p'],
                                                repetition_penalty=perm_0['repetition_penalty'],
                                                presence_penalty=perm_0['presence_penalty'],
                                                frequency_penalty=perm_0['frequency_penalty'])
                print(
                    f"Trial #: {trial_id} Text Sample #: {test_sample_number} Perplexity: {result_0} GENERATE PARAMS: max_new_tokens={perm_0['max_new_tokens']} temperature={perm_0['temperature']}, top_k={perm_0['top_k']}, top_p={perm_0['top_p']}, repetition_penalty={perm_0['repetition_penalty']} presence_penalty={perm_0['presence_penalty']} frequency_penalty{perm_0['frequency_penalty']} PROMPT: '{test_prompt}' RESPONSE: '{response_0}'")


    prompt_samples = [
        "The next day, something unexpected happened. The bird changed into a big, scary",
        "I have an idea, Ben. Let's build a",
        '"Yes, we do," Mia says.'
    ]

    counter = 0
    for sample in prompt_samples:
        test_text(
            test_prompt=sample,
            max_new_tokens=MAX_NEW_TOKENS,
            result_cutoff=999,
            trial_id=meta_trial_number,
            test_sample_number=counter,
            result_0=phase_i_a_result)
        counter += 1

    collect()

    # Continue training the same model with a larger dataset (Phase I-b)

    print(f"Trial: {meta_trial_number} proceeding to phase I-b:")


    # Create the Dataset Generator:
    #     Allows us to process larger data than we can hold in memory


    # Replace your existing class and function with these:
    class SampleExpansionGenerator:
        def __init__(self,
                     raw_text_samples,
                     tokenizer,
                     sample_expansion_batch_size=50,
                     model_batch_size=10,
                     prompt_length_0=PROMPT_LENGTH,
                     max_seq_length=MAX_SEQ_LENGTH,
                     vocabulary_size=VOCABULARY_SIZE):

            self.raw_text_samples = raw_text_samples
            self.tokenizer = tokenizer
            self.sample_expansion_batch_size = sample_expansion_batch_size
            self.model_batch_size = model_batch_size
            self.prompt_length_0 = prompt_length_0
            self.max_seq_length = max_seq_length
            self.vocabulary_size = vocabulary_size
            self.data = []
            self.labels = []
            self.current_index = 0

        def _expand_next_batch(self):
            # If we've already processed all raw samples for this epoch, do nothing.
            if self.current_index >= len(self.raw_text_samples):
                return

            # Determine the next meta-batch
            start_idx = self.current_index
            end_idx = min(start_idx + self.sample_expansion_batch_size, len(self.raw_text_samples))

            batch_samples = self.raw_text_samples[start_idx:end_idx]
            self.current_index = end_idx

            # Run prepare_data on this batch
            input_ids_list, labels_list, _ = prepare_data(
                data_0=batch_samples,
                tokenizer_0=self.tokenizer,
                max_seq_length=self.max_seq_length,
                prompt_length=self.prompt_length_0)

            # Add the new data to our internal queues
            self.data.extend(input_ids_list)
            self.labels.extend(labels_list)

        def __iter__(self):
            # Reset to initial state for new epoch
            self.current_index = 0
            self.data = []
            self.labels = []
            return self

        def __next__(self):
            # If queues are empty, try to expand them from raw samples
            if not self.data:
                self._expand_next_batch()

            # If they are STILL empty after trying to expand, the epoch is over.
            if not self.data:
                raise StopIteration

            # Pop and return one sample
            input_sample = self.data.pop(0)
            label_sample = self.labels.pop(0)

            return ((input_sample,), label_sample)


    # Create the tf.data.Dataset
    def create_dataset(raw_text_samples, tokenizer, sample_expansion_batch_size=50, model_batch_size=10) -> tf.data.Dataset:
        generator_0 = SampleExpansionGenerator(
            raw_text_samples=raw_text_samples,
            tokenizer=tokenizer,
            sample_expansion_batch_size=sample_expansion_batch_size,
            model_batch_size=model_batch_size  # Pass this parameter
        )

        dataset = tf.data.Dataset.from_generator(
            lambda: generator_0,
            # output_signature=(
            #     (tf.TensorSpec(shape=(generator_0.max_seq_length,), dtype=tf.int32),),
            #     # tf.TensorSpec(shape=(generator_0.max_seq_length,), dtype=tf.int32),  # Use generator's parameter
            #     tf.TensorSpec(shape=(generator_0.vocabulary_size,), dtype=tf.float32)  # Use generator's parameter
            # )
            output_signature=(
                (tf.TensorSpec(shape=(generator_0.max_seq_length,), dtype=tf.int32),),  # A tuple containing ONE TensorSpec
                tf.TensorSpec(shape=(), dtype=tf.int32)
                # tf.TensorSpec(shape=(generator_0.vocabulary_size,), dtype=tf.float32)  # A single TensorSpec
            )
        )

        # Batch it
        dataset = dataset.batch(model_batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch for performance
        return dataset


    phase_i_b_train_dataset = \
        create_dataset(
            raw_text_samples=phase_i_b_train_samples,
            tokenizer=tokenizer,
            sample_expansion_batch_size=PHASE_I_B_SAMPLE_EXPANSION_BATCH_SIZE,
            model_batch_size=batch_size)

    phase_i_b_val_dataset = \
        create_dataset(
            raw_text_samples=phase_i_b_val_samples,
            tokenizer=tokenizer,
            sample_expansion_batch_size=PHASE_I_B_SAMPLE_EXPANSION_BATCH_SIZE,
            model_batch_size=batch_size)

    phase_i_b_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    phase_i_b_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    ib_perplexity_key = "perplexity_stage_i_b"
    phase_i_b_perplexity = SparsePerplexity(name=ib_perplexity_key)

    # Create the schedule instance
    lr_scheduler = WarmupCosineDecayRestarts(
        initial_learning_rate=INITIAL_LR_STAGE_I_B,
        warmup_steps=WARMUP_STEPS,
        first_decay_steps=FIRST_DECAY_STEPS_STAGE_I_B,
        t_mul=1.0,  # Keep the cycle length constant (restart every epoch)
        m_mul=0.9,  # Decrease the peak LR by 10% at each restart for finer tuning
        alpha=0.01  # Don't let the LR decay to zero within a cycle
    )

    # Recompile the existing model
    generator.model.compile(
        loss=phase_i_b_loss,
        metrics=[
            phase_i_b_categorical_accuracy,
            phase_i_b_perplexity
        ],
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=lr_scheduler,
            weight_decay=phase_i_b_weight_decay,
            gradient_accumulation_steps=phase_i_b_gradient_accumulation_steps
        ),
        jit_compile=True
    )

    # 2. Define the Early Stopping callback
    # This stops training when validation perplexity stops improving.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=ib_perplexity_key,  # Monitor validation perplexity
        patience=10,  # Number of epochs with no improvement after which training will be stopped.
        verbose=1,
        restore_best_weights=True,  # Restores model weights from the epoch with the best value of the monitored metric.
        mode='min',
        start_from_epoch=40
    )

    callbacks_list = [early_stopping]

    print("Calculating steps per epoch...")

    # Calculate steps for the training dataset
    train_steps = 0
    for _ in phase_i_b_train_dataset:
        train_steps += 1
    print(f"Calculated training steps per epoch: {train_steps}")

    # Calculate steps for the validation dataset
    val_steps = 0
    for _ in phase_i_b_val_dataset:
        val_steps += 1
    print(f"Calculated validation steps: {val_steps}")

    phase_i_b_history = \
        generator.model.fit(
            x=phase_i_b_train_dataset,
            validation_data=phase_i_b_val_dataset,
            epochs=phase_i_b_epochs,
            # steps_per_epoch=train_steps,
            # validation_steps=val_steps # ,
            callbacks=callbacks_list
        )

    phase_i_b_history = \
        pd.DataFrame(phase_i_b_history.history)

    result_phase_i_b = float(phase_i_b_history[ib_perplexity_key].min())
    if MLFLOW_PORT:
        mlflow.log_metric(ib_perplexity_key, result_phase_i_b)

    print("########### Phase I-b Model Checkpoint Generation Samples: ###########")

    counter = 0
    for sample in prompt_samples:
        test_text(
            test_prompt=sample,
            max_new_tokens=MAX_NEW_TOKENS,
            result_cutoff=35,
            trial_id=meta_trial_number,
            test_sample_number=counter,
            result_0=result_phase_i_b)
        counter += 1

    # Serialize stage I-b tokenizer
    # TOKENIZER_SAVE_PATH = f"tokenizer-tr-{meta_trail_number}-stage-i-a"

    tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)
    print(f"Tokenizer saved to {TOKENIZER_SAVE_PATH}")

    # Serialize stage I-b model

    generator.save(MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")

    print(f"🧪 Running serialization test for Stage I-b trial {meta_trial_number}...")
    ser_test_cmd = f"""python3 test_llm_serialization.py "{TOKENIZER_SAVE_PATH}" "{MODEL_SAVE_PATH}" """
    print(f"""Running command: "{ser_test_cmd}" """)
    result = subprocess.run(
        ser_test_cmd,
        capture_output=True,
        shell=True
    )

    if result.returncode == 0:
        print("✅ Serialization test passed.")
        print(str(result.stdout))
    else:
        print("? Serialization test returned some strerr.")
        print("STDERR:", str(result.stderr))
        if result.stdout is not None:
            print(str(result.stdout))
