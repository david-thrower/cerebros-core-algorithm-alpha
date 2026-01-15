
import time
from gc import collect

import tensorflow as tf
import pandas as pd
import pendulum

from transformers import AutoTokenizer
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
    # GatedMergeLayer,
    ManifoldHyperConnect,
    ChunkedAttentionBlock,
    MambaBlock,
    VoxelBlock,
    LinformerBlock,
    AdapterBlock,
    CerebrosNotGPTConfig,
    CerebrosNotGPT,
    # WarmupCosineDecayRestarts
)

from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component \
    import zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid

from vanilladatasets.web_english_bible import samples as bible

# It is obvious that anything used for production we would train with a
# dickens of a lot more than 10 and 20 samples... This script can be
# re-used with a scaled up data set.  This script as used here  is
# a vanilla demo and for CICD testing purposes on a 4 CPU / 16 GB RAM
# environment.


# Number of sample to use during neural architecture search:
PHASE_I_A_SAMPLES_TO_CREATE = 300

# Samples to use for the main training stage
PHASE_I_B_SAMPLES_TO_CREATE = 200


# Samples to use for the neural architecture seaerch stage

PHASE_I_B_VAL_SPLIT = 0.15

PHASE_I_B_SAMPLE_EXPANSION_BATCH_SIZE = 100

PROMPT_LENGTH = 1
MAX_SEQ_LENGTH = 96

GENERATION_PROMPT_LEN = 25
MAX_NEW_TOKENS = MAX_SEQ_LENGTH - GENERATION_PROMPT_LEN

# General Model & Training Params
POSITIONAL_EMBEDDING_DROPOUT = 0.1 # trial.suggest_float("POSITIONAL_EMBEDDING_DROPOUT", low=0.0467, high=0.15)
activation = "softplus"
predecessor_level_connection_affinity_factor_first = 28.2975136
predecessor_level_connection_affinity_factor_main = 12.45
max_consecutive_lateral_connections = 9
p_lateral_connection = 0.628396083507019
num_lateral_connection_tries_per_unit = 24
learning_rate = 0.000474
epochs = 113
batch_size = 20
gradient_accumulation_steps = 4
minimum_levels = 2
maximum_levels = 2
minimum_units_per_level = 2
maximum_units_per_level = 2
minimum_neurons_per_unit = 2
maximum_neurons_per_unit = 2 # trial.suggest_int("maximum_neurons_per_unit", low=minimum_neurons_per_unit, high=4)

# LR Scheduler & Stage I-b Params
# WARMUP_STEPS = trial.suggest_int("WARMUP_STEPS", low=500, high=2700)
# FIRST_DECAY_STEPS_STAGE_I_B = trial.suggest_int("FIRST_DECAY_STEPS_STAGE_I_B", low=1000, high=3000)
phase_i_b_epochs =80 # trial.suggest_int("phase_i_b_epochs", low=54, high=100)
phase_i_b_gradient_accumulation_steps = 3 # trial.suggest_int("phase_i_b_gradient_accumulation_steps", low=2, high=4)
phase_i_b_weight_decay = 0.0259 # trial.suggest_float("phase_i_b_weight_decay", low=0.0007455880, high=0.0259, log=True)
STAGE_I_B_LEARN_RATE = 0.000745588 # 0.000685075852792669  # trial.suggest_float("STAGE_I_B_LEARN_RATE", 0.0001, 0.0007)

# Tokenization & Embedding Params
tokenizer_checkpoint = "HuggingFaceTB/SmolLM3-3B"  # Fixed value
EMBEDDING_N = 7  # trial.suggest_int("EMBEDDING_N", low=6, high=7)

# --- Derived Parameters ---
# These depend on other parameters and are calculated after suggestion.
EMBEDDING_DIM = int(EMBEDDING_N * 2)

# Attention Block Constants
K_PROJ_CHUNKED = 6 # 5 is the optimal for 40
DFF_CHUNKED = 11
DROPOUT_RATE_CHUNKED = 0.05258

# Mamba Block Constants
MAMBA_D_STATE = 24
MAMBA_D_CONV = 4
MAMBA_EXPAND = 4
MAMBA_DROPOUT = 0.0765

# VoxelAttentionLayer Constants
VOXEL_MAX_GRID_SIZE = 7
VOXEL_CA_STEPS = 2
VOXEL_DROPOUT = 0.129083395 # trial.suggest_float("VOXEL_DROPOUT", low=0.0203562186, high=0.25)

# Linformer Block Constants
LINFORMER_K_PROJ = 11
LINFORMER_DFF = 42
LINFORMER_DROPOUT = 0.203913939482128 # trial.suggest_float("LINFORMER_DROPOUT", low=0.2039, high=0.3075)
LINFORMER_FFN_DROPOUT = 0.2505778

# Adapter Block Constants
ADAPTER_DROPOUT = 0.0903144117533307

# Tokenization

tokenizer_checkpoint = "HuggingFaceTB/SmolLM3-3B"  # "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

# Step 1: Add special tokens
special_tokens = {
    "additional_special_tokens": ["<prompt>", "</prompt>", "<response>", "</response>"]
}
tokenizer.add_special_tokens(special_tokens)

VOCABULARY_SIZE = len(tokenizer)
PADDING_TOKEN = tokenizer.pad_token_id

PROJECTION_N = 1

moities_to_try = 3
tries_per_moity = 1


# Package the data (WEB Bible Data Set)

non_instruct_samples = bible[:PHASE_I_A_SAMPLES_TO_CREATE]
phase_i_b_samples = bible[PHASE_I_A_SAMPLES_TO_CREATE:PHASE_I_B_SAMPLES_TO_CREATE + PHASE_I_A_SAMPLES_TO_CREATE]
print(
    f"Samples from KJV bible consisting of {len(non_instruct_samples)} look like this (sub-sample of 3): {non_instruct_samples[:3]}")

# Preprocess data for Stage I-a training
x, y, vocab_size = prepare_data(data_0=non_instruct_samples, tokenizer_0=tokenizer,
                                max_seq_length=MAX_SEQ_LENGTH,
                                prompt_length=PROMPT_LENGTH)  # Preprocess data for Stage I-a training

X_train, X_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.85, shuffle=False)

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

####### Define the text embedding base model #####################
# Inputs: [(BATCH_SIZE, MAX_SEQ_LENGTH,)] # Int tokens
# Outputs = (BATCH_SIZE, MAX_SEQ_LENGTH)

# 1. Input Layer
inp = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32)

# 2. Embedding & Initial Processing
embedded = tf.keras.layers.Embedding(
    input_dim=VOCABULARY_SIZE,
    output_dim=EMBEDDING_DIM,
    input_length=MAX_SEQ_LENGTH,
    mask_zero=False,
    name="base_embedding"
)(inp) # Output (BATCH_SIZE, MAX_SEQ_LENGTH, EMBEDDING_DIM)

# iRoPE Stream
position_embedding = InterleavedRoPE(
    dim=EMBEDDING_DIM,
    max_seq_len=MAX_SEQ_LENGTH,
    name="interleaved_rope"
)(embedded) # Output (BATCH_SIZE, MAX_SEQ_LENGTH, EMBEDDING_DIM)

# Skip Connection Stream is the `embedded` tensor itself

# Stream Merging: Use a GatedMergeLayer for optimal combination

# initial_merge = GatedMergeLayer(d_model=EMBEDDING_DIM, name="initial_stream_merge")
initial_merge = ManifoldHyperConnect(name="initial_stream_merge_mhc")

x = initial_merge([embedded, position_embedding]) # Output (BATCH_SIZE, MAX_SEQ_LENGTH, EMBEDDING_DIM)
x = tf.keras.layers.Dropout(POSITIONAL_EMBEDDING_DROPOUT, name="initial_dropout")(x) # Output (BATCH_SIZE, MAX_SEQ_LENGTH, EMBEDDING_DIM)

# 3. Core Attention/Processing Stack (Sequential Order)

# --- Block 1: SingleHeadChunkedAttention ---
x = ChunkedAttentionBlock(
    d_model=EMBEDDING_DIM,
    k_proj=K_PROJ_CHUNKED,
    dff=DFF_CHUNKED,
    dropout_rate=DROPOUT_RATE_CHUNKED,
    name="chunked_attention_block"
)(x) # Output (BATCH_SIZE, MAX_SEQ_LENGTH, EMBEDDING_DIM)

# --- Block 2: MAMBA Layer ---
x = MambaBlock(
    d_model=EMBEDDING_DIM,
    d_state=MAMBA_D_STATE,
    d_conv=MAMBA_D_CONV,
    expand=MAMBA_EXPAND,
    dropout_rate=MAMBA_DROPOUT,
    name="mamba_block"
)(x) # Output (BATCH_SIZE, MAX_SEQ_LENGTH, EMBEDDING_DIM)

# --- Block 3: VoxelAttentionLayer ---
x = VoxelBlock(
    d_model=EMBEDDING_DIM,
    dropout_rate=VOXEL_DROPOUT,
    max_voxel_grid_size=VOXEL_MAX_GRID_SIZE,
    ca_steps=VOXEL_CA_STEPS,
    name="voxel_block"
)(x) # Output (BATCH_SIZE, MAX_SEQ_LENGTH, EMBEDDING_DIM)

# --- Block 4: Linformer Layer ---
x = LinformerBlock(
    d_model=EMBEDDING_DIM,
    k_proj=LINFORMER_K_PROJ,
    dff=LINFORMER_DFF,
    dropout_rate=LINFORMER_DROPOUT,
    ffn_dropout_rate=LINFORMER_FFN_DROPOUT,
    name="linformer_block"
)(x) # Output (BATCH_SIZE, MAX_SEQ_LENGTH, EMBEDDING_DIM)

# 4. Adapter Block
# Reduces dimension from (BATCH, SEQ_LEN, EMBEDDING_DIM) to (BATCH, SEQ_LEN)
# using the new custom AdapterBlock layer.
flattened_output = AdapterBlock(
    d_model=EMBEDDING_DIM,
    dropout_rate=ADAPTER_DROPOUT,
    name="adapter_block"
)(x) # Output (BATCH_SIZE, MAX_SEQ_LENGTH)

# 5. Final Model Assembly
cerebros_base_model = tf.keras.Model(
    inputs=inp,
    outputs=flattened_output,  # Output shape is now (BATCH_SIZE, MAX_SEQ_LENGTH)
    name="cerebros_base_model"
)

# Display the model summary to verify the architecture
cerebros_base_model.summary()

# Metadata and file naming
TIME = pendulum.now(tz='America/New_York').__str__()[:16] \
    .replace('T', '_') \
    .replace(':', '_') \
    .replace('-', '_')
PROJECT_NAME = f'{TIME}_cerebros_not-gpt'

meta_trial_number = 7  # irrelevant unless in distributed training

# Instantiate the custom metric: Perplexity:
sparse_perplexity_metric = SparsePerplexity()


# Instantiate the Neural Architecture Search
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
    merging_strategy='concatenate')

# Pull the trigger on the NAS run
cerebros_t0 = time.time() # Time of the run start
phase_i_a_result_0 = cerebros_automl.run_random_search()
phase_i_a_result = float(phase_i_a_result_0)  # Deep copy of resuly that survives del() of parent object ...
cerebros_t1 = time.time() # Time of NAS run completion
cerebros_time_all_models_min = (cerebros_t1 - cerebros_t0) / 60
models_tried = moities_to_try * tries_per_moity
cerebros_time_per_model = cerebros_time_all_models_min / models_tried
print(f"We trained {models_tried} LLM models in {cerebros_time_all_models_min}, averaging {cerebros_time_per_model} per model")

# Save the model
MODEL_FILE_NAME = f"tr-{meta_trial_number}-cerebros-foundation-model.keras"

# Get the best model and clear temporary storage of disused models: (Returns ft.Keras.Model)
best_model_found = cerebros_automl.get_best_model(purge_model_storage_files='slate')

# Instantiate generative model wrapper (.generate() functionality)

# Create config and generative model
config = CerebrosNotGPTConfig(
    max_sequence_length=MAX_SEQ_LENGTH,
    padding_token=PADDING_TOKEN
)
# Instantiate generative model
generator = CerebrosNotGPT(config, model=best_model_found)


# Dummy call to generator.call() in order to ensure the model's layers are "built".
text = "This is a test ..."

input_ids = tokenizer(
    text,
    add_special_tokens=False
)['input_ids']
current_tokens = input_ids.copy()

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

# A utility function that will generate text with a lot of different generation params,
# so we can see what works best in the case of this unuque model

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
    "I saw the sun and it was as shining on the",
    "And God said, Let there be light: and there ",
    "In the beginning God created the heavens"
]

# Smoke test a few generations

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

