
import subprocess
import time
from gc import collect


import tensorflow as tf
import pandas as pd
import pendulum


from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from cerebros.units.units import DenseUnit
from cerebros.simplecerebrosrandomsearch.simple_cerebros_random_search\
    import SimpleCerebrosRandomSearch
from cerebrosllmutils.llm_utils import (prepare_data,
                                       InterleavedRoPE,
                                       Perplexity,
                                       CerebrosNotGPTConfig,
                                       CerebrosNotGPT,
                                       WarmupCosineDecayRestarts)
from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component\
    import zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid

from vanilladatasets.web_english_bible import samples as bible   

# It is obvious that anything used for production we would train with a
# dickens of a lot more than 10 and 20 samples... This script can be
# re-used with a scaled up data set.  This script as used here  is
# a vanilla demo and for CICD testing purposes on a 4 CPU / 16 GB RAM
# environment.

# Samples to use for the neural architecture seaerch stage
PHASE_I_A_SAMPLES_TO_CREATE = 10

# Samples to use for the main training stage
PHASE_I_B_SAMPLES_TO_CREATE = 20
PHASE_I_B_VAL_SPLIT = 0.15

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


PHASE_I_B_SAMPLE_EXPANSION_BATCH_SIZE = 10

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

MAX_SEQ_LENGTH = 40

#
# Cerebros [non-HP-tunable] configurables (Parameters to Optimize continued)
# (Parameters to Stage I-a / Neural Architecture Search stage)
#

# How many permutations of layers to try, basically.
moities_to_try = 3 # ++ Accuracy, linear increase in computation time (Raise this before resorting to raising the next one)

# How many different topologies between the same permutation of
# layers to try, basically (Multiplies the number of models to be tried:
# number of models it will try =  moities_to_try * tries_per_moity)
tries_per_moity = 1 # ++ Modest ++ Accuracy, quadratic increase in computation time


# Main tunable hyperparameters:

POSITIONAL_EMBEDDING_DROPOUT = 0.7760921781088126
activation = 'softplus'

# Directly proportional to the connectivity density between the Input layer
# (output of the text embedding) and the first Dense layer.
predecessor_level_connection_affinity_factor_first = 15.94834332206339

# Directly propertional to the connectivity density between hidden layers
# and upstream layers.
predecessor_level_connection_affinity_factor_main = 13.749457115102823

# Cerebros arranges a grid of Dense layers (Units) on rows (Levels). They connect both
# laterally with Dense layers on the same row as well as verticly with layers on other
# rows. A limit to the number of consecutive connections on the same row.
max_consecutive_lateral_connections = 3

# Basically the density of lateral comnnectiosn approximately
# equals p_lateral_connection * num_lateral_connection_tries_per_unit
p_lateral_connection = 0.22164673054843198

num_lateral_connection_tries_per_unit = 20

# The learning rate for Srage I-a
learning_rate = 0.005833579849262622

# Number of epochs for Training Stage I-a
epochs = 54

# Batch size for both stages.
batch_size = 5 # When training at scale, use a higher batch size.

# In the Neural architecture search, if set to 1, it will omit the gradient_accumulation_steps.
# It allows '1' to be selected because we want to use it in hyperparameter tuning and not raise
# an error if 1 is called...

gradient_accumulation_steps = 1

# How many hidden "Levels" or rows of Dense layers:
minimum_levels = 2
maximum_levels = 2

# Number of hidden Dense layers per row (Level):
minimum_units_per_level = 3
maximum_units_per_level = 3

# Number of units in each Dense layer:
minimum_neurons_per_unit = 1
maximum_neurons_per_unit = 3



## Training Stage I-b parameters: ###

# LR Scheduler for training stage I-b
INITIAL_LR_STAGE_I_B = 0.005132833045559803 # 5 * 10 ** -5

# A fixed number for the initial warmup
WARMUP_EPOCHS_STAGE_I_B = 7
WARMUP_STEPS = 760 # Generally between 500 and 2000
FIRST_DECAY_STEPS_STAGE_I_B = 76 # Set this to the number of steps in an epoch given your data.

phase_i_b_epochs = 46

phase_i_b_gradient_accumulation_steps = 2

phase_i_b_weight_decay = 0.08313244799928765

## Generation time configurables: ##########

GENERATION_PROMPT_LEN = 25
MAX_NEW_TOKENS = MAX_SEQ_LENGTH - GENERATION_PROMPT_LEN



# Tokenization 
    
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

# Size of the projection layer bet
PROJECTION_N = 1 # Punitive increase of ram, leaving this as 1 until we are running on HPC

## Get training data:

non_instruct_samples = bible[:PHASE_I_A_SAMPLES_TO_CREATE]
phase_i_b_samples = bible[PHASE_I_A_SAMPLES_TO_CREATE:PHASE_I_B_SAMPLES_TO_CREATE + PHASE_I_A_SAMPLES_TO_CREATE] 
print(f"Samples from KJV bible consisting of {len(non_instruct_samples)} look like this (sub-sample of 3): {non_instruct_samples[:3]}")


# Preprocess data for Stage I-a training
x, y, vocab_size = prepare_data(data_0=non_instruct_samples, tokenizer_0=tokenizer, max_seq_length=MAX_SEQ_LENGTH,
                                prompt_length=PROMPT_LENGTH)        # Preprocess data for Stage I-a training

X_train, X_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.85, shuffle=False)

        
x_train_tf = tf.constant(X_train, tf.int32)
y_train_tf = tf.constant(y_train, tf.float32)
        
x_train_packaged = [x_train_tf]
y_train_packaged = [y_train_tf]
        
x_test_tf = tf.constant(X_test, tf.int32)
y_test_tf = tf.constant(y_test, tf.float32)
        
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

inp = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32)

embedded = tf.keras.layers.Embedding(
    input_dim=VOCABULARY_SIZE,
    output_dim=EMBEDDING_DIM,
    input_length=MAX_SEQ_LENGTH,
    mask_zero=False)(inp)

position_embedding = InterleavedRoPE(
    dim=EMBEDDING_DIM,
    max_seq_len=MAX_SEQ_LENGTH,
    # initializer="uniform",
)(embedded)

# As an FYI, we tried an add layer both with and without
# LayerNorm ... Counterintuitively, adding LayerNorm degraded
# accuracy. Just an FYI for anyone trying to apply conventional
# wisdom, to save you the time ...
x = tf.keras.layers.Concatenate()([embedded, position_embedding])
x = tf.keras.layers.Dropout(POSITIONAL_EMBEDDING_DROPOUT)(x)  # AI suggested 0.4
flattened = tf.keras.layers.Flatten()(x)
projected = tf.keras.layers.Dense(EMBEDDING_DIM * PROJECTION_N)(flattened)  # Dimensionality reduction

cerebros_base_model = tf.keras.Model(
    inputs=inp,
    outputs=projected  # Output enhanced embeddings now
)

######## Cerebros Neural Architecture Search #######

#
# Project metadata
#
TIME = pendulum.now(tz='America/New_York').__str__()[:16] \
    .replace('T', '_') \
    .replace(':', '_') \
    .replace('-', '_')
PROJECT_NAME = f'{TIME}_cerebros_not-gpt'

meta_trial_number = 42  # irrelevant unless in distributed training

# Custom metric: Perplexity:
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
    minimum_levels=minimum_levels,
    maximum_levels=maximum_levels,
    minimum_units_per_level=minimum_units_per_level,
    maximum_units_per_level=maximum_units_per_level,
    minimum_neurons_per_unit=minimum_neurons_per_unit,
    maximum_neurons_per_unit=maximum_neurons_per_unit,
    activation=activation,
    final_activation='softmax',
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
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(),
             perplexity_metric,
             # tf.keras.metrics.Accuracy()
             ],
    epochs=epochs,
    project_name=f"{PROJECT_NAME}_meta_{meta_trial_number}",
    model_graphs='model_graphs',
    batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    meta_trial_number=meta_trial_number,
    base_models=[cerebros_base_model],
    train_data_dtype=tf.int32)  # Changed from tf.string to tf.int32

cerebros_t0 = time.time()
phase_i_a_result_0 = cerebros_automl.run_random_search()
phase_i_a_result = float(phase_i_a_result_0)  # Deep copy that survives del() of parent object ...
cerebros_t1 = time.time()
cerebros_time_all_models_min = (cerebros_t1 - cerebros_t0) / 60
models_tried = moities_to_try * tries_per_moity
cerebros_time_per_model = cerebros_time_all_models_min / models_tried

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



trial_number = 1


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


counter = 0
for sample in prompt_samples:
    test_text(
            test_prompt=sample,
            max_new_tokens=MAX_NEW_TOKENS,
            result_cutoff=15,
            trial_id=trial_number,
            test_sample_number=counter,
            result_0=phase_i_a_result)
    counter += 1


collect()

# Continue training the same model with a larger dataset (Phase I-b)

print(f"Trial: {trial_number} proceeding to phase I-b:")


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
            tf.TensorSpec(shape=(generator_0.vocabulary_size,), dtype=tf.float32)  # A single TensorSpec
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

phase_i_b_loss = tf.keras.losses.CategoricalCrossentropy()
phase_i_b_categorical_accuracy = tf.keras.metrics.CategoricalAccuracy()
phase_i_b_perplexity = Perplexity(name="perplexity_phase_i_b")


# Create the schedule instance
lr_scheduler = WarmupCosineDecayRestarts(
    initial_learning_rate=INITIAL_LR_STAGE_I_B,
    warmup_steps=WARMUP_STEPS,
    first_decay_steps=FIRST_DECAY_STEPS_STAGE_I_B,
    t_mul=1.0, # Keep the cycle length constant (restart every epoch)
    m_mul=0.9, # Decrease the peak LR by 10% at each restart for finer tuning
    alpha=0.01 # Don't let the LR decay to zero within a cycle
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
    monitor='perplexity_phase_i_b',  # Monitor validation perplexity
    patience=35,  # Number of epochs with no improvement after which training will be stopped.
    verbose=1,
    restore_best_weights=True,  # Restores model weights from the epoch with the best value of the monitored metric.
    mode='min'
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
        # validation_steps=val_steps,
        callbacks=callbacks_list
    )

phase_i_b_history = \
    pd.DataFrame(phase_i_b_history.history)


result_phase_i_b = float(phase_i_b_history['perplexity_phase_i_b'].min())


print("########### Phase I-b Model Checkpoint Generation Samples: ###########")


counter = 0
for sample in prompt_samples:
    test_text(
            test_prompt=sample,
            max_new_tokens=MAX_NEW_TOKENS,
            result_cutoff=35,
            trial_id=trial_number,
            test_sample_number=counter,
            result_0=result_phase_i_b)
    counter += 1

# Serialize stage I-b tokenizer
TOKENIZER_SAVE_PATH = f"tokenizer-tr-{trial_number}-stage-i-a"
tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)
print(f"Tokenizer saved to {TOKENIZER_SAVE_PATH}")

# Serialize stage I-b model
MODEL_SAVE_PATH = f"final_phase_ib_model_tr_{trial_number}-stage-i-a.keras"
generator.save(MODEL_SAVE_PATH)
print(f"Final model saved to {MODEL_SAVE_PATH}")

print(f"🧪 Running serialization test for Stage I-b trial {trial_number}...")
result = subprocess.run(
    f"python3 test_llm_serialization.py {TOKENIZER_SAVE_PATH} {MODEL_SAVE_PATH}",
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
