import optuna
import os
import mlflow
from datetime import datetime
import subprocess
from warnings import warn

MLFLOW_PORT = 7777

answer = subprocess.run(f"mlflow server --host 127.0.0.1 --port {MLFLOW_PORT} &",
   shell=True,
)
print(answer.stdout)


EXPERIMENT_ITERATION = "0001"
EXPERIMENT_NAME = "more-optimizations-br-254-single-machine"
DATA_SET_NAME = "WEB-Bible-Genesis-40-context-681-SPL"
EXPERIMENT_NAME = f"{EXPERIMENT_NAME}-{DATA_SET_NAME}-{EXPERIMENT_ITERATION}-a"

N_TRIALS = 10


mlflow.set_tracking_uri(uri=f"http://127.0.0.1:{MLFLOW_PORT}")
mlflow.set_experiment(EXPERIMENT_NAME)


# Optuna Storage Essentials
# Use JournalFileStorage to ensure concurrency safety

storage_file = f"./optuna_{EXPERIMENT_NAME}.log"
journal_backend = optuna.storages.journal.JournalFileBackend(storage_file)
optuna_storage = optuna.storages.JournalStorage(journal_backend)


def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna hyperparameter optimization
    Returns the validation loss or metric to minimize
    """
    
    import tensorflow as tf
    # import tensorflow_text
    # from keras_nlp.models import GPT2Tokenizer, GPT2Preprocessor, GPT2Backbone
    # from keras_nlp.layers import PositionEmbedding
    from transformers import AutoTokenizer
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    # from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Flatten
    import pandas as pd
    import numpy as np
    from cerebros.simplecerebrosrandomsearch.simple_cerebros_random_search\
        import SimpleCerebrosRandomSearch
    from cerebrosllmutils.llm_utils import prepare_data, \
                                           InterleavedRoPE, \
                                           Perplexity, \
                                           CerebrosNotGPTConfig, \
                                           CerebrosNotGPT
    import pendulum
    from cerebros.units.units import DenseUnit
    from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component\
        import zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid
    from ast import literal_eval
    import time
    from gc import collect
    # from os.path import getsize
    import re

    ### Non - HP tuning parameters (Optimize to RAM / CPU / GPU capacity)
    
    # Number of text samples to create: # Number of text samples (of approximately max_seq_len) to create 
    # Raises RAM in a linear fashion    
   
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
    RESULT_CUTOFF = 50 # 20 # 100 # <---<< In production 100 # Only print out verbose text samples when perplexity is < RESULT_CUTOFF

    if GENERATION_PROMPT_LEN + MAX_NEW_TOKENS > MAX_SEQ_LENGTH:
       raise ValueError("Sequence length overflow: Generated text length (GENERATION_PROMPT_LEN + MAX_NEW_TOKENS) "
                        "should be less than or equal to MAX_SEQ_LENGTH.")

    ##### HP Tuning Parameters: ######### (Parameters to be optimized by TPE or SOBOL) 

    
    # Sample hyperparameters directly
    # Begin MLflow trial run (nested inside parent run if any)


    POSITIONAL_EMBEDDING_DROPOUT = trial.suggest_float('POSITIONAL_EMBEDDING_DROPOUT', 0.72, 0.8)

    activation = trial.suggest_categorical('activation', ['relu', 'gelu', 'swish', 'softsign', 'softplus'])

    predecessor_level_connection_affinity_factor_first = trial.suggest_float('predecessor_level_connection_affinity_factor_first', 10.0, 30.0)

    predecessor_level_connection_affinity_factor_main = trial.suggest_float('predecessor_level_connection_affinity_factor_main', 10.0, 25.0)

    max_consecutive_lateral_connections = trial.suggest_int('max_consecutive_lateral_connections', 2, 7)

    p_lateral_connection = trial.suggest_float('p_lateral_connection', 0.12, 0.35)

    num_lateral_connection_tries_per_unit = trial.suggest_int('num_lateral_connection_tries_per_unit', 10, 35)
    
    learning_rate = trial.suggest_float('learning_rate', 0.003, 0.006) # log=True)
    # phase_i_b_learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.006)

    
    epochs = trial.suggest_int('epochs', 30, 75)
    phase_i_b_epochs =  trial.suggest_int('phase_i_b_epochs', 40, 60)
    
    batch_size = 5 # trial.suggest_int('batch_size', 5, 10)

    gradient_accumulation_steps = trial.suggest_int('gradient_accumulation_steps', 1, 7)
    
    
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
    
    # Prepare a record of params:
    # Log sampled hyperparameters to MLflow
    params = {"PHASE_I_A_SAMPLES_TO_CREATE":PHASE_I_A_SAMPLES_TO_CREATE,
              'PHASE_I_B_SAMPLES_TO_CREATE': PHASE_I_B_SAMPLES_TO_CREATE,
              "PROMPT_LENGTH":PROMPT_LENGTH,
              "MAX_SEQ_LENGTH":MAX_SEQ_LENGTH,
              "POSITIONAL_EMBEDDING_DROPOUT":POSITIONAL_EMBEDDING_DROPOUT,
              "activation":activation,
              "predecessor_level_connection_affinity_factor_first":predecessor_level_connection_affinity_factor_first,
              "predecessor_level_connection_affinity_factor_main":predecessor_level_connection_affinity_factor_main,
              "max_consecutive_lateral_connections": max_consecutive_lateral_connections,
              "p_lateral_connection":p_lateral_connection,
              "num_lateral_connection_tries_per_unit": num_lateral_connection_tries_per_unit,
              "learning_rate":learning_rate,
              "epochs":epochs,
              "phase_i_b_epochs": phase_i_b_epochs,
              "batch_size":batch_size,
              "gradient_accumulation_steps":gradient_accumulation_steps,
              "minimum_levels":minimum_levels,
              "maximum_levels":maximum_levels,
              "minimum_units_per_level":minimum_units_per_level,
              "maximum_units_per_level":maximum_units_per_level,
              "minimum_neurons_per_unit":minimum_neurons_per_unit,
              "maximum_neurons_per_unit":maximum_neurons_per_unit,
              "VOCABULARY_SIZE":VOCABULARY_SIZE,
              "EMBEDDING_DIM":EMBEDDING_DIM,
              "PROJECTION_N":PROJECTION_N
             }

    run_name = f"trial_{trial.number}"
    trial_start_time = datetime.utcnow()
   
    tags = {"phase": "poc", "script": os.path.basename(__file__), "trial_number": str(trial.number), "Start_time": str(trial_start_time)}

    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        # Log the hyperparameters
        mlflow.log_params(params)

        
        ############    Data Preprocessing:     ###################
        
        
        ## Import data
        
        from vanilladatasets.web_english_bible import samples as bible
        
        non_instruct_samples = bible[:PHASE_I_A_SAMPLES_TO_CREATE]
        phase_i_b_samples = bible[PHASE_I_A_SAMPLES_TO_CREATE:PHASE_I_B_SAMPLES_TO_CREATE + PHASE_I_A_SAMPLES_TO_CREATE] 
        print(f"Samples from KJV bible consisting of {len(non_instruct_samples)} look like this (sub-sample of 3): {non_instruct_samples[:3]}")
        
        # Split phase_i_b_samples into train and validation sets
        phase_i_b_train_samples, phase_i_b_val_samples = train_test_split(
            phase_i_b_samples, 
            test_size=PHASE_I_B_VAL_SPLIT, 
            shuffle=False
        )
        
        # Preprocess data for Stage I-a training
        x, y, vocab_size =  prepare_data(data_0=non_instruct_samples, tokenizer_0=tokenizer, max_seq_length=MAX_SEQ_LENGTH, prompt_length = PROMPT_LENGTH)

        # QC check 
        print("Input IDs shape:", len(x), "x", len(x[0]) if x else 0)
        print("Labels shape:", len(y), "x", len(y[0]) if y else 0)
        print("Vocabulary size:", vocab_size)
        print("First few samples generated:", len(x))

        del(bible)
        collect()
        
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
        # LayerNorm ... It degraded accuracy
        # Just an FYI for anyone trying to apply conventional wisdom
        # to save you the time ...
        x = tf.keras.layers.Concatenate()([embedded, position_embedding])
        x = tf.keras.layers.Dropout(POSITIONAL_EMBEDDING_DROPOUT)(x)  # AI suggested 0.4 
        flattened = tf.keras.layers.Flatten()(x)
        projected = tf.keras.layers.Dense(EMBEDDING_DIM * PROJECTION_N)(flattened) # Dimensionality reduction
        
        cerebros_base_model = tf.keras.Model(
            inputs=inp,
            outputs=projected  # Output enhanced embeddings now
        )
        
        
        ######## Cerebros Neural Architecture Search #######
        
        #
        # Project metadata
        #
        TIME = pendulum.now(tz='America/New_York').__str__()[:16]\
            .replace('T', '_')\
            .replace(':', '_')\
            .replace('-', '_')
        PROJECT_NAME = f'{TIME}_cerebros_not-gpt'
        
        meta_trial_number = 42 # irrelevant unless in distributed training
        
        
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
        phase_i_a_result = float(phase_i_a_result_0) # Deep copy that survives del() of parent object ...
        cerebros_t1 = time.time()
        cerebros_time_all_models_min = (cerebros_t1 - cerebros_t0) / 60
        models_tried = moities_to_try  * tries_per_moity
        cerebros_time_per_model = cerebros_time_all_models_min / models_tried
        
        
        
        print(f"Cerebros trained {models_tried} models FROM A COLD START in ONLY {cerebros_time_all_models_min} min. Cerebros took only {cerebros_time_per_model} minutes on average per model.")        
        print(f'Cerebros best perplexity achieved in Phase I-a is {phase_i_a_result}')
        # Log the metric to MlFLow
        mlflow.log_metric("phase-i-a-perplexity", phase_i_a_result, step=trial.number)

        """### Testing the best model found"""
        
        MODEL_FILE_NAME = "cerebros-foundation-model.keras"
        
        best_model_found = cerebros_automl.get_best_model(purge_model_storage_files='slate')
        # mlflow.keras.log_model(best_model_found, artifact_path="base")
        # best_model_found.save(MODEL_FILE_NAME)

        
        # file_size_bytes = getsize(MODEL_FILE_NAME)
        # print(f"Model size on disk: {file_size_bytes / (1024*1024):.2f} MB")
        
        # reconstituted_model = tf.keras.models.load_model(MODEL_FILE_NAME)
        
        
        # Create config and generative model
        config = CerebrosNotGPTConfig(
            max_sequence_length=MAX_SEQ_LENGTH,
            padding_token=tokenizer.pad_token_id
        )
        generator = CerebrosNotGPT(config, model=best_model_found)


        # Need to explicitly call the LLM directly to build it,
        # otherwise tf.keras.Model.save() will be unsuccessful.
        # (the weights will not be preserved). Strangely, the
        # model being called internally by .generate() before
        # calling tf.keras.Model.save() does not accomplish
        # this 🤷‍♀️.

        text = "This is a test ..."

        PADDING_TOKEN = tokenizer.pad_token_id

        input_ids = tokenizer(
                text,
                add_special_tokens=False
            )['input_ids']
        current_tokens = input_ids.copy()

        # Pad (Had been advised by AI when writing .generate() that this
        # should be done manually, not using the tokenizer ...)
        if len(current_tokens) >  MAX_SEQ_LENGTH:
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

        
        # mlflow.keras.log_model(generator, artifact_path="generator")


        # Utility function to generate text from greedy sampling:
        def complete_text_greedy(text: str, max_new_tokens:int=10) -> str:
            input_ids = tokenizer(
                text,
                add_special_tokens=False
            )['input_ids']
        
            generated_tokens = generator.generate(
                token_ids=input_ids,  # Just the actual tokens, no padding
                do_sample=False,
                max_new_tokens=max_new_tokens
            )
            generated_text =\
                    tokenizer.decode(generated_tokens).replace(text, "")
            return generated_text

        # Utility function to generate text from beam sampling:
        def complete_text_beam(text: str,
                               max_new_tokens: int=10, 
                               temperature: float=0.75, 
                               top_k: int=75, 
                               top_p: float=0.98, 
                               repetition_penalty: float=None, 
                               presence_penalty: float=1.3, 
                               frequency_penalty: float=1.4) -> str:

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
                presence_penalty= presence_penalty,
                frequency_penalty=frequency_penalty
            )
            generated_text =\
                    tokenizer.decode(generated_tokens).replace(text, "")
            return generated_text


        trial_number = int(trial.number)
        def test_text(test_prompt: str, max_new_tokens: int, sample_number: int, result_cutoff: float, trial_id: int, test_sample_number: int, result_0: float) -> None:
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
                        }, # 
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
                response_1 = response = complete_text_greedy(text=test_prompt, max_new_tokens=max_new_tokens)
                print(f"Trial #: {trial_id} Text Sample #: {test_sample_number} Perplexity: {result_0}  GENERATE SAMPLING PARAMS: Greedy max_new_tokens=10 otherwise - N/A: PROMPT: '{test_prompt}' RESPONSE: '{response_1}'")
                # print(f"Sample {sample_number}: I ask the generator (greedy): {test_prompt}... It responds: '{response_1}'.")
                response_2 = complete_text_beam(text=test_prompt, max_new_tokens=max_new_tokens)
                print(f"Trial #: {trial_id} Text Sample #: {test_sample_number} Perplexity: {result_0} GENERATE PARAMS: Beam Default - max_new_tokens = 10, temperature=0.75, top_k=75,  top_p=0.98, repetition_penalty=None, presence_penalty=1.3, frequency_penalty=1.4: PROMPT: '{test_prompt}' RESPONSE: '{response_2}'.")
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
                    print(f"Trial #: {trial_id} Text Sample #: {test_sample_number} Perplexity: {result_0} GENERATE PARAMS: max_new_tokens={perm_0['max_new_tokens']} temperature={perm_0['temperature']}, top_k={perm_0['top_k']}, top_p={perm_0['top_p']}, repetition_penalty={perm_0['repetition_penalty']} presence_penalty={perm_0['presence_penalty']} frequency_penalty{perm_0['frequency_penalty']} PROMPT: '{test_prompt}' RESPONSE: '{response_0}'")

       # Sample prompts to test:

        print("########### Phase I-a Model Checkpoint Generation Samples: ###########")
       
        prompt_samples = [
                "I saw the sun and it was as shining on the",
                # "And God said to Moses:",
                # "In the beginning God created the ",
                # "And the earth was without form, and",
                "And God said, Let there be light: and there ",
                # "Shall we all go to the river and"
                # "Try to",
                # "You must go and",
                "In the beginning God created the heavens",
                # "The earth was formless and empty, with darkness over",
                # "God called the light 'day' and the darkness 'night,' marking evening and morning",
                # "God called the expanse 'sky,' and there was",
                # "The earth brought forth grass, seed-bearing"
        ]


        counter = 0
        for sample in prompt_samples:
            test_text(
                   test_prompt=sample,
                   max_new_tokens=MAX_NEW_TOKENS,
                   sample_number=counter,
                   result_cutoff=RESULT_CUTOFF,
                   trial_id=trial_number,
                   test_sample_number=counter,
                   result_0=phase_i_a_result)
            counter += 1
            

        # del(best_model_found)
        # del(generator)
        collect()

        print(f"Trial: {trial_number} proceeding to phase I-b:")


        # Create the Dataset Generaror:
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
                # Determine the next meta-batch
                start_idx = self.current_index
                end_idx = min(start_idx + self.sample_expansion_batch_size, len(self.raw_text_samples))
                collect()
              
                if start_idx >= end_idx:
                    raise StopIteration("No more raw samples to process.")
             
                batch_samples = self.raw_text_samples[start_idx:end_idx]
                self.current_index = end_idx
      
                # Run prepare_data on this batch - use the instance parameters
                input_ids_list, labels_list, _ = prepare_data(
                    data_0=batch_samples,
                    tokenizer_0=self.tokenizer,
                    max_seq_length=self.max_seq_length,
                    prompt_length=self.prompt_length_0)
              
                # Assign to internal queues
                self.data = input_ids_list
                self.labels = labels_list
      
            def __iter__(self):
                # Reset to initial state for new epoch
                self.current_index = 0
                self.data = []
                self.labels = []
                return self
      
            def __next__(self):
                # Check for mismatched state
                if (len(self.data) == 0) != (len(self.labels) == 0):
                    raise ValueError("Data and labels queues are out of sync.")
      
                # If queues are empty, expand next batch
                if len(self.data) == 0:
                    self._expand_next_batch()
      
                # Pop and return one sample
                input_sample = self.data.pop(0)
                label_sample = self.labels.pop(0)
      
                return (input_sample, label_sample)
      
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
                output_signature=(
                    tf.TensorSpec(shape=(generator_0.max_seq_length,), dtype=tf.int32),  # Use generator's parameter
                    tf.TensorSpec(shape=(generator_0.vocabulary_size,), dtype=tf.float32)  # Use generator's parameter
                )
            )
          
            # Batch it
            dataset = dataset.batch(model_batch_size)
            return dataset
        
        phase_i_b_train_dataset =\
           create_dataset(
              raw_text_samples=phase_i_b_train_samples,
              tokenizer=tokenizer,
              sample_expansion_batch_size=PHASE_I_B_SAMPLE_EXPANSION_BATCH_SIZE,
              model_batch_size=batch_size)

        
        phase_i_b_val_dataset =\
            create_dataset(
               raw_text_samples=phase_i_b_val_samples,
               tokenizer=tokenizer,
               sample_expansion_batch_size=PHASE_I_B_SAMPLE_EXPANSION_BATCH_SIZE,
               model_batch_size=batch_size)


        phase_i_b_history =\
                generator.model.fit(
                   x=phase_i_b_train_dataset,
                   validation_data=phase_i_b_val_dataset,
                   epochs=phase_i_b_epochs)


        phase_i_b_history =\
               pd.DataFrame(phase_i_b_history.history)
        # To Do: Find best metric: Reference: cerebros/simplecerebrosrandomsearch/simple_cerebros_random_search.py: Line ~ 590
        #  = phase_i_b_history.
        result_phase_i_b = float(phase_i_b_history['perplexity'].min())
        mlflow.log_metric("phase_i_b-perplexity", result_phase_i_b, step=trial_number)

        print("########### Phase I-b Model Checkpoint Generation Samples: ###########")
       
        # Text samples after Phase I-b training
        counter = 0
        for sample in prompt_samples:
            test_text(
                   test_prompt=sample,
                   max_new_tokens=MAX_NEW_TOKENS,
                   sample_number=counter,
                   result_cutoff=RESULT_CUTOFF,
                   trial_id=trial_number,
                   test_sample_number=counter,
                   result_0=result_phase_i_b)
            counter += 1


        TOKENIZER_SAVE_PATH = f"tokenizer-tr-{trial_number}-a"
        tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)
        print(f"Tokenizer saved to {TOKENIZER_SAVE_PATH}")
        
       
        MODEL_SAVE_PATH = f"final_phase_ib_model_tr_{trial_number}-a.keras"
        generator.save(MODEL_SAVE_PATH)
        print(f"Final model saved to {MODEL_SAVE_PATH}")

        # Test inter - module serialization

        print(f"🧪 Running serialization test for trial {trial_number}...")
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

        # Return the final result to Optuna
        return result_phase_i_b


def main():
    n_trials = N_TRIALS
    sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=5)
    study = optuna.create_study(direction="minimize", sampler=sampler, storage=optuna_storage)
    study.optimize(objective, n_trials=N_TRIALS)
    print('Best trial:')
    best_trial = study.best_trial
    print('  Value: ', best_trial.value)
    print('  Params: ')
    for key, value in best_trial.params.items():
        print(f'    {key}: {value}')

if __name__ == '__main__':
    main()


