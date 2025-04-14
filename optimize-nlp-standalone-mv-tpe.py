
# Objective
def objective(trial):


    import tensorflow as tf
    import tensorflow_text
    from keras_nlp.models import GPT2Tokenizer, GPT2Preprocessor, GPT2Backbone
    from keras_nlp.layers import PositionEmbedding
    from transformers import AutoTokenizer
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Flatten
    import pandas as pd
    import numpy as np
    from cerebros.simplecerebrosrandomsearch.simple_cerebros_random_search\
        import SimpleCerebrosRandomSearch
    import pendulum
    from cerebros.units.units import DenseUnit
    from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component\
        import zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid
    from ast import literal_eval
    import time

    # Hyperparameters:
    
    # embedding_n = trial.suggest_categorical(
    #         name="embedding_n",choices=[int(n)
    #                                     for n in
    #                                     np.arange(10,18,2).tolist()])
    embedding_n = trial.suggest_int(low=10,high=18, ster=1)
    activation = trial.suggest_categorical(
            name="activation",
            choices=["relu", "gelu", "elu"])
    predecessor_level_connection_affinity_factor_first =\
            trial.suggest_float(
                    name="predecessor_level_connection_affinity_factor_first",
                    low=0.1,
                    high=50.0,
                    step=0.1)
    predecessor_level_connection_affinity_factor_main =\
            trial.suggest_float(
                    name="predecessor_level_connection_affinity_factor_main",
                    low=0.1,
                    high=50.0,
                    step=0.1)
    max_consecutive_lateral_connections =\
            trial.suggest_int(
                    name="max_consecutive_lateral_connections",
                    low=1,
                    high=50)
    p_lateral_connection =\
            trial.suggest_float(
                    name="p_lateral_connection",
                    low=0.1,
                    high=50.0,
                    step=0.1)
    num_lateral_connection_tries_per_unit =\
            trial.suggest_int(
                    name="num_lateral_connection_tries_per_unit",
                    low=1,
                    high=50)
    learning_rate =\
            trial.suggest_float(
                    name="learning_rate",
                    low=10**-6,
                    high=0.7,
                    log=True)
    epochs =\
            trial.suggest_int(
                    name="epochs",
                    low=1,
                    high=25)
    batch_size =\
            trial.suggest_int(
                    name="batch_size", 
                    low=1, 
                    high=35)
    dropout =\
            trial.suggest_float(
                    name="dropout",
                    low=0.05,
                    high=0.95,
                    step=0.05)
    maximum_units_per_level =\
            trial.suggest_int(
                    name="maximum_units_per_level",
                    low=5,
                    high=10)
    maximum_neurons_per_unit =\
            trial.suggest_int(
                    name="maximum_neurons_per_unit",
                    low=1,
                    high=9)
    # 
    temperature =\
            trial.suggest_int(
                    name="temperature",
                    low=10 ** 4,
                    high=10 ** 6,
                    log=True)


    #
    # Load the email data
    #
    df = pd.read_csv("Phishing_Email.csv")
    #
    # Get the rows where 'Email Text' is a string, remove everything else
    #
    df = df[df['Email Text'].apply(lambda x: isinstance(x, str))]
    #
    # Reset the index
    #
    df.reset_index(drop=True, inplace=True)

    #
    # Binary label for email type: positive type is "phishing"
    #
    label_mapping = {"Safe Email": 0, "Phishing Email": 1}
    df["Binary Label"] = df["Email Type"].map(label_mapping)
    #
    # Data and labels ready
    #
    X = df["Email Text"].to_numpy()
    y = df["Binary Label"].to_numpy()
    #
    # Shuffle the data
    #
    X, y = shuffle(X, y)

    # Train / test split : we give 85% of the data for *testing*
    X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.85, shuffle=False)

    #
    # Tensors for training data and labels
    #

    # Training data for baseline model
    baseline_train_x = tf.constant(X_train, dtype=tf.string)
    baseline_train_y = tf.constant(y_train, dtype=tf.int8)

    # Packaged for Cerebros (multimodal, takes inputs as a list)
    training_x   = [baseline_train_x]
    train_labels = [baseline_train_y]

    #
    # Input and output shapes
    #
    INPUT_SHAPES  = [()]
    OUTPUT_SHAPES = [1]

    ### Cerebros model:



    class NewTokenizerLayer(tf.keras.layers.Layer):
        def __init__(self, max_seq_length, tokenizer_checkpoint, **kwargs):
            super().__init__(**kwargs)
            self.max_seq_length = max_seq_length
            self.tokenizer_checkpoint = tokenizer_checkpoint
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
        
            # Ensure tokenizer has a padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        def call(self, inputs):
            def tokenize_py_fn(inputs):
                # Convert TensorFlow bytes to Python strings
                texts = [text.decode('utf-8') for text in inputs.numpy()]

                # Tokenize with Hugging Face tokenizer
                tokenized = self.tokenizer(
                        texts,
                        max_length=self.max_seq_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='tf'
                )
                return tokenized['input_ids'].numpy()

            # Wrap Python function in TensorFlow operation
            input_ids = tf.py_function(
                    tokenize_py_fn,
                    [inputs],
                    Tout=tf.int32
            )
        
            # Set shape for downstream layers
            batch_size = tf.shape(inputs)[0]
            input_ids.set_shape([None, self.max_seq_length])

            return input_ids

        def get_config(self):
            config = super().get_config()
            config.update({
                    'max_seq_length': self.max_seq_length,
                    'tokenizer_checkpoint': self.tokenizer_checkpoint
            })
            return config

        @classmethod
        def from_config(cls, config):
            return cls(
                    max_seq_length=config['max_seq_length'],
                    tokenizer_checkpoint=config['tokenizer_checkpoint']
            )




    # --- Updated RotaryEmbedding ---
    class RotaryEmbedding(tf.keras.layers.Layer):
        def __init__(self, dim, max_seq_len=1024, temperature=10000.0, **kwargs):
            super().__init__(**kwargs)
            self.dim = dim
            # Ensure dim is even right at initialization
            if self.dim % 2 != 0:
                raise ValueError(f"Embedding dimension `dim` ({self.dim}) must be even for RotaryEmbedding.")
            self.max_seq_len = max_seq_len
            self.temperature = temperature
            # *** No calculation or storage of inv_freq here or in build ***

        def build(self, input_shape):
            # Build should primarily be for creating trainable weights, which we don't have.
            # Call super().build() for Keras compatibility.
            super().build(input_shape)

        def call(self, x): # Removed seq_len argument, calculate from x
            shape = tf.shape(x)
            batch_size = shape[0]
            actual_seq_len = shape[1]

            # *** Calculate inv_freq inside call ***
            inv_freq_base = tf.range(0, self.dim, 2, dtype=tf.float32)
            inv_freq = 1.0 / (self.temperature ** (inv_freq_base / self.dim))
            # Ensure inv_freq has the correct shape [dim/2]
            inv_freq = tf.cast(inv_freq, dtype=x.dtype) # Match dtype early

            # Use actual_seq_len for calculations
            position = tf.range(actual_seq_len, dtype=x.dtype) # Match dtype

            # Calculate sinusoid input using einsum or broadcasting
            # Einsum approach: Ensure correct dimensions [seq_len, dim/2]
            sinusoid_inp = tf.einsum("i,j->ij", position, inv_freq)

            # Calculate sin and cos based on the actual sequence length
            sin = tf.sin(sinusoid_inp)
            cos = tf.cos(sinusoid_inp)

            # Repeat sin/cos for interleaving: [a, b] -> [a, a, b, b]
            # Result needs shape [actual_seq_len, dim]
            sin = tf.repeat(sin, 2, axis=-1)
            cos = tf.repeat(cos, 2, axis=-1)

            # Expand dims for batch and tile
            # Output shape needs to be [batch_size, actual_seq_len, dim]
            # Add batch dimension: [1, actual_seq_len, dim]
            sin = tf.expand_dims(sin, axis=0)
            cos = tf.expand_dims(cos, axis=0)

            # Tile to match the batch size: [batch_size, actual_seq_len, dim]
            sin = tf.tile(sin, [batch_size, 1, 1])
            cos = tf.tile(cos, [batch_size, 1, 1])

            # Casting to x.dtype was already done for inv_freq, sin/cos will inherit
            # sin = tf.cast(sin, x.dtype) # Already done via calculation chain
            # cos = tf.cast(cos, x.dtype) # Already done via calculation chain

            # Return sin and cos needed by InterleavedRoPE
            return sin, cos

        def get_config(self):
            config = super().get_config()
            config.update({
                    "dim": self.dim,
                    "max_seq_len": self.max_seq_len,
                    "temperature": self.temperature,
            })
            return config

        @classmethod
        def from_config(cls, config):
            return cls(**config)





    def split_alternate(x):
        shape = tf.shape(x)
        x = tf.reshape(x, [shape[0], shape[1], shape[2] // 2, 2])
        x = tf.transpose(x, [0, 1, 3, 2])
        x = tf.reshape(x, [shape[0], shape[1], -1])
        return x


    def rotate_half(x):
        x = split_alternate(x)
        d = tf.shape(x)[-1]
        rotated_x = tf.concat([-x[..., d//2:], x[..., :d//2]], axis=-1)
        return tf.reshape(rotated_x, tf.shape(x))


    def apply_rotary_pos_emb(x, sin, cos):
        cos = tf.reshape(cos, [tf.shape(cos)[0], tf.shape(cos)[1], -1])
        sin = tf.reshape(sin, [tf.shape(sin)[0], tf.shape(sin)[1], -1])
        x_rotated = x * cos + rotate_half(x) * sin
        return x_rotated


    class InterleavedRoPE(tf.keras.layers.Layer):
        def __init__(self, dim, max_seq_len=1024, temperature=100000, **kwargs):
            super().__init__(**kwargs)
            if dim % 2 != 0:
                raise ValueError(f"Embedding dimension `dim` ({dim}) must be even for InterleavedRoPE.")
            self.dim = dim
            self.max_seq_len = max_seq_len
            # Instantiate the RotaryEmbedding layer
            # Ensure the name is consistent if needed for saving/loading
            self.rotary_emb = RotaryEmbedding(dim, max_seq_len, temperature, name="rotary_embedding")

        def call(self, x):
            # Get sin and cos from the RotaryEmbedding layer's call method
            # *** Pass only 'x'. RotaryEmbedding calculates seq_len internally. ***
            sin, cos = self.rotary_emb(x)

            # Apply the positional embeddings
            x_embedded = apply_rotary_pos_emb(x, sin, cos)
            return x_embedded

        def get_config(self):
            config = super().get_config()
            config.update({
                    "dim": self.dim,
                    "max_seq_len": self.max_seq_len,
            })
            # Keras handles nested layer serialization automatically
            return config

        @classmethod
        def from_config(cls, config):
            # Keras handles nested layer restoration automatically
            return cls(**config)


    # GPT2 configurables

    # Optimal for accuracy thus far:
    max_seq_length = 1536
    tokenizer_checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

    inp = tf.keras.layers.Input(shape=(), dtype=tf.string)
    gp2_tokenizer = NewTokenizerLayer(max_seq_length=max_seq_length,tokenizer_checkpoint=tokenizer_checkpoint)
    VOCABULARY_SIZE = gp2_tokenizer.tokenizer.vocab_size
    tokens = gp2_tokenizer(inp)

    # On larger hardware, this could probably be increased considerably and
    # Probably would improve performance ...
    embedding_n = 12  # Define EMBEDDING_DIM here, to match your embedding layer.
    EMBEDDING_DIM = int(embedding_n * 2)

    embedded = tf.keras.layers.Embedding(
            input_dim=VOCABULARY_SIZE,
            output_dim=EMBEDDING_DIM,
            input_length=max_seq_length,
            mask_zero=True)(tokens)

    position_embedding = InterleavedRoPE(
            dim=EMBEDDING_DIM,
            max_seq_len=max_seq_length,
            temperature=temperature,
            # initializer="uniform",
    )(embedded)

    # As an FYI, we tried an add layer both with and without
    # LayerNorm ... It degraded accuracy
    # Just an FYI for anyone trying to apply conventional wisdom
    # to save you the time ...
    x = x = tf.keras.layers.Concatenate()([embedded, position_embedding])
    x = tf.keras.layers.Dropout(dropout)(x)
    flattened = tf.keras.layers.Flatten()(x)

    cerebros_base_model = tf.keras.Model(
        inputs=inp,
        outputs=flattened  # Output enhanced embeddings now
    )


    """### Cerebros search for the best model"""

    #
    # Cerebros configurables
    #

    # Hard set intentionally
    minimum_levels = 2
    maximum_levels = 2 # [3,7]

    minimum_units_per_level = 4


    minimum_neurons_per_unit = 1

    moities_to_try = 5
    tries_per_moity = 1

    #
    # Logging
    #
    TIME = pendulum.now(tz='America/New_York').__str__()[:16]\
            .replace('T', '_')\
            .replace(':', '_')\
            .replace('-', '_')
    PROJECT_NAME = f'{TIME}_cerebros_auto_ml_phishing_email_test'

    meta_trial_number = 42 # irrelevant unless in distributed training


    cerebros_automl = SimpleCerebrosRandomSearch(
            unit_type=DenseUnit,
            input_shapes=INPUT_SHAPES,
            output_shapes=OUTPUT_SHAPES,
            training_data=training_x,
            labels=train_labels,
            validation_split=0.35,
            direction='maximize',
            metric_to_rank_by="val_binary_accuracy",
            minimum_levels=minimum_levels,
            maximum_levels=maximum_levels,
            minimum_units_per_level=minimum_units_per_level,
            maximum_units_per_level=maximum_units_per_level,
            minimum_neurons_per_unit=minimum_neurons_per_unit,
            maximum_neurons_per_unit=maximum_neurons_per_unit,
            activation=activation,
            final_activation='sigmoid',
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
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                    tf.keras.metrics.BinaryAccuracy(), 
                    tf.keras.metrics.Precision(), 
                    tf.keras.metrics.Recall()
            ],
            epochs=epochs,
            project_name=f"{PROJECT_NAME}_meta_{meta_trial_number}",
            model_graphs='model_graphs',
            batch_size=batch_size,
            meta_trial_number=meta_trial_number,
            base_models=[cerebros_base_model],
            train_data_dtype=tf.string)
        
    cerebros_t0 = time.time()
    result = cerebros_automl.run_random_search()
    cerebros_t1 = time.time()
    cerebros_time_all_models_min = (cerebros_t1 - cerebros_t0) / 60
    models_tried = moities_to_try  * tries_per_moity
    cerebros_time_per_model = cerebros_time_all_models_min / models_tried

    print(f"Cerebros trained {models_tried} models FROM A COLD START in ONLY {cerebros_time_all_models_min} min. Cerebros took only {cerebros_time_per_model} minutes on average per model.")



    print(f'Cerebros best accuracy achieved is {result}')
    print(f'val set accuracy')
    
    return float(result)

import optuna




# Define the Optuna study
study =\
        optuna.create_study(
                study_name="NLP-optimization-study-0001",      
                direction="maximize",
                storage="sqlite:///NLP-optimization-study-0001.db",
                sampler=optuna.samplers.TPESampler(multivariate=True))

study.optimize(
        objective,
        n_trials=15)

print(f"Best parameters: {study.best_params}")
