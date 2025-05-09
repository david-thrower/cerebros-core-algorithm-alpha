# -*- coding: utf-8 -*-
"""phishing-email-detection-gpt2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10KKTHjBkdfKBpT9OLIj2eZs533BuCS6h
"""

## GPT2 + Cerebros for Phishing email detection


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

"""### A custom GPT2 encoder layer for text embedding"""


class GPT2Layer(tf.keras.layers.Layer):

    def __init__(self, max_seq_length, **kwargs):
        #
        super(GPT2Layer, self).__init__(**kwargs)
        #
        # Load the GPT2 tokenizer, preprocessor and model
        self.tokenizer = GPT2Tokenizer.from_preset("gpt2_base_en")
        self.preprocessor = GPT2Preprocessor(self.tokenizer,
                                             sequence_length=max_seq_length)
        self.encoder   = GPT2Backbone.from_preset("gpt2_base_en")
        #
        # Set whether the GPT2 model's layers are trainable
        #self.encoder.trainable = False
        for layer in self.encoder.layers:
            layer.trainable = True
        #
        # self.encoder.layers[-2].trainable = True
        #
        # Set the maximum sequence length for tokenization
        self.max_seq_length = max_seq_length

    def call(self, inputs):
        #
        # Output the GPT2 embedding
        prep = self.preprocessor([inputs])
        embedding  = self.encoder(prep)
        avg_pool = tf.reduce_mean(embedding, axis=1)
        #
        return avg_pool

    def get_config(self):
        #
        config = super(GPT2Layer, self).get_config()
        config.update({'max_seq_length': self.max_seq_length})
        #
        return config

    @classmethod
    def from_config(cls, config):
        #
        return cls(max_seq_length=config['max_seq_length'])

# GPT2 configurables
max_seq_length = 96

# GPT Baseline Model
input_layer = Input(shape=(), dtype=tf.string)
gpt2_layer = GPT2Layer(max_seq_length)(input_layer)
#output = Flatten()(gpt2_layer)
binary_output = tf.keras.layers.Dense(1, activation='sigmoid')(gpt2_layer)

gpt_baseline_model = Model(inputs=input_layer, outputs=binary_output)


gpt_baseline_model.compile(
    optimizer=Adam(learning_rate=1e-4),  # Small LR since we're fine-tuning GPT
    loss='binary_crossentropy',
    # metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    metrics=[tf.keras.metrics.BinaryAccuracy(), 
         tf.keras.metrics.Precision(), 
         tf.keras.metrics.Recall()]
)

gpt_t0 = time.time()

print(gpt_baseline_model.summary())

history = gpt_baseline_model.fit(
    x=X_train,  # Input data
    y=y_train,  # Labels
    epochs=3,  # Number of training iterations
    batch_size=16,  # Batch size small due to GPU memory constraints
    validation_split=0.2,  # Hold out 20% of training data for validation
    shuffle=True,  # Shuffle data at each epoch
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            min_delta=0.001
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6
        )
    ]
)

gpt_t1 = time.time()
gpt_time_on_one_model_min =  (gpt_t1 - gpt_t0) / 60

hy_df = pd.DataFrame(history.history)
print(hy_df)



### Cerebros model:

from transformers import AutoTokenizer
import tensorflow as tf

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
    def __init__(self, dim, max_seq_len=1024, **kwargs):
        super().__init__(**kwargs)
        if dim % 2 != 0:
             raise ValueError(f"Embedding dimension `dim` ({dim}) must be even for InterleavedRoPE.")
        self.dim = dim
        self.max_seq_len = max_seq_len
        # Instantiate the RotaryEmbedding layer
        # Ensure the name is consistent if needed for saving/loading
        self.rotary_emb = RotaryEmbedding(dim, max_seq_len, name="rotary_embedding")

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
EMBEDDING_N = 12  # Define EMBEDDING_DIM here, to match your embedding layer.
EMBEDDING_DIM = int(EMBEDDING_N * 2)

embedded = tf.keras.layers.Embedding(
    input_dim=VOCABULARY_SIZE,
    output_dim=EMBEDDING_DIM,
    input_length=max_seq_length,
    mask_zero=True)(tokens)

position_embedding = InterleavedRoPE(
    dim=EMBEDDING_DIM,
    max_seq_len=max_seq_length,
    # initializer="uniform",
)(embedded)

# As an FYI, we tried an add layer both with and without
# LayerNorm ... It degraded accuracy
# Just an FYI for anyone trying to apply conventional wisdom
# to save you the time ...
x = x = tf.keras.layers.Concatenate()([embedded, position_embedding])
x = tf.keras.layers.Dropout(0.4)(x)  # AI suggested 0.4
flattened = tf.keras.layers.Flatten()(x)

cerebros_base_model = tf.keras.Model(
    inputs=inp,
    outputs=flattened  # Output enhanced embeddings now
)


"""### Cerebros search for the best model"""

#
# Cerebros configurables
#
activation = "relu"
predecessor_level_connection_affinity_factor_first = 10
predecessor_level_connection_affinity_factor_main = 40
max_consecutive_lateral_connections = 20
p_lateral_connection = 30
num_lateral_connection_tries_per_unit = 25
learning_rate = 3 * 10 ** -3
epochs = 15  #
batch_size = 17
minimum_levels = 2
maximum_levels = 2 # [3,7]

minimum_units_per_level = 4
maximum_units_per_level = 7

minimum_neurons_per_unit = 1
maximum_neurons_per_unit = 2

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
    # loss=tf.keras.losses.CategoricalHinge(),
    metrics=[tf.keras.metrics.BinaryAccuracy(), 
         tf.keras.metrics.Precision(), 
         tf.keras.metrics.Recall()],
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
print(f"GPT2 took {gpt_time_on_one_model_min} just to FINE TUNE one PRE - TRAINED model for 3 epochs. Although this is a small scale test, this shows the advantage of scaling in ON timing VS ON**2 timing.")


print(f'Cerebros best accuracy achieved is {result}')
print(f'val set accuracy')

# """### Testing the best model found"""
