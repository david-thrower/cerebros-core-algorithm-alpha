"""
EfficientNet PEFT on CIFAR-100 with Cerebros
"""

import sys
sys.path.insert(0, '../..')

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Resizing, Lambda, Flatten, Dense
import pandas as pd
import numpy as np
from cerebros.simplecerebrosrandomsearch.simple_cerebros_random_search\
    import SimpleCerebrosRandomSearch
import pendulum
from cerebros.units.units import DenseUnit
from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component\
    import zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid
from ast import literal_eval

# PEFT with Cerebros

# Download Cifar-100 data

(X_train, y_train), (X_test, y_test) = cifar100.load_data()

# Input and output shapes: Cifar-100 has 32x32 images with 3 channels (RGB). The output is 100 categorical probabilities.

input_shape   = (32,32,3)
INPUT_SHAPES  = [input_shape]
OUTPUT_SHAPES = [100]

# Subsampling a small balanced set of samples (with of without shuffling) from the train dataset and train labels

def subsample_train(X_train, y_train, num_samples):
    #
    X_sub = []
    y_sub = []
    #
    assert 1 <= num_samples <= 500
    #
    for cat in range(100):
        #
        ind, _ = np.where(y_train==cat)
        #
        X_cat = X_train[ind]
        X_cat = X_cat[:num_samples]
        y_cat = y_train[ind]
        y_cat = y_cat[:num_samples]
        #
        X_sub += [X_cat]
        y_sub += [y_cat]
    #
    X_sub = np.vstack(X_sub)
    y_sub = np.vstack(y_sub)
    #
    assert X_sub.shape[0] == y_sub.shape[0]
    #
    ind = np.arange(X_sub.shape[0])
    np.random.shuffle(ind)
    #
    return X_sub[ind], y_sub[ind]

# We take only 50 samples in each category

num_samples = 50
X_sub, y_sub = subsample_train(X_train, y_train, num_samples)

# Preparing tensors for the training set and labels

training_x   = [tf.constant(X_sub)]
y_train_cat  = to_categorical(y_sub, 100)
train_labels = [tf.constant(y_train_cat)]

# Donwloading EfficientNet (v.2, small)

enet = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax',
    include_preprocessing=True
)

# Resizing images to meet EfficientNet's input shape

def resize(x):
    return tf.image.resize(x,size=(384,384),method='bilinear')

# Make only the last convolutional layer trainable

for layer in enet.layers:
    layer.trainable = False
enet.layers[-6].trainable = True

# Preparing the base model for Cerebros search

enet_io = Model(inputs=enet.layers[0].input,
                outputs=enet.layers[-3].output)

input_layer = Input(shape=input_shape)
prep = Lambda(resize)(input_layer)
out = Flatten()(enet_io(prep))
base_mod = Model(inputs=input_layer, outputs=out)

# Cerebros configurables

activation = 'swish'
predecessor_level_connection_affinity_factor_first = 2.0
predecessor_level_connection_affinity_factor_main = 0.97
max_consecutive_lateral_connections = 5
p_lateral_connection = 0.97
num_lateral_connection_tries_per_unit = 2
learning_rate = 0.001
epochs = 5  # [1, 100]
batch_size = 20
maximum_levels = 4  # [3,7]
maximum_units_per_level = 7  # [2,10]
maximum_neurons_per_unit = 4  # [2,20]

# Final training task

TIME = pendulum.now(tz='America/New_York').__str__()[:16]\
    .replace('T', '_')\
    .replace(':', '_')\
    .replace('-', '_')
#
PROJECT_NAME = f'{TIME}_cerebros_auto_ml_test_cifar10_efficientnet'
#
meta_trial_number = 42
#
cerebros_automl = SimpleCerebrosRandomSearch(
    unit_type=DenseUnit,
    input_shapes=INPUT_SHAPES,
    output_shapes=OUTPUT_SHAPES,
    training_data=training_x,
    labels=train_labels,
    validation_split=0.2,
    direction='maximize',
    metric_to_rank_by="val_top_1_categorical_accuracy",
    minimum_levels=2,
    maximum_levels=maximum_levels,
    minimum_units_per_level=1,
    maximum_units_per_level=maximum_units_per_level,
    minimum_neurons_per_unit=1,
    maximum_neurons_per_unit=maximum_neurons_per_unit,
    activation=activation,
    final_activation='softmax',
    number_of_architecture_moities_to_try=3,
    number_of_tries_per_architecture_moity=2,
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
    metrics=[tf.keras.metrics.TopKCategoricalAccuracy(
                k=1, name='top_1_categorical_accuracy')
             ],
    epochs=epochs,
    project_name=f"{PROJECT_NAME}_meta_{meta_trial_number}",
    model_graphs='model_graphs',
    batch_size=batch_size,
    meta_trial_number=meta_trial_number,
    base_models=[base_mod])

# Search and save the best model

result = cerebros_automl.run_random_search()

print(f'Best accuracy achieved is {result}')
print(f'top-1 categorical accuracy')

# Evaluating the best model found

best_model_found = cerebros_automl.get_best_model()

# because of the resizing Lambda layer, needs safe_mode=False in some environments
#best_model_found = tf.keras.models.load_model(cerebros_automl.best_model_path, safe_mode=False)

#
eval_loss = tf.keras.losses.CategoricalCrossentropy()
#
eval_metrics =\
[tf.keras.metrics.TopKCategoricalAccuracy(k=1,\
            name='eval_top_1_categorical_accuracy'),
 tf.keras.metrics.TopKCategoricalAccuracy(k=5,\
            name='eval_top_5_categorical_accuracy')
]

best_model_found.compile(loss=eval_loss, metrics=eval_metrics)
best_model_found.summary()

print("Evaluating best model found ...")
print("Loss | Top-1 accuracy | Top-5 accuracy")
y_test_cat = to_categorical(y_test, 100)
best_model_found.evaluate(X_test, y_test_cat)

