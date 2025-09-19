import numpy as np
import pandas as pd
import tensorflow as tf
from cerebros.simplecerebrosrandomsearch.simple_cerebros_random_search import SimpleCerebrosRandomSearch
from cerebros.units.units import DenseUnit
from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component import zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid

# Minimal, fast Ames regression smoke test
LABEL_COLUMN = 'price'
raw_data = pd.read_csv('ames.csv')
needed_cols = [c for c in raw_data.columns if raw_data[c].dtype != 'object' and c != LABEL_COLUMN]
features = raw_data[needed_cols].fillna(0).astype(float)
labels = raw_data[LABEL_COLUMN].astype(float)

# Tiny subset for speed
features = features.head(256)
labels = labels.head(256)

x = [tf.constant(features.values, dtype=tf.float32)]
y = [tf.constant(labels.values, dtype=tf.float32)]

INPUT_SHAPES = [x[0].shape[1]]
OUTPUT_SHAPES = [1]

cerebros = SimpleCerebrosRandomSearch(
    unit_type=DenseUnit,
    input_shapes=INPUT_SHAPES,
    output_shapes=OUTPUT_SHAPES,
    training_data=x,
    labels=y,
    validation_split=0.2,
    direction='minimize',
    metric_to_rank_by='val_root_mean_squared_error',
    minimum_levels=2,
    maximum_levels=2,
    minimum_units_per_level=2,
    maximum_units_per_level=2,
    minimum_neurons_per_unit=8,
    maximum_neurons_per_unit=8,
    activation='relu',
    final_activation=None,
    number_of_architecture_moities_to_try=1,
    number_of_tries_per_architecture_moity=1,
    minimum_skip_connection_depth=1,
    maximum_skip_connection_depth=2,
    predecessor_level_connection_affinity_factor_first=2,
    predecessor_level_connection_affinity_factor_first_rounding_rule='ceil',
    predecessor_level_connection_affinity_factor_main=0.7,
    predecessor_level_connection_affinity_factor_main_rounding_rule='ceil',
    predecessor_level_connection_affinity_factor_decay_main=zero_7_exp_decay,
    seed=123,
    max_consecutive_lateral_connections=2,
    gate_after_n_lateral_connections=2,
    gate_activation_function=simple_sigmoid,
    p_lateral_connection=0.5,
    p_lateral_connection_decay=zero_95_exp_decay,
    num_lateral_connection_tries_per_unit=1,
    learning_rate=0.001,
    loss='mse',
    metrics=[tf.keras.metrics.RootMeanSquaredError()],
    epochs=5,
    patience=3,
    project_name='ames_regression_smoke',
    batch_size=32,
    meta_trial_number=0,
    chart_network_graph=False,
    # Force dense output for regression sanity check
    output_layer_kind='dense',
)

best = cerebros.run_random_search()
print('Smoke test best val RMSE:', best)

best_model = cerebros.get_best_model(purge_model_storage_files=True)
print(best_model.summary())
