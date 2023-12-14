
import numpy as np
# from multiprocessing import Pool  # , Process
from cerebros.simplecerebrosrandomsearch.simple_cerebros_random_search\
    import SimpleCerebrosRandomSearch
import pendulum
import pandas as pd
import tensorflow as tf
from cerebros.units.units import DenseUnit
from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component\
    import zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid
from ast import literal_eval

NUMBER_OF_TRAILS_PER_BATCH = 2
NUMBER_OF_BATCHES_OF_TRIALS = 2

###

## your data:


TIME = pendulum.now().__str__()[:16]\
    .replace('T', '_')\
    .replace(':', '_')\
    .replace('-', '_')
PROJECT_NAME = f'{TIME}_cerebros_auto_ml_test'


# white = pd.read_csv('wine_data.csv')

raw_data = pd.read_csv('ames.csv')
needed_cols = [
    col for col in raw_data.columns if raw_data[col].dtype != 'object']
data_numeric = raw_data[needed_cols].fillna(0).astype(float)
label = raw_data.pop('price')

data_np = data_numeric.values

tensor_x =\
    tf.constant(data_np)

training_x = [tensor_x]

INPUT_SHAPES = [training_x[i].shape[1] for i in np.arange(len(training_x))]

train_labels = [label.values]

OUTPUT_SHAPES = [1]  # [train_labels[i].shape[1]


# Params for a training function (Approximately the oprma
# discovered in a bayesian tuning study done on Katib)

meta_trial_number = 0  # In distributed training set this to a random number
activation = "gelu"
predecessor_level_connection_affinity_factor_first = 19.613
predecessor_level_connection_affinity_factor_main = 0.5518
max_consecutive_lateral_connections = 34
p_lateral_connection = 0.36014
num_lateral_connection_tries_per_unit = 11
learning_rate = 1.0  # Ideal lr may be in [0.4, 3.0] strangely  # 0.095
epochs = 145
batch_size = 634
maximum_levels = 5
maximum_units_per_level = 5
maximum_neurons_per_unit = 25


# meta_trial_number = 0  # In distributed training set this to a random number
# activation = "gelu"
# predecessor_level_connection_affinity_factor_first = 19.613
# predecessor_level_connection_affinity_factor_main = 0.5518
# max_consecutive_lateral_connections = 34
# p_lateral_connection = 0.36014
# num_lateral_connection_tries_per_unit = 11
# learning_rate = 0.7 # 0.25 # 0.095
# epochs = 200 # 145
# batch_size = 300 # 634
# maximum_levels = 4 # 5
# maximum_units_per_level = 4 # 5
# maximum_neurons_per_unit = 10 # 25


cerebros =\
    SimpleCerebrosRandomSearch(
        unit_type=DenseUnit,
        input_shapes=INPUT_SHAPES,
        output_shapes=OUTPUT_SHAPES,
        training_data=training_x,
        labels=train_labels,
        validation_split=0.35,
        direction='minimize',
        metric_to_rank_by='val_root_mean_squared_error',
        minimum_levels=1,
        maximum_levels=maximum_levels,
        minimum_units_per_level=1,
        maximum_units_per_level=maximum_units_per_level,
        minimum_neurons_per_unit=1,
        maximum_neurons_per_unit=maximum_neurons_per_unit,
        activation=activation,
        final_activation=None,
        number_of_architecture_moities_to_try=7,
        number_of_tries_per_architecture_moity=1,
        number_of_generations=3,
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
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
        epochs=epochs,
        patience=7,
        project_name=f"{PROJECT_NAME}_meta_{meta_trial_number}",
        # use_multiprocessing_for_multiple_neural_networks=False,  # pull this param
        model_graphs='model_graphs',
        batch_size=batch_size,
        meta_trial_number=meta_trial_number)
result = cerebros.run_random_search()

print("Best model: (May need to re-initialize weights, and retrain with early stopping callback)")
best_model_found = cerebros.get_best_model()
print(best_model_found.summary())

print("result extracted from cerebros")
print(f"Final result was (val_root_mean_squared_error): {result}")
