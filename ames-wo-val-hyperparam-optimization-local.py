"""

Hyperparameter optimization script for the updated Ames script

"""

import numpy as np
import optuna
import pendulum
import pandas as pd
import tensorflow as tf
from cerebros.simplecerebrosrandomsearch.simple_cerebros_random_search\
    import SimpleCerebrosRandomSearch
from cerebros.units.units import DenseUnit
from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component\
    import zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid
from ast import literal_eval

# Define constants
LABEL_COLUMN = 'price'
NUMBER_OF_TRAILS_PER_BATCH = 2
NUMBER_OF_BATCHES_OF_TRIALS = 2

# Load data
raw_data = pd.read_csv('ames.csv')
needed_cols = [
    col for col in raw_data.columns 
    if raw_data[col].dtype != 'object' 
    and col != LABEL_COLUMN]
data_numeric = raw_data[needed_cols].fillna(0).astype(float)
label = raw_data.pop(LABEL_COLUMN)

data_np = data_numeric.values

tensor_x = tf.constant(data_np)

training_x = [tensor_x]

INPUT_SHAPES = [training_x[i].shape[1] for i in np.arange(len(training_x))]

train_labels = [tf.constant(label.values.astype(float))]

OUTPUT_SHAPES = [1]  

def objective(trial):
    # Define hyperparameter space
    minimum_levels = trial.suggest_int('minimum_levels', 1, 8)
    maximum_levels = trial.suggest_int('maximum_levels', minimum_levels, 8)
    minimum_units_per_level = trial.suggest_int('minimum_units_per_level', 1, 8)
    maximum_units_per_level = trial.suggest_int('maximum_units_per_level', minimum_units_per_level, 8)
    minimum_neurons_per_unit = trial.suggest_int('minimum_neurons_per_unit', 1, 8)
    maximum_neurons_per_unit = trial.suggest_int('maximum_neurons_per_unit', minimum_neurons_per_unit, 8)
    activation = trial.suggest_categorical('activation', ['relu', 'elu', 'gelu', 'swish', 'softplus'])
    predecessor_level_connection_affinity_factor_first = trial.suggest_loguniform('predecessor_level_connection_affinity_factor_first', 0.1, 40.0)
    predecessor_level_connection_affinity_factor_main = trial.suggest_loguniform('predecessor_level_connection_affinity_factor_main', 0.1, 40.0)
    max_consecutive_lateral_connections = trial.suggest_int('max_consecutive_lateral_connections', 1, 40)
    p_lateral_connection = trial.suggest_loguniform('p_lateral_connection', 0.1, 40.0)
    num_lateral_connection_tries_per_unit = trial.suggest_int('num_lateral_connection_tries_per_unit', 1, 40)
    learning_rate = trial.suggest_loguniform('learning_rate', 10**-6, 0.6)
    epochs = trial.suggest_int('epochs', 1, 150)
    batch_size = trial.suggest_int('batch_size', 1, 800)

    meta_trial_number = 0  

    TIME = pendulum.now().__str__()[:16]\
        .replace('T', '_')\
        .replace(':', '_')\
        .replace('-', '_')
    PROJECT_NAME = f'{TIME}_cerebros_auto_ml_test'

    cerebros = SimpleCerebrosRandomSearch(
        unit_type=DenseUnit,
        input_shapes=INPUT_SHAPES,
        output_shapes=OUTPUT_SHAPES,
        training_data=training_x,
        labels=train_labels,
        validation_split=0.35,
        direction='minimize',
        metric_to_rank_by='val_root_mean_squared_error',
        minimum_levels=minimum_levels,
        maximum_levels=maximum_levels,
        minimum_units_per_level=minimum_units_per_level,
        maximum_units_per_level=maximum_units_per_level,
        minimum_neurons_per_unit=minimum_neurons_per_unit,
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
        model_graphs='model_graphs',
        batch_size=batch_size,
        meta_trial_number=meta_trial_number)

    result = cerebros.run_random_search()
    return result

def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=NUMBER_OF_TRAILS_PER_BATCH * NUMBER_OF_BATCHES_OF_TRIALS)
    print('Best trial:')
    best_trial = study.best_trial
    print('  Value: ', best_trial.value)
    print('  Params: ')
    for key, value in best_trial.params.items():
        print(f'    {key}: {value}')

if __name__ == '__main__':
    main()
