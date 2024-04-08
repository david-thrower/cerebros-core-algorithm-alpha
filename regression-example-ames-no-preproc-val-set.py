
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

def hash_a_row(row):
    """casts a row of a Pandas DataFrame as a String, hashes it, and casts it
    as an unsigned integer. This is used to modulous divide by 100, leaving a
    remainder in range 0:99. In hash_based_split(), this is used to assign rows
    to train or test, without a risk of identical rows being assigned
    to both sets. (if the remainder / 100 is <= test_set_proportion,
    it is in train, otherwisw it goes to test.)"""
    str_cells = [str(cell) for cell in row]
    cat_cells = "".join(str_cells)
    signed_hash = hash(cat_cells)
    unsigned_hash = abs(signed_hash)
    print(f"unsigned_hash is: {unsigned_hash}")
    return unsigned_hash  # Unsigned always positive hash...


def hash_based_split(df,  # Pandas dataframe
                     labels,  # Pandas series
                     test_size: float = 0.1,
                     hash_column: str = "*"):
    """
    Split a pandas dataframe to train and test splits using hashing and
    modulus division. This ensures that duplicate rows always fall on the same
    side of the train test split.

    Args:
    - df: pandas dataframe to be split
    - test_size: float between 0 and 1 representing the proportion of data to
      be used in the test split
    - hash_column: string representing the name of the column to be used for
      hashing. By default, all rows will be used '*' to ensure only a given
      column is unique, enter as a string the name of the string.

    Returns:
    - train_df: pandas dataframe containing the training data
    - test_df: pandas dataframe containing the test data
    """

    if not (test_size > 0 and test_size < 1):
        raise ValueError("Test and val splits must be in range (0,1)")

    if hash_column == "*":
        # Make a concat all.
        hash_values = df.apply(hash_a_row,
                               axis=1)
    elif isinstance(hash_column, list):
        hash_values = df[hash_column].apply(hash_a_row,
                                            axis=1)
    # Compute the hash values for the hash column
    else:
        hash_values = df[hash_column].apply(hash)

    # Calculate the cutoff hash value for the test split
    # hash_cutoff = hash_values.max() * test_size
    # Split the data based on the hash value
    # (hash_values % 100) <= 100 * test_size
    train_idx = (hash_values % 100) >= 100 * test_size

    #train_idx = hash_values % hash_cutoff
    train_df = df[train_idx]
    test_idx = ~ train_idx

    # test_idx = hash_values % cutoff_hash == 0
    test_df = df[test_idx]
    if len(labels.shape) == 1:
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
    elif len(labels.shape) > 1:
        train_labels = labels[train_idx, :]
        test_labels = labels[test_idx, :]
    else:
        raise ValueError("It appears the labels have "
                         f"{labels.shape} axes. "
                         "That is not supported yet.")
    return train_df, train_labels, test_df, test_labels


# white = pd.read_csv('wine_data.csv')

raw_data = pd.read_csv('ames.csv')
needed_cols = [
    col for col in raw_data.columns if raw_data[col].dtype != 'object']
data_numeric = raw_data[needed_cols].fillna(0).astype(float)
label = raw_data.pop('price')


train_df, train_labels_pd, val_df, val_labels_pd =\
    hash_based_split(
        df=data_numeric,  # Pandas dataframe
        labels=label,  # Pandas series
        test_size=0.35,
        hash_column="*")


train_data_np = train_df.values
print(f"Shape of train data: {train_data_np.shape}")

tensor_x =\
    tf.constant(train_df.values)


training_x = [tensor_x]

INPUT_SHAPES = [training_x[i].shape[1] for i in np.arange(len(training_x))]

train_labels = [tf.constant(train_labels_pd.values.astype(float))]
print(f"Shape of train labels: {train_labels_pd.shape}")

OUTPUT_SHAPES = [1]  # [train_labels[i].shape[1]

## Val set:

print(f"Shape of val data: {val_df.shape}")
val_tensor_x = tf.constant(val_df.values)
val_x = [val_tensor_x]


val_labels = [val_labels_pd.values]
print(f"Shape of val labels: {val_labels_pd.shape}")

# Params for a training function (Approximately the oprma
# discovered in a bayesian tuning study done on Katib)

meta_trial_number = 0  # In distributed training set this to a random number
activation = "gelu"
predecessor_level_connection_affinity_factor_first = 19.613
predecessor_level_connection_affinity_factor_main = 0.5518
max_consecutive_lateral_connections = 34
p_lateral_connection = 0.36014
num_lateral_connection_tries_per_unit = 11
learning_rate = 0.095
epochs = 145
batch_size = 634
maximum_levels = 5
maximum_units_per_level = 5
maximum_neurons_per_unit = 25


cerebros =\
    SimpleCerebrosRandomSearch(
        unit_type=DenseUnit,
        input_shapes=INPUT_SHAPES,
        output_shapes=OUTPUT_SHAPES,
        training_data=training_x,
        labels=train_labels,
        validation_split=0.0,
        direction='minimize',
        metric_to_rank_by='val_root_mean_squared_error',
        minimum_levels=1,
        maximum_levels=maximum_levels,
        minimum_units_per_level=1,
        maximum_units_per_level=maximum_units_per_level,
        minimum_neurons_per_unit=1,
        maximum_neurons_per_unit=maximum_neurons_per_unit,
        validation_data=(val_x, val_labels),
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
