#
# This test is based on Sam Blake's preprint "Integer Factorisation, Fermat & 
# Machine Learning on a Classical Computer", arXiv:2308.12290
#
# Detecting the ratio of semiprime factors may theoretically help improving 
# the classical Lawrence algorithm for semiprime factorization
#

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, '../..')

import numpy as np
import gmpy2
from gmpy2 import mpz, mpq, mpfr
from cerebros.simplecerebrosrandomsearch.simple_cerebros_random_search\
    import SimpleCerebrosRandomSearch
import pendulum
import pandas as pd
import tensorflow as tf
from cerebros.units.units import DenseUnit
from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component\
    import zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid
from ast import literal_eval
from sklearn.metrics import confusion_matrix

TIME = pendulum.now().__str__()[:16]\
    .replace('T', '_')\
    .replace(':', '_')\
    .replace('-', '_')
PROJECT_NAME = f'{TIME}_cerebros_auto_ml_test_semiprimes'

raw_data = pd.read_csv("semiprime_ratio_training_data_128_ratio_2_3.csv")
raw_data.iloc[:,0] = raw_data.iloc[:,0].apply(mpz)
raw_data = raw_data.values
X = np.array(list(map(lambda x: [int(i) for i in gmpy2.digits(x,2)], raw_data[:,0])), dtype=np.int32)
y = np.array(raw_data[:,1], dtype=np.int32)

num_data_points = 200_000
X_train, y_train = X[:num_data_points,:], y[:num_data_points]
X_test,  y_test  = X[num_data_points:,:], y[num_data_points:]

tensor_x = tf.constant(X_train)
training_x = [tensor_x]
INPUT_SHAPES = [training_x[i].shape[1] for i in np.arange(len(training_x))]
train_labels = [y_train]
OUTPUT_SHAPES = [1] 

meta_trial_number = 42  # in distributed training set this to a random number

activation = "relu"
predecessor_level_connection_affinity_factor_first = 19.613
predecessor_level_connection_affinity_factor_main = 0.5518
max_consecutive_lateral_connections = 34
p_lateral_connection = 0.36014
num_lateral_connection_tries_per_unit = 11
learning_rate = 0.095
epochs = 10
batch_size = 634
maximum_levels = 5
maximum_units_per_level = 5
maximum_neurons_per_unit = 25

cerebros_automl =\
    SimpleCerebrosRandomSearch(
        unit_type=DenseUnit,
        input_shapes=INPUT_SHAPES,
        output_shapes=OUTPUT_SHAPES,
        training_data=training_x,
        labels=train_labels,
        validation_split=0.35,
        direction='maximize',
        metric_to_rank_by='val_binary_accuracy',
        minimum_levels=1,
        maximum_levels=maximum_levels,
        minimum_units_per_level=1,
        maximum_units_per_level=maximum_units_per_level,
        minimum_neurons_per_unit=1,
        maximum_neurons_per_unit=maximum_neurons_per_unit,
        activation=activation,
        final_activation='sigmoid',
        number_of_architecture_moities_to_try=3,
        number_of_tries_per_architecture_moity=2,
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
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.TrueNegatives(),
                 tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives(),
                 tf.keras.metrics.TruePositives()],
        epochs=epochs,
        patience=7,
        project_name=f"{PROJECT_NAME}_meta_{meta_trial_number}",
        model_graphs='model_graphs',
        batch_size=batch_size,
        meta_trial_number=meta_trial_number)

result = cerebros_automl.run_random_search()
best_model_found = cerebros_automl.get_best_model()

trainable_params = np.sum([np.prod(w.get_shape()) for w in best_model_found.trainable_weights])
non_trainable_params = np.sum([np.prod(w.get_shape()) for w in best_model_found.non_trainable_weights])
total_params = trainable_params + non_trainable_params

print(f"Best model found: {total_params} total parameters ({trainable_params} trainable, {non_trainable_params} non-trainable)")

print(f"Best accuracy is ({cerebros_automl.metric_to_rank_by}): {result}")

best_model_found.compile()
best_model_found.summary()

y_pred = best_model_found.predict(X_test)
threshold = y_pred.mean()
y_pred = (y_pred > threshold).astype(int) 

cm = confusion_matrix(y_test, y_pred, normalize='all')
print("Confusion matrix")
print(cm)