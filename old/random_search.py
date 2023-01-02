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


white = pd.read_csv('wine_data.csv')

tensor_x_0 =\
    tf.constant(white[['residual sugar',
                       'chlorides',
                       'total sulfur dioxide',
                       'free sulfur dioxide']].values)

tensor_x_1 =\
    tf.constant(white[['fixed acidity',
                       'volatile acidity',
                       'citric acid',
                       'density',
                       'pH',
                       'sulphates',
                       'alcohol']].values)


training_x = [tensor_x_0, tensor_x_1]
INPUT_SHAPES = [training_x[i].shape[1] for i in np.arange(len(training_x))]


#alcohol_labels = tf.constant(pd.get_dummies(pd.cut(white['alcohol'], np.arange(
#    white['alcohol'].min(), white['alcohol'].max()))).values)
quality_labels = tf.constant(pd.get_dummies(white['quality']).values)

train_labels = [quality_labels]

OUTPUT_SHAPES = [train_labels[i].shape[1]
                 for i in np.arange(len(train_labels))]

###


def generate_trial(trial_number):
    trial = {}
    trial['meta_trial_number'] = trial_number
    trial['activation'] = np.random.choice(['relu', 'elu', 'gelu'])
    trial['predecessor_level_connection_affinity_factor_first'] =\
        np.random.choice(np.linspace(0.2, 20, num=50))
    trial['predecessor_level_connection_affinity_factor_main'] =\
        np.random.choice(np.linspace(0.2, 20, 50))
    trial['max_consecutive_lateral_connections'] =\
        np.random.choice(np.arange(200))
    trial['p_lateral_connection'] =\
        np.random.choice(np.linspace(0.001, .999, 100))
    trial['num_lateral_connection_tries_per_unit'] =\
        np.random.choice(np.arange(1, 20))
    trial['learning_rate'] = np.random.choice(
        np.linspace(.000001, .999999, 50))
    trial['epochs'] = np.random.choice(np.arange(2, 8))
    trial['batch_size'] =\
        np.random.choice(np.arange(10, 1000))
    return trial


def run_param_permutation(trial):
    meta_trial_number = trial['meta_trial_number']
    activation = trial['activation']
    predecessor_level_connection_affinity_factor_first = trial[
        'predecessor_level_connection_affinity_factor_first']
    predecessor_level_connection_affinity_factor_main = trial[
        'predecessor_level_connection_affinity_factor_main']
    max_consecutive_lateral_connections = trial['max_consecutive_lateral_connections']
    p_lateral_connection = trial['p_lateral_connection']
    num_lateral_connection_tries_per_unit = trial['num_lateral_connection_tries_per_unit']
    learning_rate = trial['learning_rate']
    epochs = trial['epochs']
    batch_size = trial['batch_size']

    cerebros = SimpleCerebrosRandomSearch(
        unit_type=DenseUnit,
        input_shapes=INPUT_SHAPES,
        output_shapes=OUTPUT_SHAPES,
        training_data=training_x,
        labels=train_labels,
        validation_split=0.35,
        direction='maximize',
        metric_to_rank_by='val_top_1',
        minimum_levels=2,
        maximum_levels=5,
        minimum_units_per_level=1,
        maximum_units_per_level=4,
        minimum_neurons_per_unit=1,
        maximum_neurons_per_unit=4,
        activation=activation,
        final_activation='softmax',
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
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1'),
                 tf.keras.metrics.TopKCategoricalAccuracy(
                 k=2, name='top_2'),
                 tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3')],
        epochs=epochs,
        patience=7,
        project_name=f"{PROJECT_NAME}_meta_{meta_trial_number}",
        use_multiprocessing_for_multiple_neural_networks=False,  # pull this param
        model_graphs='model_graphs',
        batch_size=batch_size,
        meta_trial_number=meta_trial_number)
    result = cerebros.run_random_search()
    print("result extracted from cerebros")
    print(result)
    print(type(result))
    return result


trial = generate_trial(0)
result = run_param_permutation(trial)
print(f"Final result was: {result}")
print(f"of type {type(result)}")

# trials=[generate_trial(i) for i in np.arange(
#     NUMBER_OF_BATCHES_OF_TRIALS * NUMBER_OF_TRAILS_PER_BATCH)]
# with Pool(2) as p:
# p.map(run_param_permutation, trials)

# trial_number = 0
# for i in np.arange(NUMBER_OF_BATCHES_OF_TRIALS):
#     batch_processes = []
#     for j in np.arange(NUMBER_OF_TRAILS_PER_BATCH):
#         trial_0 = generate_trial(trial_number)
#         trial_number += 1
#         process = Process(target=run_param_permutation(trial_0))
#         batch_processes.append(process)
#     for process in batch_processes:
#         process.start()
#     for process in batch_processes:
#         process.join()
#     results = [process.returnValue() for process in batch_processes]
#     print(f"Raw results")
#     print(results)
#     results_as_numpy = np.array(results)
#     best_result = results_as_numpy.max()
#     print(f"Best result is: {best_result}")
