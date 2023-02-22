

shell:

Clone the repo
`git clone https://github.com/david-thrower/cerebros-core-algorithm-alpha.git`

cd into it
`cd cerebros-core-algorithm-alpha`

install all required packages
`pip3 install -r requirements.tx`

Run the Ames housing data example:

`python3 regression-example-ames-no-preproc.py`

Let's look at the example: `regression-example-ames-no-preproc.py`, which is in the main folder of this repo:

Import packages
```python3

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
```

Set how much compute resources you want to spend (Cerebros will build and train a number of models that is the product of these 2 numbers)
```python3

NUMBER_OF_TRAILS_PER_BATCH = 2
NUMBER_OF_BATCHES_OF_TRIALS = 2
```
Set up project and load data
```python3


## Set a project name:


TIME = pendulum.now().__str__()[:16]\
    .replace('T', '_')\
    .replace(':', '_')\
    .replace('-', '_')
PROJECT_NAME = f'{TIME}_cerebros_auto_ml_test'


# Read in the data
raw_data = pd.read_csv('ames.csv')

# Rather than doing elaborate preprocessing, let's just drop all the columns
# that aren't numbers and impute 0 for anything missing

needed_cols = [
    col for col in raw_data.columns if raw_data[col].dtype != 'object']
data_numeric = raw_data[needed_cols].fillna(0).astype(float)
label = raw_data.pop('price')

# Convert to numpy
data_np = data_numeric.values

# convert to a tensor
tensor_x =\
    tf.constant(data_np)

# Since Cerebros allows multiple inputs, the inputs are a list of tenors, even if there is just 1
training_x = [tensor_x]

# Shape if the trining data [number of rows,number of columns]
INPUT_SHAPES = [training_x[i].shape[1] for i in np.arange(len(training_x))]

train_labels = [label.values]

# Labels are a list of numbers, shape is the length of it
OUTPUT_SHAPES = [1]  # [train_labels[i].shape[1]
```

Cerebros hyperparameters
```python3

# Params for Cebros training (Approximately the oprma
# discovered in a bayesian tuning study done on Katib
# for this data set)

# In distributed training set this to a random number, otherwise,
# you can just set it to 0. (it keeps file names unique when this runs multiple
# times with the same project, like we would in distributed training.)

meta_trial_number = 0  # In distributed training set this to a random number

# For the rest of these parameters, these are the tunable hyperparameters.
# We recommend searching a broad but realistic search space on these using a
# suitable tuner such as Katib on Kubeflow, Optuna, ray, etc.

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

```

Instantiate an instance of Cerebros Neural Architecture Search (NAS)
```python3

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

```

Run the Neural Architecture Search and get results back.
```python3
result = Cerebros.run_random_search()

print("Best model: (May need to re-initialize weights, and retrain with early stopping callback)")
best_model_found = Cerebros.get_best_model()
print(best_model_found.summary())

print("result extracted from Cerebros")
print(f"Final result was (val_root_mean_squared_error): {result}")

```

## Example output from this task:

- Ames housing data set, not pre-processed or scaled:
- House sell price predictions, val_rmse $169.04592895507812.
- The mean sale price in the data was $180,796.06.
- Val set RMSE was 0.00935% the mean sale price. No, there's not an extra 0 in there, yes, you are reading it right.
- There was no pre-trained base model. The data in [ames.csv](ames.csv) is the only data any of the model's weights has ever seen.

```
...
Best result this trial was: 169.04592895507812
Type of best result: <class 'float'>
Best medel name: 2023_01_12_23_42_cerebros_auto_ml_test_meta_0/models/tr_0000000000000006_subtrial_0000000000000000
Best model: (May need to re-initialize weights, and retrain with early stopping callback)
Model: "NeuralNetworkFuture_0000000000000nan_tr_6_nn_materialized"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 NeuralNetworkFuture_0000000000  [(None, 39)]        0           []                               
 000nan_tr_6_InputLevel_0000000                                                                   
 000000000_tr_6_InputUnit_00000                                                                   
 00000000000_tr_6_0_inp (InputL                                                                   
 ayer)                                                                                            

 NeuralNetworkFuture_0000000000  (None, 1560)        0           ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_InputLevel_00000000000
 000000001_tr_6_DenseUnit_00000                                  00000_tr_6_InputUnit_000000000000
 00000000001_tr_6_1_cat_ (Conca                                  0000_tr_6_0_inp[0][0]',          
 tenate)                                                          'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]']          

 NeuralNetworkFuture_0000000000  (None, 1560)        0           ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_InputLevel_00000000000
 000000001_tr_6_DenseUnit_00000                                  00000_tr_6_InputUnit_000000000000
 00000000001_tr_6_0_cat_ (Conca                                  0000_tr_6_0_inp[0][0]',          
 tenate)                                                          'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]']          

 NeuralNetworkFuture_0000000000  (None, 1560)        6240        ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000001_tr_6_DenseUnit_00000                                  00001_tr_6_DenseUnit_000000000000
 00000000001_tr_6_1_btn_ (Batch                                  0001_tr_6_1_cat_[0][0]']         
 Normalization)                                                                                   

 NeuralNetworkFuture_0000000000  (None, 1560)        6240        ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000001_tr_6_DenseUnit_00000                                  00001_tr_6_DenseUnit_000000000000
 00000000001_tr_6_0_btn_ (Batch                                  0001_tr_6_0_cat_[0][0]']         
 Normalization)                                                                                   

 NeuralNetworkFuture_0000000000  (None, 8)           12488       ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000001_tr_6_DenseUnit_00000                                  00001_tr_6_DenseUnit_000000000000
 00000000001_tr_6_1_dns_ (Dense                                  0001_tr_6_1_btn_[0][0]']         
 )                                                                                                

 NeuralNetworkFuture_0000000000  (None, 23)          35903       ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000001_tr_6_DenseUnit_00000                                  00001_tr_6_DenseUnit_000000000000
 00000000001_tr_6_0_dns_ (Dense                                  0001_tr_6_0_btn_[0][0]']         
 )                                                                                                

 NeuralNetworkFuture_0000000000  (None, 109)         0           ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_InputLevel_00000000000
 000000002_tr_6_DenseUnit_00000                                  00000_tr_6_InputUnit_000000000000
 00000000002_tr_6_1_cat_ (Conca                                  0000_tr_6_0_inp[0][0]',          
 tenate)                                                          'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]']         

 NeuralNetworkFuture_0000000000  (None, 109)         436         ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000002_tr_6_DenseUnit_00000                                  00002_tr_6_DenseUnit_000000000000
 00000000002_tr_6_1_btn_ (Batch                                  0002_tr_6_1_cat_[0][0]']         
 Normalization)                                                                                   

 NeuralNetworkFuture_0000000000  (None, 4)           440         ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000002_tr_6_DenseUnit_00000                                  00002_tr_6_DenseUnit_000000000000
 00000000002_tr_6_1_dns_ (Dense                                  0002_tr_6_1_btn_[0][0]']         
 )                                                                                                

 NeuralNetworkFuture_0000000000  (None, 144)         0           ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000002_tr_6_DenseUnit_00000                                  00002_tr_6_DenseUnit_000000000000
 00000000002_tr_6_2_cat_ (Conca                                  0002_tr_6_1_dns_[0][0]',         
 tenate)                                                          'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]']         

 NeuralNetworkFuture_0000000000  (None, 144)         576         ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000002_tr_6_DenseUnit_00000                                  00002_tr_6_DenseUnit_000000000000
 00000000002_tr_6_2_btn_ (Batch                                  0002_tr_6_2_cat_[0][0]']         
 Normalization)                                                                                   

 NeuralNetworkFuture_0000000000  (None, 11)          1595        ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000002_tr_6_DenseUnit_00000                                  00002_tr_6_DenseUnit_000000000000
 00000000002_tr_6_2_dns_ (Dense                                  0002_tr_6_2_btn_[0][0]']         
 )                                                                                                

 NeuralNetworkFuture_0000000000  (None, 151)         0           ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000002_tr_6_DenseUnit_00000                                  00002_tr_6_DenseUnit_000000000000
 00000000002_tr_6_3_cat_ (Conca                                  0002_tr_6_2_dns_[0][0]',         
 tenate)                                                          'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]']         

 NeuralNetworkFuture_0000000000  (None, 151)         604         ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000002_tr_6_DenseUnit_00000                                  00002_tr_6_DenseUnit_000000000000
 00000000002_tr_6_3_btn_ (Batch                                  0002_tr_6_3_cat_[0][0]']         
 Normalization)                                                                                   

 NeuralNetworkFuture_0000000000  (None, 11)          1672        ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000002_tr_6_DenseUnit_00000                                  00002_tr_6_DenseUnit_000000000000
 00000000002_tr_6_3_dns_ (Dense                                  0002_tr_6_3_btn_[0][0]']         
 )                                                                                                

 NeuralNetworkFuture_0000000000  (None, 250)         0           ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000002_tr_6_DenseUnit_00000                                  00002_tr_6_DenseUnit_000000000000
 00000000002_tr_6_4_cat_ (Conca                                  0002_tr_6_3_dns_[0][0]',         
 tenate)                                                          'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_3_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_3_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_3_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]']         

 NeuralNetworkFuture_0000000000  (None, 250)         1000        ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000002_tr_6_DenseUnit_00000                                  00002_tr_6_DenseUnit_000000000000
 00000000002_tr_6_4_btn_ (Batch                                  0002_tr_6_4_cat_[0][0]']         
 Normalization)                                                                                   

 NeuralNetworkFuture_0000000000  (None, 15)          3765        ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000002_tr_6_DenseUnit_00000                                  00002_tr_6_DenseUnit_000000000000
 00000000002_tr_6_4_dns_ (Dense                                  0002_tr_6_4_btn_[0][0]']         
 )                                                                                                

 NeuralNetworkFuture_0000000000  (None, 150)         0           ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_InputLevel_00000000000
 000000003_tr_6_DenseUnit_00000                                  00000_tr_6_InputUnit_000000000000
 00000000003_tr_6_1_cat_ (Conca                                  0000_tr_6_0_inp[0][0]',          
 tenate)                                                          'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_4_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_4_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_4_dns_[0][0]']         

 NeuralNetworkFuture_0000000000  (None, 150)         600         ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000003_tr_6_DenseUnit_00000                                  00003_tr_6_DenseUnit_000000000000
 00000000003_tr_6_1_btn_ (Batch                                  0003_tr_6_1_cat_[0][0]']         
 Normalization)                                                                                   

 NeuralNetworkFuture_0000000000  (None, 5)           755         ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000003_tr_6_DenseUnit_00000                                  00003_tr_6_DenseUnit_000000000000
 00000000003_tr_6_1_dns_ (Dense                                  0003_tr_6_1_btn_[0][0]']         
 )                                                                                                

 NeuralNetworkFuture_0000000000  (None, 170)         0           ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000003_tr_6_DenseUnit_00000                                  00003_tr_6_DenseUnit_000000000000
 00000000003_tr_6_2_cat_ (Conca                                  0003_tr_6_1_dns_[0][0]',         
 tenate)                                                          'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_3_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_4_dns_[0][0]']         

 NeuralNetworkFuture_0000000000  (None, 170)         680         ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000003_tr_6_DenseUnit_00000                                  00003_tr_6_DenseUnit_000000000000
 00000000003_tr_6_2_btn_ (Batch                                  0003_tr_6_2_cat_[0][0]']         
 Normalization)                                                                                   

 NeuralNetworkFuture_0000000000  (None, 9)           1539        ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000003_tr_6_DenseUnit_00000                                  00003_tr_6_DenseUnit_000000000000
 00000000003_tr_6_2_dns_ (Dense                                  0003_tr_6_2_btn_[0][0]']         
 )                                                                                                

 NeuralNetworkFuture_0000000000  (None, 131)         0           ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_InputLevel_00000000000
 000000003_tr_6_DenseUnit_00000                                  00000_tr_6_InputUnit_000000000000
 00000000003_tr_6_0_cat_ (Conca                                  0000_tr_6_0_inp[0][0]',          
 tenate)                                                          'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_3_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_3_dns_[0][0]']         

 NeuralNetworkFuture_0000000000  (None, 223)         0           ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000003_tr_6_DenseUnit_00000                                  00003_tr_6_DenseUnit_000000000000
 00000000003_tr_6_3_cat_ (Conca                                  0003_tr_6_2_dns_[0][0]',         
 tenate)                                                          'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_4_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_4_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_3_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_2_dns_[0][0]']         

 NeuralNetworkFuture_0000000000  (None, 109)         0           ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_InputLevel_00000000000
 000000002_tr_6_DenseUnit_00000                                  00000_tr_6_InputUnit_000000000000
 00000000002_tr_6_0_cat_ (Conca                                  0000_tr_6_0_inp[0][0]',          
 tenate)                                                          'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]']         

 NeuralNetworkFuture_0000000000  (None, 131)         524         ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000003_tr_6_DenseUnit_00000                                  00003_tr_6_DenseUnit_000000000000
 00000000003_tr_6_0_btn_ (Batch                                  0003_tr_6_0_cat_[0][0]']         
 Normalization)                                                                                   

 NeuralNetworkFuture_0000000000  (None, 223)         892         ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000003_tr_6_DenseUnit_00000                                  00003_tr_6_DenseUnit_000000000000
 00000000003_tr_6_3_btn_ (Batch                                  0003_tr_6_3_cat_[0][0]']         
 Normalization)                                                                                   

 NeuralNetworkFuture_0000000000  (None, 109)         436         ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000002_tr_6_DenseUnit_00000                                  00002_tr_6_DenseUnit_000000000000
 00000000002_tr_6_0_btn_ (Batch                                  0002_tr_6_0_cat_[0][0]']         
 Normalization)                                                                                   

 NeuralNetworkFuture_0000000000  (None, 14)          1848        ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000003_tr_6_DenseUnit_00000                                  00003_tr_6_DenseUnit_000000000000
 00000000003_tr_6_0_dns_ (Dense                                  0003_tr_6_0_btn_[0][0]']         
 )                                                                                                

 NeuralNetworkFuture_0000000000  (None, 14)          3136        ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000003_tr_6_DenseUnit_00000                                  00003_tr_6_DenseUnit_000000000000
 00000000003_tr_6_3_dns_ (Dense                                  0003_tr_6_3_btn_[0][0]']         
 )                                                                                                

 NeuralNetworkFuture_0000000000  (None, 14)          1540        ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000002_tr_6_DenseUnit_00000                                  00002_tr_6_DenseUnit_000000000000
 00000000002_tr_6_0_dns_ (Dense                                  0002_tr_6_0_btn_[0][0]']         
 )                                                                                                

 NeuralNetworkFuture_0000000000  (None, 311)         0           ['NeuralNetworkFuture_00000000000
 000nan_tr_6_FinalDenseLevel_00                                  00nan_tr_6_DenseLevel_00000000000
 00000000000004_tr_6_FinalDense                                  00003_tr_6_DenseUnit_000000000000
 Unit_0000000000000004_tr_6_0_c                                  0003_tr_6_0_dns_[0][0]',         
 at_ (Concatenate)                                                'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_3_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_3_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_3_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_InputLevel_00000000000
                                                                 00000_tr_6_InputUnit_000000000000
                                                                 0000_tr_6_0_inp[0][0]',          
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_4_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00002_tr_6_DenseUnit_000000000000
                                                                 0002_tr_6_3_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00003_tr_6_DenseUnit_000000000000
                                                                 0003_tr_6_1_dns_[0][0]']         

 NeuralNetworkFuture_0000000000  (None, 311)         1244        ['NeuralNetworkFuture_00000000000
 000nan_tr_6_FinalDenseLevel_00                                  00nan_tr_6_FinalDenseLevel_000000
 00000000000004_tr_6_FinalDense                                  0000000004_tr_6_FinalDenseUnit_00
 Unit_0000000000000004_tr_6_0_b                                  00000000000004_tr_6_0_cat_[0][0]'
 tn_ (BatchNormalization)                                        ]                                

 NeuralNetworkFuture_0000000000  (None, 1)           312         ['NeuralNetworkFuture_00000000000
 000nan_tr_6_FinalDenseLevel_00                                  00nan_tr_6_FinalDenseLevel_000000
 00000000000004_tr_6_FinalDense                                  0000000004_tr_6_FinalDenseUnit_00
 Unit_0000000000000004_tr_6_0_d                                  00000000000004_tr_6_0_btn_[0][0]'
 ns_ (Dense)                                                     ]                                

==================================================================================================
Total params: 84,465
Trainable params: 74,729
Non-trainable params: 9,736
__________________________________________________________________________________________________
None
result extracted from cerebros
Final result was (val_root_mean_squared_error): 169.04592895507812
```
