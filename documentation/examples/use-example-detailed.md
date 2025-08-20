

From a shell on a suitable Linux machine (tested on Ubuntu 22.04):

Clone the repo

`git clone https://github.com/david-thrower/cerebros-core-algorithm-alpha.git`

cd into it: `cd cerebros-core-algorithm-alpha`

install all required packages: `pip3 install -r requirements.txt`

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

LABEL_COLUMN = 'price'

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
    col for col in raw_data.columns 
    if raw_data[col].dtype != 'object' 
    and col != LABEL_COLUMN]
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

meta_trial_number = 0  # In distributed training set this to a random number
activation = 'swish'
predecessor_level_connection_affinity_factor_first = 0.506486683067576
predecessor_level_connection_affinity_factor_main = 1.6466748663373876
max_consecutive_lateral_connections = 35
p_lateral_connection = 3.703218275217572
num_lateral_connection_tries_per_unit = 12
learning_rate = 0.02804912925494706
epochs = 130
batch_size = 78
maximum_levels = 4
maximum_units_per_level = 3
maximum_neurons_per_unit = 3

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
        minimum_levels=4,
        maximum_levels=maximum_levels,
        minimum_units_per_level=2,
        maximum_units_per_level=maximum_units_per_level,
        minimum_neurons_per_unit=3,
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

## Example output from this task (Tail of the logs):

- Ames housing data set, not pre-processed or scaled, non-numerical columns dropped:
- House sell price predictions, val_rmse $24866.93.
- The mean sale price in the data was $180,796.06.
- Val set RMSE was 13.7% of the mean sale price.
- In other words, on average, the model’s predictions were within about 14% of the actual sale price.
- There was no pre-trained base model used. The data in ames.csv which was selected for training is the only data any of the model's weights have ever seen.


```
...
# (Ignore the timestamps on the left ..., these were added to the output by the Github Actions Runner)
2025-08-19T20:35:10.8849137Z [1m47/47[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 887308224.0000 - root_mean_squared_error: 29787.7207 - val_loss: 766242432.0000 - val_root_mean_squared_error: 27681.0859
2025-08-19T20:35:10.8854003Z Epoch 82/91
2025-08-19T20:35:10.9022325Z 
2025-08-19T20:35:10.9527105Z [1m 1/47[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 986979328.0000 - root_mean_squared_error: 31416.2285
2025-08-19T20:35:11.0033735Z [1m23/47[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 1192501632.0000 - root_mean_squared_error: 34416.1328
2025-08-19T20:35:11.0775289Z [1m46/47[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 2ms/step - loss: 1060636224.0000 - root_mean_squared_error: 32444.6543
2025-08-19T20:35:11.0776444Z [1m47/47[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 892061312.0000 - root_mean_squared_error: 29867.3945 - val_loss: 833299200.0000 - val_root_mean_squared_error: 28866.9238
2025-08-19T20:35:11.0781821Z Epoch 83/91
2025-08-19T20:35:11.0954052Z 
2025-08-19T20:35:11.1474500Z [1m 1/47[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 844657408.0000 - root_mean_squared_error: 29062.9922
2025-08-19T20:35:11.1977702Z [1m24/47[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 575624384.0000 - root_mean_squared_error: 23870.5234 
2025-08-19T20:35:11.2747741Z [1m46/47[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 2ms/step - loss: 651192960.0000 - root_mean_squared_error: 25384.7539
2025-08-19T20:35:11.2748898Z [1m47/47[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 772981120.0000 - root_mean_squared_error: 27802.5371 - val_loss: 770631104.0000 - val_root_mean_squared_error: 27760.2441
2025-08-19T20:35:11.2754777Z Epoch 84/91
2025-08-19T20:35:11.2933068Z 
2025-08-19T20:35:11.3450702Z [1m 1/47[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 606467968.0000 - root_mean_squared_error: 24626.5723
2025-08-19T20:35:11.3970088Z [1m22/47[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 1000504512.0000 - root_mean_squared_error: 31436.4238
2025-08-19T20:35:11.4782547Z [1m43/47[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 2ms/step - loss: 986090624.0000 - root_mean_squared_error: 31295.4766 
2025-08-19T20:35:11.4783822Z [1m47/47[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 882980032.0000 - root_mean_squared_error: 29714.9805 - val_loss: 757041600.0000 - val_root_mean_squared_error: 27514.3887
2025-08-19T20:35:11.4789050Z Epoch 85/91
2025-08-19T20:35:11.4957796Z 
2025-08-19T20:35:11.5487993Z [1m 1/47[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 381375552.0000 - root_mean_squared_error: 19528.8398
2025-08-19T20:35:11.6009232Z [1m22/47[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m0s[0m 3ms/step - loss: 795846080.0000 - root_mean_squared_error: 28093.7520 
2025-08-19T20:35:11.6930853Z [1m41/47[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 3ms/step - loss: 787023872.0000 - root_mean_squared_error: 27989.8008
2025-08-19T20:35:11.6939559Z [1m47/47[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 786045888.0000 - root_mean_squared_error: 28036.5098 - val_loss: 896475328.0000 - val_root_mean_squared_error: 29941.1973
2025-08-19T20:35:11.6941041Z Epoch 86/91
2025-08-19T20:35:11.7111568Z 
2025-08-19T20:35:11.7635720Z [1m 1/47[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 454624192.0000 - root_mean_squared_error: 21321.9180
2025-08-19T20:35:11.8136551Z [1m20/47[0m [32m━━━━━━━━[0m[37m━━━━━━━━━━━━[0m [1m0s[0m 3ms/step - loss: 923940480.0000 - root_mean_squared_error: 30079.6055 
2025-08-19T20:35:11.9051104Z [1m40/47[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 3ms/step - loss: 830049728.0000 - root_mean_squared_error: 28603.9512
2025-08-19T20:35:11.9052236Z [1m47/47[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 744700544.0000 - root_mean_squared_error: 27289.2051 - val_loss: 782694592.0000 - val_root_mean_squared_error: 27976.6797
2025-08-19T20:35:11.9057603Z Epoch 87/91
2025-08-19T20:35:11.9235719Z 
2025-08-19T20:35:11.9751298Z [1m 1/47[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 594085568.0000 - root_mean_squared_error: 24373.8711
2025-08-19T20:35:12.0255545Z [1m20/47[0m [32m━━━━━━━━[0m[37m━━━━━━━━━━━━[0m [1m0s[0m 3ms/step - loss: 659671424.0000 - root_mean_squared_error: 25545.8457 
2025-08-19T20:35:12.1138209Z [1m40/47[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 3ms/step - loss: 747576320.0000 - root_mean_squared_error: 27218.8984
2025-08-19T20:35:12.1139375Z [1m47/47[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 835048000.0000 - root_mean_squared_error: 28897.1973 - val_loss: 716810816.0000 - val_root_mean_squared_error: 26773.3242
2025-08-19T20:35:12.1145214Z Epoch 88/91
2025-08-19T20:35:12.1313742Z 
2025-08-19T20:35:12.1829957Z [1m 1/47[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 2526426112.0000 - root_mean_squared_error: 50263.5703
2025-08-19T20:35:12.2338304Z [1m24/47[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 920953664.0000 - root_mean_squared_error: 29922.1699  
2025-08-19T20:35:12.3046256Z [1m47/47[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 846584576.0000 - root_mean_squared_error: 28847.2461
2025-08-19T20:35:12.3047996Z [1m47/47[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 803368192.0000 - root_mean_squared_error: 28343.7500 - val_loss: 818222912.0000 - val_root_mean_squared_error: 28604.5957
2025-08-19T20:35:12.3052741Z Epoch 89/91
2025-08-19T20:35:12.3221197Z 
2025-08-19T20:35:12.3738580Z [1m 1/47[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 360061088.0000 - root_mean_squared_error: 18975.2773
2025-08-19T20:35:12.4932307Z [1m24/47[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 639683584.0000 - root_mean_squared_error: 25087.5352 
2025-08-19T20:35:12.4934278Z [1m47/47[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 743076800.0000 - root_mean_squared_error: 27259.4375 - val_loss: 787779520.0000 - val_root_mean_squared_error: 28067.4121
2025-08-19T20:35:12.4940473Z Epoch 90/91
2025-08-19T20:35:12.5110294Z 
2025-08-19T20:35:12.5619909Z [1m 1/47[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 487451712.0000 - root_mean_squared_error: 22078.3105
2025-08-19T20:35:12.6138441Z [1m23/47[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 697663680.0000 - root_mean_squared_error: 26270.8516 
2025-08-19T20:35:12.6900931Z [1m46/47[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 2ms/step - loss: 706929984.0000 - root_mean_squared_error: 26513.6504
2025-08-19T20:35:12.6902117Z [1m47/47[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 737493376.0000 - root_mean_squared_error: 27156.8301 - val_loss: 777493632.0000 - val_root_mean_squared_error: 27883.5742
2025-08-19T20:35:12.6907360Z Epoch 91/91
2025-08-19T20:35:12.7081870Z 
2025-08-19T20:35:12.7586524Z [1m 1/47[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 640692800.0000 - root_mean_squared_error: 25311.9102
2025-08-19T20:35:12.8109641Z [1m21/47[0m [32m━━━━━━━━[0m[37m━━━━━━━━━━━━[0m [1m0s[0m 3ms/step - loss: 590127104.0000 - root_mean_squared_error: 24266.0000 
2025-08-19T20:35:12.9012681Z [1m41/47[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 3ms/step - loss: 628289152.0000 - root_mean_squared_error: 25033.5508
2025-08-19T20:35:12.9014403Z [1m47/47[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 872569152.0000 - root_mean_squared_error: 29539.2812 - val_loss: 731443904.0000 - val_root_mean_squared_error: 27045.2188
2025-08-19T20:35:13.0099681Z this is neural_network_spec_file 2025_08_19 20_31_cerebros_auto_ml_test_meta_0/model_architectures/tr_0000000000000006_subtrial_0000000000000000.txt
2025-08-19T20:35:13.0100899Z returning trial 6 oracles
2025-08-19T20:35:13.0101447Z             loss  ...                                         model_name
2025-08-19T20:35:13.0102068Z 0   4.082387e+10  ...  2025_08_19 20_31_cerebros_auto_ml_test_meta_0/...
2025-08-19T20:35:13.0102679Z 1   4.022610e+10  ...  2025_08_19 20_31_cerebros_auto_ml_test_meta_0/...
2025-08-19T20:35:13.0103108Z 2   3.876942e+10  ...  2025_08_19 20_31_cerebros_auto_ml_test_meta_0/...
2025-08-19T20:35:13.0103545Z 3   3.646121e+10  ...  2025_08_19 20_31_cerebros_auto_ml_test_meta_0/...
2025-08-19T20:35:13.0103967Z 4   3.346101e+10  ...  2025_08_19 20_31_cerebros_auto_ml_test_meta_0/...
2025-08-19T20:35:13.0104334Z ..           ...  ...                                                ...
2025-08-19T20:35:13.0104808Z 86  8.350480e+08  ...  2025_08_19 20_31_cerebros_auto_ml_test_meta_0/...
2025-08-19T20:35:13.0105238Z 87  8.033682e+08  ...  2025_08_19 20_31_cerebros_auto_ml_test_meta_0/...
2025-08-19T20:35:13.0105652Z 88  7.430768e+08  ...  2025_08_19 20_31_cerebros_auto_ml_test_meta_0/...
2025-08-19T20:35:13.0106076Z 89  7.374934e+08  ...  2025_08_19 20_31_cerebros_auto_ml_test_meta_0/...
2025-08-19T20:35:13.0106493Z 90  8.725692e+08  ...  2025_08_19 20_31_cerebros_auto_ml_test_meta_0/...
2025-08-19T20:35:13.0106761Z 
[91 rows x 7 columns]
/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()

Global task progress: 100%|[38;2;22;206;235m██████████[0m| 7/7 [03:48<00:00, 31.42s/it]
Global task progress: 100%|[38;2;22;206;235m██████████[0m| 7/7 [03:48<00:00, 32.66s/it]
Index(['loss', 'root_mean_squared_error', 'val_loss',
       'val_root_mean_squared_error', 'trial_number', 'subtrial_number',
       'model_name'],
      dtype='object')
metric_to_rank_by is: 'val_root_mean_squared_error'
Type of metric_to_rank_by is: <class 'str'>
metric_to_rank_by is: 'val_root_mean_squared_error'
Type of metric_to_rank_by is: <class 'str'>
Best result this trial was: 24866.931640625
Type of best result: <class 'float'>
Best model name: 2025_08_19 20_31_cerebros_auto_ml_test_meta_0/models/tr_0000000000000001_subtrial_0000000000000000.keras
SimpleCerebrosRandomSearch.input_shapes: [38]

Final result was (val_root_mean_squared_error): 24866.931640625
```
