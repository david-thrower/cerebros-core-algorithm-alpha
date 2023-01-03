# Cerebros AutoML

The Cerebros package is an ultra-precise Neural Architecture Search (NAS) / AutoML that is intended to much more closely mimic biological neurons than conventional neural network architecture strategies.

## Cerebros Community Edition and Cerebros Enterprise

The Cerebros community edition provides an open-source minimum viable single parameter set NAS and also also provides an example manifest for an exhaustive Neural Architecture Search to run on Kubeflow/Katib. This is licensd for free use provided that the use is consistent with the ethical use provisions in the license described at the bottom of this page. You can easily reproduce this with the Jupyter notebook in the directory `/kubeflow-pipeline`, using the Kale Jupyter notebook extension. For a robust managed neural architecture search experience hosted on Google Cloud Platform and supported by our SLA, we recommend Cerebros Enterprise, our commercial version. Soon you will be able to sign up and immediately start using it at `https://www.cerebros.one`. In the meantime, we can set up your own Cerbros managed neural architecture search pipeline for you with a one business day turnaround. We offer consulting, demos, full service machine learning service and can provision you with your own full neural architecture search pipeline complete with automated Bayesian hyperparameter search. Contact David Thrower:`david@cerebros.one` or call us at (US area code 1) `(650) 789-4375`. Additionally, we can comlete machine learning tasks for your orgniation. Give us a call.



## In summary what is it and what is different:

A biological brain looks like this:

![assets/brain.png](assets/brain.png)

Multi layer perceptrons look like this:

![assets/mpl.png](assets/mlp.png)

If the goal of MLPs was to mimic how a biological neuron works, why do we still build neural networks that are structurally similar to the first prototypes from 1989? At the time, it was the closest we could get, but both hardware and software have changed since.

The goal here is to recursively generate models consisting of "Levels" which consist of of Dense Layers in parallel, where the Dense layers on one level randomly connect to layers on not only its subsequent Level, but multiple levels below. In addition to these randomized vertical connections, the Dense layers also connect **latrally** at random to not only their neighboing layer, but to layers multiple layers to the right of them (remember, this architectural pattern consists of "Levels" of Dense layers. The Dense layers make lateral connections tothe othr Dense layers in the same level, and vertial connections to Dense layers in their leval's successor levels). There may also be moe than one connection between a given Dense layer and another , both laterally nd vertically, which if you ave the patience to follow the example neural architectre created by the Ames housing data example, you ill see many instances where this occurs. This may allow more complex networks to gain deeper, more granular insight on smaller data sets before problems like internal covariate shift, vanishinggradients, and exploding gradients drive overfitting, zeroed out weights, and "predictions of [0 | infiniti] for all samples". Bear in mind that the deepest layers of a Multi - layer perceptron will have the most granular and specific information about a given data set.  In recent years, we got a step closer to this by using single skip connections, but why not simply randomize the connectivity to numerous levels in the network's structure altogether and add lateral connections?

What if we made a multi-layer pereceptron that looks like this: (Green triangles are Keras Input layers. Blue Squares are Keras Concatenate layers. The Pink stretched ovals are Keras Dense layers. The one stretched red oval is the networ's Output layer. It is presumed that thre isa batch normaliation layer between each Concatenate layer and Dense layer.)

![assets/Brain-lookalike1.png](assets/Brain-lookalike1.png)

... or like this:

![assets/Brain-lookalike2.png](assets/Brain-lookalike2.png)

and like this

![assets/Neuron-lookalike6.png](assets/Neuron-lookalike6.png)

What if we made a single-layer perceptron that looks like this:

![assets/Neuron-lookalike1.png](assets/Neuron-lookalike1.png)

## Use example: Try it for yourself:
clone the repo

shell:
`git checkout https://github.com/david-thrower/cerebros-core-algorithm-alpha.git`
`cd cerebros-core-algorithm-alpha`

install all required packages
```
pip3 install -r requirements.txt
```

Let's look at the eample: `regression-example-ames-no-preproc.py`, which is in the main folder of this repo:

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

# In distributed training set this to a random number, otherwise, set it to 0. (it keeps file names unique when this runs multiple times with the same project, like we would in distributed training.)
meta_trial_number = 0
activation = "elu"
predecessor_level_connection_affinity_factor_first = 15.0313
predecessor_level_connection_affinity_factor_main = 10.046
max_consecutive_lateral_connections = 23
p_lateral_connection = 0.19668
num_lateral_connection_tries_per_unit = 20
learning_rate = 0.0664
epochs = 96
batch_size = 93
```

Instantiate the Cerebros Neural Architecture Search (NAS)
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
        minimum_levels=2,
        maximum_levels=7,
        minimum_units_per_level=1,
        maximum_units_per_level=4,
        minimum_neurons_per_unit=1,
        maximum_neurons_per_unit=4,
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

Run Neural Architecture Search and get results
```python3
result = cerebros.run_random_search()

print("Best model: (May need to re-initialize weights, and retrain with early stopping callback)")
best_model_found = cerebros.get_best_model()
print(best_model_found.summary())

print("result extracted from cerebros")
print(f"Final result was (val_root_mean_squared_error): {result}")

```

## Example output from this task (Ames data set not pre-processed or scaled: House sell price predictions, val_rmse $856.25)

```
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
                                                                                                  
 NeuralNetworkFuture_0000000000  (None, 1248)        0           ['NeuralNetworkFuture_00000000000
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
                                                                 0000_tr_6_0_inp[0][0]']          
                                                                                                  
 NeuralNetworkFuture_0000000000  (None, 1248)        4992        ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000001_tr_6_DenseUnit_00000                                  00001_tr_6_DenseUnit_000000000000
 00000000001_tr_6_1_btn_ (Batch                                  0001_tr_6_1_cat_[0][0]']         
 Normalization)                                                                                   
                                                                                                  
 NeuralNetworkFuture_0000000000  (None, 4)           4996        ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000001_tr_6_DenseUnit_00000                                  00001_tr_6_DenseUnit_000000000000
 00000000001_tr_6_1_dns_ (Dense                                  0001_tr_6_1_btn_[0][0]']         
 )                                                                                                
                                                                                                  
 NeuralNetworkFuture_0000000000  (None, 1248)        0           ['NeuralNetworkFuture_00000000000
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
                                                                 0000_tr_6_0_inp[0][0]']          
                                                                                                  
 NeuralNetworkFuture_0000000000  (None, 1260)        0           ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000001_tr_6_DenseUnit_00000                                  00001_tr_6_DenseUnit_000000000000
 00000000001_tr_6_2_cat_ (Conca                                  0001_tr_6_1_dns_[0][0]',         
 tenate)                                                          'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
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
                                                                                                  
 NeuralNetworkFuture_0000000000  (None, 1248)        4992        ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000001_tr_6_DenseUnit_00000                                  00001_tr_6_DenseUnit_000000000000
 00000000001_tr_6_0_btn_ (Batch                                  0001_tr_6_0_cat_[0][0]']         
 Normalization)                                                                                   
                                                                                                  
 NeuralNetworkFuture_0000000000  (None, 1260)        5040        ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000001_tr_6_DenseUnit_00000                                  00001_tr_6_DenseUnit_000000000000
 00000000001_tr_6_2_btn_ (Batch                                  0001_tr_6_2_cat_[0][0]']         
 Normalization)                                                                                   
                                                                                                  
 NeuralNetworkFuture_0000000000  (None, 1)           1249        ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000001_tr_6_DenseUnit_00000                                  00001_tr_6_DenseUnit_000000000000
 00000000001_tr_6_0_dns_ (Dense                                  0001_tr_6_0_btn_[0][0]']         
 )                                                                                                
                                                                                                  
 NeuralNetworkFuture_0000000000  (None, 1)           1261        ['NeuralNetworkFuture_00000000000
 000nan_tr_6_DenseLevel_0000000                                  00nan_tr_6_DenseLevel_00000000000
 000000001_tr_6_DenseUnit_00000                                  00001_tr_6_DenseUnit_000000000000
 00000000001_tr_6_2_dns_ (Dense                                  0001_tr_6_2_btn_[0][0]']         
 )                                                                                                
                                                                                                  
 NeuralNetworkFuture_0000000000  (None, 494)         0           ['NeuralNetworkFuture_00000000000
 000nan_tr_6_FinalDenseLevel_00                                  00nan_tr_6_DenseLevel_00000000000
 00000000000002_tr_6_FinalDense                                  00001_tr_6_DenseUnit_000000000000
 Unit_0000000000000002_tr_6_0_c                                  0001_tr_6_0_dns_[0][0]',         
 at_ (Concatenate)                                                'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
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
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
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
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_0_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_2_dns_[0][0]',         
                                                                  'NeuralNetworkFuture_00000000000
                                                                 00nan_tr_6_DenseLevel_00000000000
                                                                 00001_tr_6_DenseUnit_000000000000
                                                                 0001_tr_6_1_dns_[0][0]']         
                                                                                                  
 NeuralNetworkFuture_0000000000  (None, 494)         1976        ['NeuralNetworkFuture_00000000000
 000nan_tr_6_FinalDenseLevel_00                                  00nan_tr_6_FinalDenseLevel_000000
 00000000000002_tr_6_FinalDense                                  0000000002_tr_6_FinalDenseUnit_00
 Unit_0000000000000002_tr_6_0_b                                  00000000000002_tr_6_0_cat_[0][0]'
 tn_ (BatchNormalization)                                        ]                                
                                                                                                  
 NeuralNetworkFuture_0000000000  (None, 1)           495         ['NeuralNetworkFuture_00000000000
 000nan_tr_6_FinalDenseLevel_00                                  00nan_tr_6_FinalDenseLevel_000000
 00000000000002_tr_6_FinalDense                                  0000000002_tr_6_FinalDenseUnit_00
 Unit_0000000000000002_tr_6_0_d                                  00000000000002_tr_6_0_btn_[0][0]'
 ns_ (Dense)                                                     ]                                
                                                                                                  
==================================================================================================
Total params: 25,001
Trainable params: 16,501
Non-trainable params: 8,500
__________________________________________________________________________________________________
None
result extracted from cerebros
Final result was (val_root_mean_squared_error): 856.2445678710938

```

## Documentation

Classes: (Meant for direct use)

```
SimpleCerebrosRandomSearch
    - Args:
                unit_type: Unit,
                                  The type of units.units.___ obuect that the
                                  in the body of the neural networks created
                                  will consist of. Ususlly will be a
                                  DenseUnit object.
                 input_shapes: list,
                                  List of tuples representing the shape
                                  of input(s) to the network.
                 minimum_levels: int,
                                  Basically the minimum deprh of the neural
                                  networks to be tried.
                 maximum_levels: int,
                                  Basically the maximum deprh of the neural
                                  networks to be tried.
        21-add-a-way-to-get-the-best-model
                 minimum_neurons_per_unit: int,
                                  This is the mininmum number of neurons each
                                  Dense unit (Dense layer) nested in any Level
                                  may consist of.
                 maximum_neurons_per_unit: int,
                                  This is the mininmum number of neurons each
                                  Dense unit (Dense layer) nested in any Level
                                  may consist of.
                 activation: str: defaults to 'elu',
                                  keras activation for dense unit
                 number_of_architecture_moities_to_try: int: defaults to 7,
                                  The auto-ml will select this many
                                  permutations of architectural moities, in
                                  other words (number of levels, number of
                                  Units or Dense Layers per level for each
                                  level.
                 minimum_skip_connection_depth: int: defaults to 1,
                                  The DenseLevels will be randomly connected
                                  to others in Levels below it. These
                                  connections will not be made exclusively to
                                  DenseUnits in its immediate successor Level.
                                  Some may be 2,3,4,5, ... Levels down. This
                                  parameter controls the minimum depth of
                                  connection allowed for connections not at
                                  the immediate successor Level.
                 maximum_skip_connection_depth: int: defaults to 7,
                                  The DenseLevels will be randomly connected
                                  to others in Levels below it. These
                                  connections will not be made exclusively to
                                  DenseUnits in its immediate successor Level.
                                  Some may be 2,3,4,5, ... Levels down. This
                                  parameter controls the minimum depth of
                                  connection allowed for connections not at
                                  the immediate successor Level.
                 predecessor_level_connection_affinity_factor_first: int: defaults to 5,
                                  The number of randomly selected Input Unit
                                  connestions which the first Dense Level's
                                  Units will make to randomly selected Input
                                  units per input unit. In other words if
                                  there are n input units, each first Level
                                  Dense unit will select predecessor_level_connection_affinity_factor_first * n
                                  InputUnits to connect to (WITH REPLACEMENT).
                                  Hence this may be > 1, as a given
                                  InputUnit may be chosen multiple times.
                                  For example, with this set to 5, and where
                                  were 2 connections, then each Dense
                                  unit will make 10 connections to randomly
                                  selected Input Units. If there were
                                  10 inputs, and this were set to 0.5, then
                                  the each Dense unit on the first layer would
                                  connect to 5 randomly selected InputUnits.
                 predecessor_level_connection_affinity_factor_first_rounding_rule: str: defaults to 'ceil',
                                  Since the numnber of randomly selected
                                  connections to an InputUnit that a given
                                  Dense unit on the first layer makes is
                                  [number of InputUnits] *  predecessor_level_connection_affinity_factor_first,
                                  then this will usually come out to a floating
                                  point number. The number of upstream
                                  connections to make is a discrete variable,
                                  so this float must be cast as an integer.
                                  This supports "floor" and "ceil" as the
                                  options with "ceil" as the recommended
                                  option, because it can never ruturn 0,
                                  which would throw and error
                                  (It would create a disjointed graph).
                 predecessor_level_connection_affinity_factor_main: float: defaults to 0.7,
                                  For the second and subsequent DenseLevels,
                                  if its immediate predecessor level has
                                  n DenseUnits, the DenseUnits on this layer will randomly select
                                  predecessor_level_connection_affinity_factor_main * n
                                  DenseUnits form its immediate predecessor
                                  Level to connect to. The selection is
                                  WITH REPLACEMENT and a given DenseUnit may
                                  be selected multiple times.
                 predecessor_level_connection_affinity_factor_main_rounding_rule: str: defaults to 'ceil',
                                 The number of connections calculated by
                                 predecessor_level_connection_affinity_factor_main * n
                                 will usually come out to a floating point
   21-add-a-way-to-get-the-best-model                              the number of DenseUnits  in its predecessor
                                 Level to connect to is a discrete variable,
                                 hence this must be cast to an integer. The
                                 options to do this are "floor" and "ceil",
                                 with "ceil" being the generally recommended
                                 option, as it will never return 0. 0 will
                                 create a disjointed graph and throw an error.
                 predecessor_level_connection_affinity_factor_decay_main: object: defaults to zero_7_exp_decay,
                                 It is likely that the connections with the
                                 imemdiate predecessor, grandparent, and
                                 great - grandparent predecessors will be more
                                 (or less) infuential than more distant
                                 predecessors. This parameter allows any
                                 function that accepts an integer and returns
                                 a floating point or integer to be used to
                                 decrease (or increase) the number of
                                 connections made between the k th layer and
                                 its nthe predecessor, where n = 1 is its
                                 GRANDPARENT predecesor level, and n-2 is its great - grandparent level. What this does exactly: The kth layer will make
                                 predecessor_level_connection_affinity_factor_main
                 seed: int: defaults to 8675309,
                 max_consecutive_lateral_connections: int: defaults to 7,
                                 The maximum number of consecutive Units on a Level that make
                                 a lateral connection. Setting this too high can cause exploding /
                                 vanishing gradients and internal covariate shift. Setting this too
                                 low may miss some patterns the network may otherwise pick up.
                 gate_after_n_lateral_connections: int: defaults to 3,
                                 After this many kth consecutive lateral connecgtions, gate the
                                 output of the k + 1 th DenseUnit before creating a subsequent
                                 lateral connection.
                 gate_activation_function: object: defaults to simple_sigmoid,
                                 Which activation function to gate the output of one DenseUnit
                                 before making a lateral connection.
                 p_lateral_connection: float: defaults to 0.97,
                                 The probability of the first given DenseUnit on a level making a
                                 lateral connection with the second DenseUnit.
                 p_lateral_connection_decay: object: defaults to zero_95_exp_decay,
                                 A function that descreases or increases the probability of a
                                 lateral connection being made with a subsequent DenseUnit.
                                 Accepts an unsigned integer x. returns a floating point number that
                                 will be multiplied by p_lateral_connection where x is the number
                                 of subsequent connections after the first
                num_lateral_connection_tries_per_unit: int: defaults to 1
                learning_rate: Keras API optimizer parameter
                loss: Keras API parameter
                metrics: Keras API parameter
                epochs: int: Keras API parameter
                patience: int Keras API parameter for early stopping callback 
                project_name: str: An arbitrary name for the projet. Must be a vaild POSIX file name.
                model_graphs='model_graphs': str: prefix for model graph image files
                batch_size: int: Keras param
                meta_trial_number: int: Arbitraty trial number that adds uniqueness to files created suring distributed tuning of tunable hyperparams.

    Methods:
        run_random_search:
            Runs a random search over the parametrs chosen and returns the best metric found.
            
            Params:
                None
            Returns:
                int: Best metric
        get_best_model:
            Params:
                None
            Returns:
                keras.Model: Best model found.

```


## How this all works:

We start with some basic structural components:

The SimpleCerebrosRandomSearch: The core auto-ML that recursively creates the neural networks, vicariously through the NeuralNetworkFuture object:

NeuralNetworkFuture:
  - A data structure that is essentially a wrapper around the whole neural network system. 
  - You will note the word "Future" in the name of this data structure. This is for a reason. The solution to the problem of recursively parsing neural networks having topologies similar to the connectivity between biological neurons involves a chicken before the egg problem. Specifically this was that randomly assigning neural connectivity will create some errors and disjointed graphs, which once created is impractical correct without starting over. Additionally, I have found that for some reson, graphs having some dead ends, but not completey disjointed train very slowly. 
  - The probability of a given graph passing on the first try is very low, especially if you are building a complex network, nonetheless, the randomized vertical and lateral, and potentially repeating connections are the key to making this all work. 
  - The solution was to create a Futures objects that first planned how many levels of Dense layers would exist in the network, how many Dense layers each level would consist of, and how many neurons each would consist of, allowing the random connections to be tentatively selected, but not materialized. Then it applies a protocol to detect and resolve any inconsistencies, disjointed connections, etc in the planned connectivities before any actual neural network componenta are  actually materialized.
  - A list of errors is compiled and another protocol appends the planned connectivity with additionl connections to fix the breaks in the connectivity. 
  - Lastly, once the network's connectivity has been validated, it then materializes the Dense layers of the neural network per the connectivities planned, resultin in a neural network ready to be trained.

Level:
  - A data structure adding a new layer of abstraction above the concept of a Dense layer, which is a wrapper consisting of multiple instances of a future for what we historically called a Dense layer in a neural network. A level will consist of multiple Dense Units, which each will materialize to a Dense Layer. Since we are makng both vertical and lateral connections, the term "Layer" loses relevance it has in the traditional sequential MLP context. The Level. A NeuralNetworkFuture has many Levels, and a Level belongs to a NeuralNetworkFuture. A Level has many Units, and a Unit belongs to a Level.
  
Unit:
  - A data structure which is a future for a single Dense layer. A Level has many Units, and a Unit belongs to a Level.

![assets/Cerebros.png](assets/Cerebros.png)

Here are the steps to the process:

0. Some nomenclature:
  1.1. k referrs to the Level number being immediately discussed.
  1.2. l referres to the number of DenseUnits the kth Level has.
  1.3. k-1 refers to the immediate predecessor Level (Parent Level) number of the kth level of the level being discussed.
  1.4 n refers to the number of DenseUnits the kth Level's parent predecessor has.
1. SimpleCerebrosRandomSearch.run_random_search().
    1. This calls SimpleCerebrosRandomSearch.parse_neural_network_structural_spec_random() which chooses the following random unsigned integers:
        1. How many Levels, the archtecture will consist of;
        2. For each Level, how many Units the levl will consist of;
        3. For each unit, how many neurons the Dense layer it will materialize will consist of.
        4. This is parsed into a dictionary as a high-level specification for the nodes, but not edges called a neural_network_spec.
        5. This will instantiate a NeuralNetworkFuture of the selected specification for number of Levels, Units per Level, and neurons per Unit, and the NeuralNetworkFuture takes the neural_network_spec as an argument.
        6. This entire logic will repeat once for each number in range of number of the number_of_architecture_moities_to_try.
        7. Step 5 will repea multiple times, once threin with the same neural_network_spec, once for each number in range number_of_tries_per_architecture_moity.
        8. All replictions are done as separate Python multiprocessing proces (multiple workers in parallel on separate processor cores).
2. **(This is a top down operation starting with InputLevel and proceeding to the last hidden layer in the network)** In each NeuralNetworkFuture, the neural_network_spec will be iterated through, instantiating a DenseLevel object for each element in the dictionary , which will be passed as the argument level_prototype. Each will be linked to the last, and each will maintain access to the same chain of Levels as the list predecessor_levels (The whole thing is essentially like a linked list, having many necessary nested elements).
3. A dictionary of possible predecessor connections will also be parsed. This is a sybmolic representation of the levels and units above it that is faster to iterate through than the actual Levels and Units objects themselves.
4. **(Add direction top down or bottom - up)** CerebrosAutoML calls each NeuralNetworkFuture object's .parse_units(), method, which recursively calls the .parse_units() belonging to each Levels object. Within each Levels object, this will iterate through its level_prototype list and will instantiate a DenseUnits object for each item and append DenseLevel.parallel_units with it.
5. CerebrosAutoML calls each NeuralNetworkFuture object's .determine_upstream_random_connectivity() method. This will recursively call each Levels object's determine_upstream_random_connectivity(). Which will trigger each layer to recursively call each of its constituent DenseUnit objects' determine_upstream_random_connectivity() method. **(This is a bottom - up operation starting with the last hidden layer and and proceeding to the InputLevel)** Each DenseUnit will calculate how many connections to make to DenseUnits in each predecessor Level and will select that many units from its possible_predecessor_connections dictionary, from each of those levels, appending each to its __predecessor_connections_future list.
6. Once the random predecessor connections have been selected, now, the system will then need to validate the DOWNSTREAM network connectivity of each Dense unit and repair any breaks that would cause a disjointed graph.  (verify that each DenseUnit has at least one connection to a SUCCESSOR Layer's DenseUnit). Here is why: If the random connections were chosen bottom - up, (which is the lesser of two evils) AND each DenseUnit will always select at least one PREDECESSOR connection (I ether validate this to throw an error if the number of connections to the immediate predecessor layer is less than 1 OR it just coerce it to 1 if 0 is calculated, with this said, then upstream connectivity is not possible to break, however, it is inherently possible that at least one predecessor unit was NOT selected by any of its successor's randomly selected connections. (especially if a low value is selected for the predecessor_level_connection_affinity_factor_main), something, I speculate may be advantageous to do, as some research has indicate that making sparse connections can outperform Dense connections, which is what we aim to do. We want not all connections to be made to the immediate successor Level. We want some to connect 2 Levels downstream, 3, ...  4, ... 5, Levels downstream ... The trouble is that the random connections that facilitate this can "leave out" a DenseUnit in a predecessor Level as not being picked at all by any DenseUnit in a SUCCESSOR level, leaving us with a dead end in the network, gence a disjointed graph. There are 3 rules this has to follow: **Rule 1:** Each DenseUnit must connect to SOMETHING upstream (PREDECESSORs) WITHIN max_skip_connection_depth layers of its immediate predecessor. (Random bottom - up assignment can't accidentally violate this rule if it always selects at least one selection, so nothing to worry about or validate here). **Rule 2:** Everything must connect to something something DOWNSTREAM (successors) within max_skip_connection_depth Levels of its level_number. **(Random bottom - up assignment will frequently leave violations of this rule behind. Where this happens, these should either be connected to a randomly selected DenseUnit max_skip_connection_depth layers below | **or a randomly selected DenseUnit residing a randomly chosen number of layers below in the range of [minimum_skip_connection_depth, maximum_skip_connection_depth] below when possible | and if necessary, to the last hidden DenseLevel or the output level)**. Now the third rule **Rule 3:** The connectivity must flow in only one direction vertically and one direction laterally (on a layer - by - layer basis). In other words, a kth Level's DenseUnit can't take its own output as one of its inputs. Nor can a kth Level's DenseUnit take its k+[any number]th successor's output as one of its inputs (because it is also a function of the kth Level's DenseUnit's own output). This would obviously be a contradiction, like filling the tank of an empty fire truck using the fire hose that draws water from its own tank.. or an empty fuel pump at a fuel station filling its own empty tank using the hose that goes in a car's gas tank, ... and gets that fuel from its own tank... Fortunately, both the logic setting the vertical connectivity and the logic setting the lateral connectivity both can't create this inconsistency, so there is nothing to worry about or validate here either. We only have to screen for and fix violations of rule 2, "every DenseUnit must connect to some (DOWNSTREAM / SUCCESSOR) DenseUnit within max_skip_connection_depth of itsself. Here is the logic for this validation and rectification:   
  5.1. This check must be done by DenseLevel objects:
  5.2 Scenario 1: **(If the kth DenseLayer is the last Layer)**:
  5.3 For each layer having a  **layer_number >= k - maximum_skip_connection_depth** (look at possible_predecessor_connections):
  5.4. For each DenseUnit in said layer, check each successor layer's Dense units' __predecessor_connections_future list for it being selected. if found: pass, else, the kth layer will add it to its own __predecessor_connections_future list.
  5.5. Scenario 2: **(If the kth DenseLayer is not the last Layer)** Do the same as scenario 1, EXCEPT, only check the layer where **its layer number == (k - maximum_skip_connection_depth)**.    
7. Lateral connectivity: For each DesnseLayer:
  7.1 For each DenseUnit:
  7.2 Calculate the number of lateral connections to make.
  7.3 Select said number of units from level_prototype where unit's unit_id is [less than | greater than (respectively based on right or left connectivity)]. There is nothing to validate here. If this is set uo to only allow right OR left connections, then this can't create a disjointed graph or contradiction. Now all connectivities are planned.
8. Materialize Dense layers. **(This is a top down operation starting with InputLevel and proceeding to the last hidden layer in the network)**
9. Create output layer.
10. compile model.
11. Fit models.
12. Iterate through the results and find best the model.  

## Open source license:

License: Licensed under the general terms of the Apache license, but with the following exclusions. The following uses and anything like this is prohibited:
    1. Military use, except explicitly authorized by the author
    2. Law enforcement use intended to aide in making decisions that lead to a anyone being  incarcerated or in any way managing an incarceration operation, or criminal prosecution operation, jail, prison, or participating in decisions that flag citizens for investigation or exclusion from public locations, whether physical or virtual.
    3. Use in committing property or violent crimes
    4. Use in any application supporting the adult films industry
    5. Use in any application supporting or in any way promoting the alcoholic beverages, firearms, and / or tobacco industries
    6. Any use supporting the trade, marketing of, or administration of prescription drugs which are commonly abused
    7. Use in a manner intended to identify or discriminate against anyone on any ethnic, ideological, religious, racial, demographic,familial status,family of origin, sex or gender, gender identity, sexual orientation, status of being a victim of any crime, having a history or present status of being a good-faith litigant, national origin(including citizenship or lawful resident status), disability, age, pregnancy, parental status, mental health, income, or socioeconomic / *credit status (which includes lawful credit, tenant, and HR screening* other than screening for criminal history).
    8. Promoting controversial services such as abortion, via any and all types of marketing, market targeting, operational,administrative, or financial support for providers of such serices.
    9. Any use supporting any operation which attempts to sway public opinion, political alignment, or purchasing habits via means such as:
        1. Misleading the public to believe that the opinions promoted by said operation are those of a different group of people than those which the campaign portrays them as being. For example, a political group attempting to cast an image that a given political alignment is that of low income rural citizens, when such is not consistent with known statistics on such population (commonly referred to as astroturfing).
        2. Leading the public to believe premises that contradict duly accepted scientific findings, implausible doctrines, or premises that are generally regarded as heretical or occult.
        3. Promoting or managing any operation profiting from dishonest or unorthodox marketing practices or marketing unorthodox products generally regarded as a junk deal to consumers or employees: (e.g. multi-level marketing operations, 'businesses' that rely on 1099 contractors not ensured a regular wage for all hours worked, companies having any full time employee paid less than $40,000 per year at the time of this writing weighted to BLS inflation, short term consumer lenders and retailers / car dealers offering credit to consumers who could not be approved for the same loan by an FDIC insured bank, operations that make sales through telemarketing or automated phone calls, non-opt-in email distribution marketing, vacation timeshare operations, etc.)
    10. Any use that supports copyright, trademark, patent, or trade secret infringement.
    11. Any use that may reasonably be deemed as negligent.
    12. Any use intended to prevent Cerebros from operating their own commercial distribution of Cerebros or any attempt to gain a de-facto monopoly on commercial or managed platform use of this or a derivitive work. 
    12. Any use in an AI system that is inherently designed to avoid contact from customers, employees, applicants, citizens, or otherwise makes decisions that significantly affect a person's life or finances without human review of ALL decisions made by said system having an unfavorable impact on a person.
        1. Example of acceptable uses under this term:
        2. An IVR or email routing system that predicts which department a customer's inquiry should be routed to.
        3. Examples of unacceptable uses under this term:
        4. An IVR system that is designed to make it cumbersome for a customer to reach a human representative at a company (e.g. the system has no option to reach a human representative or the option is in a nested layer of a multi - layer menu of options).
        5. Email screening applications that only allow selected categories of email from known customers, employees, constituents, etc to appear in a business or government representative's email inbox, blindly discarding or obfuscating all other inquiries.
        6. These or anything reasonably regarded as similar to these are prohibited uses of this codebase AND ANY DERIVATIVE WORK. Litigation will result upon discovery of any such violations.

## Acknowledgments:

1. My Jennifer and my stepkids who have chosen to stay around and have rode out quite a storm because of my career in science.
2. My son Aidyn, daughter Jenna, and my collaborators Max Morganbesser and Andres Espinosa.
3. Mingxing Tan, Quoc V. Le for EfficientNet.
4. My colleagues who I work with every day.
5. Tensorflow, Keras, Kubeflow, Kale, Optuna, Keras Tuner, and Ray open source communities and contributors.
6. Google Cloud Platform, Arikto, Canonical, and Paperspace and their support staff for the commercil compute and ML OPS platforms used.
7. Microk8s, minikube,and the core Kubernetes communities and associated projects.
