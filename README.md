# Cerebros AutoML (And manual ML)

The cerebros package is a Neural Architecture Search package (NAS) and library for writing manually configured neural networks that is intended to much more closely mimic a biological neurons than conventional neural network architecture strategies.

## In summary what is it and what is different:

A biological brain looks like this:

![assets/brain.png](assets/brain.png)

Multi layer perceptrons look like this:

![assets/mpl.png](assets/mlp.png)

If the goal of MLPs was to mimic how a biological neuron works, why do we still build neural networks that are structurally similar to the first prototypes from 1989? At the time, it was the closest we could get, but both hardware and software have changed since.

The goal here is to recursively generate models consisting of Levels of Dense Layers in parallel, where the Dense layers on one level randomly connect to layers on not only its subsequent Level, but multiple levels below. This may allow more complex networks to gain deeper, more granular insight on smaller data sets before internal covariate shift and vanishing, exploding gradients drive overfitting. Bear in mind that the deepest layers of a Multi - layer perceptron will have the most granular and specific information about a given data set. We have gotten a step closer to this by using single skip connections, but why not simply randomize the connectivity to numerous levels in the network's structure altogether?

What if we made a multi-layer pereceptron that looks like this:

Green triangles are Keras Input layers. Blue Squares are Keras Concatenate layers. The Pink stretched ovals are Keras Dense layers. The one stretched red oval is the networ's Output layer.

![assets/Brain-lookalike1.png](assets/Brain-lookalike1.png)

like this

![assets/Brain-lookalike2.png](assets/Brain-lookalike2.png)

and like this

![assets/Neuron-lookalike6.png](assets/Neuron-lookalike6.png)

What if we made a single-layer perceptron that looks like this:

![assets/Neuron-lookalike1.png](assets/Neuron-lookalike1.png)

## Use example:
clone the repo

`git checkout https://github.com/david-thrower/cerebros-core-algorithm-alpha.git`
`cd cerebros-core-algorithm-alpha`

install all required packages
```
pip3 install -r requirements.txt
```

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
```

Cerebros hyperparameters
```python3

# Params for a training function (Approximately the oprma
# discovered in a bayesian tuning study done on Katib)

meta_trial_number = 0  # In distributed training set this to a random number
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


print("result extracted from cerebros")

print(f"Final result was (val_root_mean_squared_error): {result}")

```
## Documentation

import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow as tf
from cerebros.denseautomlstructuralcomponent.\
    dense_automl_structural_component \
    import DenseAutoMlStructuralComponent, DenseLateralConnectivity, \
    zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid
from cerebros.units.units import Unit, InputUnit, FinalDenseUnit
from cerebros.neuralnetworkfuture.neural_network_future \
    import NeuralNetworkFuture
# from cmdutil.cmdutil import run_command
from multiprocessing import Process, Lock
import os

# import optuna
# from optuna.pruners import BasePruner
# from optuna.trial._state import TrialState
# from tensorflow.python import ipu

# Create an IPU distribution strategy
# STRATEGY = ipu.ipu_strategy.IPUStrategyV1()

#
# class ValidityPruner(BasePruner):
#
#     def prune(self, study, trial) -> bool:
#         params = trial.params
#         print(params)
#         minimum_levels = params['minimum_levels']
#         maximum_levels = params["maximum_levels"]
#         minimum_units_per_level = params['minimum_units_per_level']
#         maximum_units_per_level = params['maximum_units_per_level']
#         minimum_neurons_per_unit = params['minimum_neurons_per_unit']
#         maximum_neurons_per_unit = params['maximum_neurons_per_unit']
#         minimum_skip_connection_depth = params['minimum_skip_connection_depth']
#         maximum_skip_connection_depth = params['maximum_skip_connection_depth']

# minimum_levels = trial.suggest_int("minimum_levels", 2, 25)
# maximum_levels = trial.suggest_int("maximum_levels", 2, 25)
# minimum_units_per_level =\
#     trial.suggest_int("minimum_units_per_level",
#                       1,
#                       25)
# maximum_units_per_level =\
#     trial.suggest_int("maximum_units_per_level",
#                       1,
#                       25)
# minimum_neurons_per_unit =\
#     trial.suggest_int("minimum_neurons_per_unit",
#                       1,
#                       25)
# maximum_neurons_per_unit =\
#     trial.suggest_int("maximum_neurons_per_unit",
#                       1,
#                       25)
# minimum_skip_connection_depth =\
#     trial.suggest_int(
#         "minimum_skip_connection_depth",
#         1,
#         25)
# maximum_skip_connection_depth =\
#     trial.suggest_int("maximum_skip_connection_depth",1, 25)
# errors_for_trial = []
# if int(minimum_levels) >= int(maximum_levels):
#     errors_for_trial.\
#         append("minimum_levels must be < maximum_levels")
# if int(minimum_units_per_level) >= int(maximum_units_per_level):
#     errors_for_trial.\
#         append("minimum_units_per_level must be < "
#                "maximum_units_per_level")
# if int(minimum_neurons_per_unit) >= int(maximum_neurons_per_unit):
#     errors_for_trial.append("minimum_neurons_per_unit must be < "
#                             "maximum_neurons_per_unit")
# if int(minimum_skip_connection_depth) >= int(maximum_skip_connection_depth):
#     errors_for_trial.append("minimum_skip_connection_depth must "
#                             "be < maximum_skip_connection_depth")
# if int(maximum_skip_connection_depth) >= int(maximum_levels):
#     errors_for_trial.append("maximum_skip_connection_depth must "
#                             "be < maximum_levels")
# if len(errors_for_trial) != 0:
#     print(errors_for_trial)
#     return True
# return False


#  activation = trial.suggest_categorical("activation", ["elu", "relu"])
# learning_rate = trial.suggest_float('learning_rate', 0.00001, 0.3, log=True)
# n_estimators = trial.suggest_int("n_estimators", 50, 400)

#import sqlite3


#@jit

# plotly.express.parallel_coordinates
# pandas.plotting.parallel_coordinates

class SimpleCerebrosRandomSearch(DenseAutoMlStructuralComponent,
                                 DenseLateralConnectivity):
    """CerebrosDenseAutoML: The kingpin class for this package. This will
    recursively create multiple Cerebros neural networks, train and evaluate
    them.

    Args:
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
                 minimum_units_per_level: int,
                                  Cerebros neural networks consist of muntiple
                                  instances of what we historically would
                                  rergard as Dense Layers arranges in parallel
                                  on each "Level" of the network. Each connect
                                  to randomly selected latyers downstream. Each
                                  DenseUnit will be a Dense layer in the level.
                                  This parameter controlls the minimum number
                                  of units (Dense layers) each level may
                                  consist of.
                 maximum_units_per_level: int,
                                  Cerebros neural networks consist of muntiple
                                  instances of what we historically would
                                  rergard as Dense Layers arranges in parallel
                                  on each "Level" of the network. Each connect
                                  to randomly selected latyers downstream. Each
                                  DenseUnit will be a Dense layer in the level.
                                  This parameter controlls the maximum number
                                  of units (Dense layers) each Level may
                                  consist of.
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
                                 the number of DenseUnits  in its predecessor
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
                num_lateral_connection_tries_per_unit: int: defaults to 1,
                 *args,
                 **kwargs

    """

    def __init__(
                 self,
                 unit_type: Unit,
                 input_shapes: list,
                 output_shapes: list,
                 training_data: list,
                 labels: list,
                 validation_split: float,
                 direction: str,
                 metric_to_rank_by: str,
                 minimum_levels: int,
                 maximum_levels: int,
                 minimum_units_per_level: int,
                 maximum_units_per_level: int,
                 minimum_neurons_per_unit: int,
                 maximum_neurons_per_unit: int,
                 activation='elu',
                 final_activation=None,
                 number_of_architecture_moities_to_try=1,
                 number_of_tries_per_architecture_moity=1,
                 minimum_skip_connection_depth=1,
                 maximum_skip_connection_depth=7,
                 predecessor_level_connection_affinity_factor_first=5,
                 predecessor_level_connection_affinity_factor_first_rounding_rule='ceil',
                 predecessor_level_connection_affinity_factor_main=0.7,
                 predecessor_level_connection_affinity_factor_main_rounding_rule='ceil',
                 predecessor_level_connection_affinity_factor_decay_main=zero_7_exp_decay,
                 seed=8675309,
                 max_consecutive_lateral_connections=7,
                 gate_after_n_lateral_connections=3,
                 gate_activation_function=simple_sigmoid,
                 p_lateral_connection=.97,
                 p_lateral_connection_decay=zero_95_exp_decay,
                 num_lateral_connection_tries_per_unit=1,
                 learning_rate=0.005,
                 loss="mse",
                 metrics=[tf.keras.metrics.RootMeanSquaredError()],
                 epochs=7,
                 patience=7,
                 project_name='cerebros-auto-ml-test',
                 batch_size=200,
                 meta_trial_number=0,
                 *args,
                 **kwargs):

        # self.num_processes = int(np.max([1, np.ceil(cpu_count() * .3)]))
        #self.db_name = f"sqlite:///{project_name}/oracles.sqlite"
        # self.conn = sqlite3.connect(db_name)
        if not os.path.exists(project_name):
            os.makedirs(project_name)
        models_dir = f"{project_name}/models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_graphs_dir = f"{project_name}/model_graphs"
        if not os.path.exists(model_graphs_dir):
            os.makedirs(model_graphs_dir)
        model_architectures_dir = f"{project_name}/model_architectures"
        if not os.path.exists(model_architectures_dir):
            os.makedirs(model_architectures_dir)
        if minimum_levels > maximum_levels:
            raise ValueError("It doesn't make sense to have minimum_levels > "
                             "maximum_levels ")
        if minimum_levels < 1:
            raise ValueError("It doesn't make sense to have less than one "
                             "DenseLevel. minimum_levels "
                             "should be at least 1.")
        # Constant throughout a Talos Scon sessionor Keras Tuner search session
        self.input_shapes = input_shapes
        self.inputs = [{'1': InputUnit} for _ in input_shapes]
        self.output_shapes = output_shapes
        self.outputs = [{str(s): FinalDenseUnit} for s in output_shapes]
        self.direction = direction
        self.metric_to_rank_by = metric_to_rank_by
        self.minimum_levels = minimum_levels
        self.maximum_levels = maximum_levels
        self.minimum_units_per_level = minimum_units_per_level
        self.maximum_units_per_level = maximum_units_per_level
        self.minimum_neurons_per_unit = minimum_neurons_per_unit
        self.maximum_neurons_per_unit = maximum_neurons_per_unit
        self.activation = activation
        self.final_activation = final_activation
        self.unit_type = unit_type
        self.number_of_architecture_moities_to_try = number_of_architecture_moities_to_try
        self.number_of_tries_per_architecture_moity = number_of_tries_per_architecture_moity
        self.training_data = training_data
        self.labels = labels

        self.validation_split = validation_split
        self.patience = patience
        self.project_name = project_name
        self.oracle_table = f'{self.project_name}_oracle'
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
        self.meta_trial_number = meta_trial_number
        # Can be varied throughout the serch session;
        # must be controlled internally
        DenseAutoMlStructuralComponent.__init__(
            self,
            minimum_skip_connection_depth=minimum_skip_connection_depth,
            maximum_skip_connection_depth=maximum_skip_connection_depth,
            predecessor_level_connection_affinity_factor_first=predecessor_level_connection_affinity_factor_first,
            predecessor_level_connection_affinity_factor_first_rounding_rule=predecessor_level_connection_affinity_factor_first_rounding_rule,
            predecessor_level_connection_affinity_factor_main=predecessor_level_connection_affinity_factor_main,
            predecessor_level_connection_affinity_factor_main_rounding_rule=predecessor_level_connection_affinity_factor_main_rounding_rule,
            predecessor_level_connection_affinity_factor_decay_main=predecessor_level_connection_affinity_factor_decay_main,
            seed=seed,
            *args,
            **kwargs)

        DenseLateralConnectivity.__init__(
            self,
            max_consecutive_lateral_connections=max_consecutive_lateral_connections,
            gate_after_n_lateral_connections=gate_after_n_lateral_connections,
            gate_activation_function=gate_activation_function,
            p_lateral_connection=p_lateral_connection,
            p_lateral_connection_decay=p_lateral_connection_decay,
            num_lateral_connection_tries_per_unit=num_lateral_connection_tries_per_unit)
        self.trial_number = 0
        self.subtrial_number = 0
        self.neural_network_specs = []
        self.neural_network_futures = []
        self.needs_oracle_header = True

    def pick_num_units(self):
        return np.random.choice(
            jnp.arange(self.minimum_units_per_level,
                       self.maximum_units_per_level + 1))

    def parse_prototype_for_level(self,
                                  num_units_this_level):
        units_neuron_choices =\
            [np.random.choice(jnp.arange(self.minimum_neurons_per_unit,
                                         self.maximum_neurons_per_unit + 1))
             for _ in jnp.arange(num_units_this_level)]
        level_prototype =\
            [{f"{units}": self.unit_type}
             for units in units_neuron_choices]
        return level_prototype

    def parse_neural_network_structural_spec_random(self):

        __neural_network_spec = {"0": self.inputs}
        self.num_levels =\
            np.random.choice(jnp.arange(self.minimum_levels,
                                        self.maximum_levels + 1))
        # __oracle_entry['num_levels'] = self.num_levels
        last_level = int(jnp.arange(self.num_levels)[-1])
        for i in jnp.arange(self.num_levels):
            level_index = str(i + 1)  # 0 is taken by InputLevel
            num_units_this_level =\
                self.pick_num_units()
            for j in jnp.arange(num_units_this_level):
                if int(i) != last_level:
                    __neural_network_spec[str(level_index)] =\
                        self.parse_prototype_for_level(num_units_this_level)
                else:
                    __neural_network_spec[str(int(i) + 1)] =\
                        self.outputs

        self.__neural_network_spec = __neural_network_spec

    def get_neural_network_spec(self):
        return self.__neural_network_spec

    def run_moity_permutations(self, spec, subtrial_number, lock):
        model_graph_file = f"{self.project_name}/model_graphs/tr_{str(self.trial_number).zfill(16)}_subtrial_{str(subtrial_number).zfill(16)}.html"
        #with STRATEGY.scope():
        nnf = NeuralNetworkFuture(
            input_shapes=self.input_shapes,
            output_shapes=self.output_shapes,
            neural_network_spec=spec,
            project_name=self.project_name,
            trial_number=self.trial_number,
            subtrial_number=self.subtrial_number,
            activation=self.activation,
            final_activation=self.final_activation,
            minimum_skip_connection_depth=self.minimum_skip_connection_depth,
            maximum_skip_connection_depth=self.maximum_skip_connection_depth,
            predecessor_level_connection_affinity_factor_first=self.predecessor_level_connection_affinity_factor_first,
            predecessor_level_connection_affinity_factor_first_rounding_rule=self.predecessor_level_connection_affinity_factor_first_rounding_rule,
            predecessor_level_connection_affinity_factor_main=self.predecessor_level_connection_affinity_factor_main,
            predecessor_level_connection_affinity_factor_main_rounding_rule=self.predecessor_level_connection_affinity_factor_main_rounding_rule,
            predecessor_level_connection_affinity_factor_decay_main=self.predecessor_level_connection_affinity_factor_decay_main,
            seed=self.seed,
            max_consecutive_lateral_connections=self.max_consecutive_lateral_connections,
            gate_after_n_lateral_connections=self.gate_after_n_lateral_connections,
            gate_activation_function=self.gate_activation_function,
            p_lateral_connection=self.p_lateral_connection,
            p_lateral_connection_decay=self.p_lateral_connection_decay,
            num_lateral_connection_tries_per_unit=self.num_lateral_connection_tries_per_unit,
            learning_rate=self.learning_rate,
            loss=self.loss,
            metrics=self.metrics,
            model_graph_file=model_graph_file
            )
        nnf.materialize()
        nnf.compile_neural_network()
        neural_network = nnf.materialized_neural_network
        print(nnf.materialized_neural_network.summary())
        nnf.get_graph()

        history = neural_network.fit(x=self.training_data,
                                     y=self.labels,
                                     epochs=self.epochs,
                                     batch_size=self.batch_size,
                                     # callbacks=[early_stopping,
                                     #            tensor_board],
                                     validation_split=self.validation_split)
        oracle_0 = pd.DataFrame(history.history)

        model_architectures_folder = f"{self.project_name}/model_architectures"
        neural_network_spec_file = f"{model_architectures_folder}/tr_{str(self.trial_number).zfill(16)}_subtrial_{str(subtrial_number).zfill(16)}.txt"
        print(f"this is neural_network_spec_file {neural_network_spec_file}")
        with open(neural_network_spec_file, 'w') as f:
            f.write(str(spec))
        next_model_name =\
            f"{self.project_name}/models/tr_{str(self.trial_number).zfill(16)}_subtrial_{str(subtrial_number).zfill(16)}"
        neural_network.save(next_model_name)
        oracle_0['trial_number'] = self.trial_number
        oracle_0['subtrial_number'] = subtrial_number
        print(f"returning trial {self.trial_number} oracles")
        print(oracle_0)
        oracle_0.to_csv(f'{self.project_name}/oracle.csv',
                        index=False,
                        header=self.needs_oracle_header,
                        mode='a')
        self.needs_oracle_header = False
        return 0

    def run_random_search(self):
        processes = []
        for i in np.arange(self.number_of_architecture_moities_to_try):
            self.parse_neural_network_structural_spec_random()
            spec = self.get_neural_network_spec()

            oracles = []
            processes = []
            subtrial_number = 0
            lock = Lock()
            for i in np.arange(
                    self.number_of_tries_per_architecture_moity):
                processes.append(
                    Process(target=self.run_moity_permutations(spec=spec, subtrial_number=subtrial_number, lock=lock)))
                subtrial_number += 1
                self.seed += 1
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            # final_oracles = pd.concat(oracles, ignore_index=False)
            # if self.direction == "maximize":
            #     return float(final_oracles[self.metric_to_rank_by].values.max())
            # elif self.direction == "minimize":
            #     return float(final_oracles[self.metric_to_rank_by].values.min())
            # raise ValueError("direction must be 'maximize' or 'minimize'.")

            self.trial_number += 1
        oracles = pd.read_csv(f'{self.project_name}/oracle.csv')
        print(oracles.columns)
        print(f"metric_to_rank_by is: '{self.metric_to_rank_by}'")
        print(
            f"Type of metric_to_rank_by is: {str(type(self.metric_to_rank_by))}")
        if self.direction == "maximize" or self.direction == "max":
            best = float(oracles[oracles[self.metric_to_rank_by]
                         != self.metric_to_rank_by]
                         [self.metric_to_rank_by].astype(float).max())
        else:
            print(f"metric_to_rank_by is: '{self.metric_to_rank_by}'")
            print(
                f"Type of metric_to_rank_by is: {str(type(self.metric_to_rank_by))}")
            best = float(oracles[oracles[self.metric_to_rank_by]
                                 != self.metric_to_rank_by]
                         [self.metric_to_rank_by].astype(float).min())
        print(f"Best result this trial was: {best}")
        print(f"Type of best result: {type(best)}")
        return best


Here is how this was done:

We start with some basic structural components:

The CerebrosDenseAutoML: The core auto-ML that recursively creates the neural networks, vicariously through the NeuralNetworkFuture object:

NeuralNetworkFuture:
  - A data structure that is essentially a wrapper around the whole neural network system. You will note the word "Future" in the name of this data structure. This is for a reason. The solution to the problem of recursively parsing neural networks having topologies similar to the connectivity between biological neurons involves a chicken before the egg problem. Specifically this was that randomly assigning neural connectivity will create some errors and disjointed graphs, which once created can't be corrected without starting over. The probability of a given graph passing on the first try is very low, especially if you are building a complex network, nonetheless, the random connections are the key to making this work. The solution was to create a Futures objects that first planned how many levels of Dense layers would exist in the network, how many Dense layers each level would consist of and how many neurons each would consist of, allowed the random connections to be selected, but not materialized. Then it applies a protocol to detect and resolve any inconsistencies, disjointed connections, etc in the planned connectivities before they are materialized. Lastly, once the network's connectivity has been validated, it then materializes the  Dense layers of the neural network per the connectivities planned.

Level:
  - A data structure adding a new layer of abstraction above the concept of a Dense layer, which is a wrapper consisting of multiple instances of a future for what we historically called a Dense layer in a neural network. A NeuralNetworkFuture has many Layers, and a layer belongs to a NeuralNetworkFuture.
Unit:
  - A data structure which is a future for a single Dense layer. A Level has many Units, and a Unit belongs to a Level.

![assets/Cerebros.png](assets/Cerebros.png)

Here are the steps to the process:

0. Some nomenclature:
  1.1. k referrs to the Level number being immediately discussed.
  1.2. l referres to the number of DenseUnits the kth Levle has.
  1.3. k-1 refers to the immediate predecessor Level (Parent Level) number of the kth level.
  1.4 n refers to the number of DenseUnits the kth Level's parent predecessor has being mentioned has.
1. CerebrosDenseAutoML.get_networks_for_trials() instantiates a user defined number of NeuralNetworkFuture objects.
  1.1. CerebrosDenseAutoML.parse_neural_network_structural_spec_random()
    1.1.1. A random unsigned integer in a user defined range is chosen for the number of DenseLevels which the network will consist of (depth of the network).
    1.1.2. For each Level, a random unsigned integer in a user defined range is chosen for the number of Units that the layer will consist of.
    1.1.3. For each unit, a random unsigned integer in a user defined range is chosen for the number of neurons that Dense unit will consist of. Ultimately each DenseUnit will parse a Dense layer in the network.
    1.1.4. This high-level specification for the neural network (but not edges) will be parsed into a dictionary called a neural_network_spec.
  1.2. A NeuralNetworkFuture will be Instantiated, taking as an argument, the neural_network_spec as an argument.
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

1. 7. License: Licensed under the general terms of the Apache license, but with the following exclusions. The following uses and anything like this is prohibited:
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
    10. Any use in an AI system that is inherently designed to avoid contact from customers, employees, applicants, citizens, or otherwise makes decisions that significantly affect a person's life or finances without human review of ALL decisions made by said system having an unfavorable impact on a person.
        1. Example of acceptable uses under this term:
        2. An IVR or email routing system that predicts which department a customer's inquiry should be routed to.
        3. Examples of unacceptable uses under this term:
        4. An IVR system that is designed to make it cumbersome for a customer to reach a human representative at a company (e.g. the system has no option to reach a human representative or the option is in a nested layer of a multi - layer menu of options).
        5. Email screening applications that only allow selected categories of email from known customers, employees, constituents, etc to appear in a business or government representative's email inbox, blindly discarding or obfuscating all other inquiries.
        6. These or anything reasonably regarded as similar to these are prohibited uses of this codebase AND ANY DERIVATIVE WORK. Litigation will result upon discovery of any such violations.
2. Acknowledgments:
        1. My Jennifer and my stepkids who have chosen to stay around and have rode out quite a storm because of my career in science.
        2. O'Malley, et. al. For Keras Tuner
        3. Mingxing Tan, Quoc V. Le for EfficientNet.
        4. My colleagues who I work with every day.
        5. Tensorflow, Keras, Kubeflow, Kale, Optuna, and Ray open source communities.
        6. Google Cloud Platform, Arikto, Canonical, and Paperspace and their support staff for the commercil compute and ML OPS platforms used.
