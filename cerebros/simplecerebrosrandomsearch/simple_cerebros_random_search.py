import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from cerebros.denseautomlstructuralcomponent.\
    dense_automl_structural_component \
    import DenseAutoMlStructuralComponent, DenseLateralConnectivity, \
    zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid
from cerebros.units.units import Unit, InputUnit, FinalDenseUnit
from cerebros.neuralnetworkfuture.neural_network_future \
    import NeuralNetworkFuture, RealNeuronNeuralNetworkFuture
# from cmdutil.cmdutil import run_command
from multiprocessing import Process, Lock
import os
from gc import collect
from shutil import rmtree


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
                self.chart_network_graph: bool: default: False,
                                 Whether or not Cerebros will create visualizations of the neural
                                 network graphs. When recursively running Cerebros for hyperparameter
                                 tuning, we recommned the default of False. When running a single, easily repeatable
                                 training task that will not run for long, or when debugging a failed run with an error
                                 in the network graph, it may provide useful information.   
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
                 validation_data: tuple=None,
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
                 gradient_accumulation_steps=1,
                 meta_trial_number=0,
                 base_models=[''],
                 train_data_dtype=tf.float32,
                 chart_network_graph: bool=False,
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
        self.validation_data = validation_data
        self.patience = patience
        self.project_name = project_name
        self.oracle_table = f'{self.project_name}_oracle'
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps=gradient_accumulation_steps
        self.meta_trial_number = meta_trial_number
        self.base_models = base_models
        self.best_model_path = ""
        self.train_data_dtype = train_data_dtype
        self.chart_network_graph = chart_network_graph
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
        print(f"SimpleCerebrosRandomSearch.input_shapes: {self.input_shapes}")
        nnf = NeuralNetworkFuture(
            input_shapes=self.input_shapes,
            output_shapes=self.output_shapes,
            neural_network_spec=spec,
            project_name=self.project_name,
            trial_number=self.trial_number,
            subtrial_number=self.subtrial_number,
            base_models=self.base_models,
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
            model_graph_file=model_graph_file,
            train_data_dtype=self.train_data_dtype,
            gradient_accumulation_steps=self.gradient_accumulation_steps
            )
        tf.keras.backend.clear_session()
        collect()
        nnf.materialize()
        nnf.compile_neural_network()
        neural_network = nnf.materialized_neural_network
        print(nnf.materialized_neural_network.summary())
        if self.chart_network_graph:
            nnf.get_graph()

        history = neural_network.fit(x=self.training_data,
                                     y=self.labels,
                                     epochs=self.epochs,
                                     batch_size=self.batch_size,
                                     # callbacks=[early_stopping,
                                     #            tensor_board],
                                     validation_split=self.validation_split,
                                     validation_data=self.validation_data)
        oracle_0 = pd.DataFrame(history.history)

        model_architectures_folder = f"{self.project_name}/model_architectures"
        neural_network_spec_file = f"{model_architectures_folder}/tr_{str(self.trial_number).zfill(16)}_subtrial_{str(subtrial_number).zfill(16)}.txt"
        print(f"this is neural_network_spec_file {neural_network_spec_file}")
        with open(neural_network_spec_file, 'w') as f:
            f.write(str(spec))
        next_model_name =\
            f"{self.project_name}/models/tr_{str(self.trial_number).zfill(16)}_subtrial_{str(subtrial_number).zfill(16)}.keras"\
            .lower()
        neural_network.save(next_model_name)
        oracle_0['trial_number'] = self.trial_number
        oracle_0['subtrial_number'] = subtrial_number
        oracle_0['model_name'] = next_model_name
        print(f"returning trial {self.trial_number} oracles")
        print(oracle_0)
        oracle_0.to_csv(f'{self.project_name}/oracle.csv',
                        index=False,
                        header=self.needs_oracle_header,
                        mode='a')
        
        self.needs_oracle_header = False
        neural_network = None
        tf.keras.backend.clear_session()
        collect()
        return 0

    def run_random_search(self):
        processes = []
        for i in tqdm(np.arange(self.number_of_architecture_moities_to_try),
                      desc="Global task progress",
                      colour="#16ceeb"):

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
        def has_valid_metric(num):
            try:
                float(num)
                return True
            except Exception as exc:
                print(exc)
                return False
        # ~ pd.to_numeric(x['a'], errors="coerce").astype(float).isna()
        # rows_having_a_valid_metric = oracles[self.metric_to_rank_by].apply(lambda x: has_valid_metric(x))
        rows_having_a_valid_metric = ~ pd.to_numeric(oracles[self.metric_to_rank_by], errors="coerce").isna()
        oracles_having_valid_metrics = oracles[rows_having_a_valid_metric]
        
        if self.direction == "maximize" or self.direction == "max":
            best = float(oracles_having_valid_metrics[self.metric_to_rank_by].astype(float).max())
            # best = float(oracles[oracles[self.metric_to_rank_by]
            #              != self.metric_to_rank_by]
            #              [self.metric_to_rank_by].astype(float).max())
        else:
            print(f"metric_to_rank_by is: '{self.metric_to_rank_by}'")
            print(
                f"Type of metric_to_rank_by is: {str(type(self.metric_to_rank_by))}")
            best = float(oracles_having_valid_metrics[self.metric_to_rank_by].astype(float).min())
            # best = float(oracles[oracles[self.metric_to_rank_by]
            #                      != self.metric_to_rank_by]
            #              [self.metric_to_rank_by].astype(float).min())
        print(f"Best result this trial was: {best}")
        print(f"Type of best result: {type(best)}")
        self.best_model_path =\
            oracles[oracles[self.metric_to_rank_by]
                    == best]['model_name'].values[0]
        print(f"Best model name: {self.best_model_path}")
        return best

    def purge_model_storage(self) -> None:
        """Slates all cached models. 
        Recommended when running in a container without a mounted volume.
        It is recommened to use an artifiact registry to accession the best model.
        """
        model_cache_path = f"{self.project_name}/models"
        rmtree(model_cache_path)


    def purge_models_except_best_model(self) -> None:
        """
        Recommended when running in a container without a mounted volume and building models that take considerable time to reproduce.
        It is recommened to use an artifiact registry to accession the best model, but this will preserve a redundant
        copy in case accessioning it to a registry is unsuccessful.
        """
        if not self.best_model_path:
            return ValueError("The function purge_models_except_best_model was called prematurely: self.best_model_path is not set, maining there is no 'Best model'.")
        model_cache_path = f"{self.project_name}/models"
        files_path_obj = os.listdir(model_cache_path)
        files_str = [str(p) for p in files_path_obj]
        print("Files in model cache:")
        for file in files_str:
            model_file_path = f"{model_cache_path}/{file}"
            print(f"  {model_file_path}")
            if model_file_path != self.best_model_path:
                print(f"Removing: {model_file_path}")
                os.remove(model_file_path)
            # Temp debug code:
            else:
                print(f"Not removing {model_file_path}")


    def get_best_model(self, purge_model_storage_files=0) -> tf.keras.Model:
        """Returns the best model from this meta-trial. 
        Optionally, purges cache of models stored on disk.

        Params:
            - purge_model_storage_files Union[str, int]
                - Set to 0: Does not purge the cached modelsl, just returns the best model.
                - Set to 1: Purges all models except the best model found.
                - Set to "slate": Removes all models, whether the best or otherwise.
        When running ephemeral trials in a container without a mounted volume (to prevent 
        memory pressure accumulating from ephemeral files in memory) or are otherwise working
        with hard disk space limitations, we recommend setting this:
            - 'slate': if you are working on models that are quick to reproduce and an accidental model loss is not problematic as long as you have the parameters to reproduce it approximately.
            - 1: If you are are workign on models that take considerable time to reproduce a given model or a small performance difference from another model from the same parameters is problematic.
            - 0 If you have unlimited disk space and are not in a container or in one with a suitable mounted volume.  
        """
        best_model = tf.keras.models.load_model(self.best_model_path)
        if  purge_model_storage_files == 1:
            self.purge_models_except_best_model()
        elif purge_model_storage_files == "slate":
            self.purge_model_storage()
        elif purge_model_storage_files == 0:
            pass
        else:
            raise ValueError("The paramerter purge_model_storage_files in the method get_best_model() has 3 values: 0 (Don't purge),1 (Purge all but the best model), 'slate' (remove all cached models) ")
        return best_model

# ->


# class CerebrosRealNeuronNetwork(DenseAutoMlStructuralComponent,
#                                 DenseLateralConnectivity):
#     """CerebrosDenseAutoML: The kingpin class for this package. This will
#     recursively create multiple Cerebros neural networks, train and evaluate
#     them.

#     Args:
#                  unit_type: Unit,
#                                   The type of units.units.___ obuect that the
#                                   in the body of the neural networks created
#                                   will consist of. Ususlly will be a
#                                   DenseUnit object.
#                  input_shapes: list,
#                                   List of tuples representing the shape
#                                   of input(s) to the network.
#                  minimum_levels: int,
#                                   Basically the minimum deprh of the neural
#                                   networks to be tried.
#                  maximum_levels: int,
#                                   Basically the maximum deprh of the neural
#                                   networks to be tried.
#                  minimum_units_per_level: int,
#                                   Cerebros neural networks consist of muntiple
#                                   instances of what we historically would
#                                   rergard as Dense Layers arranges in parallel
#                                   on each "Level" of the network. Each connect
#                                   to randomly selected latyers downstream. Each
#                                   DenseUnit will be a Dense layer in the level.
#                                   This parameter controlls the minimum number
#                                   of units (Dense layers) each level may
#                                   consist of.
#                  maximum_units_per_level: int,
#                                   Cerebros neural networks consist of muntiple
#                                   instances of what we historically would
#                                   rergard as Dense Layers arranges in parallel
#                                   on each "Level" of the network. Each connect
#                                   to randomly selected latyers downstream. Each
#                                   DenseUnit will be a Dense layer in the level.
#                                   This parameter controlls the maximum number
#                                   of units (Dense layers) each Level may
#                                   consist of.
#                  minimum_neurons_per_unit: int,
#                                   This is the mininmum number of neurons each
#                                   Dense unit (Dense layer) nested in any Level
#                                   may consist of.
#                  maximum_neurons_per_unit: int,
#                                   This is the mininmum number of neurons each
#                                   Dense unit (Dense layer) nested in any Level
#                                   may consist of.
#                  activation: str: defaults to 'elu',
#                                   keras activation for dense unit
#                  number_of_architecture_moities_to_try: int: defaults to 7,
#                                   The auto-ml will select this many
#                                   permutations of architectural moities, in
#                                   other words (number of levels, number of
#                                   Units or Dense Layers per level for each
#                                   level.
#                  minimum_skip_connection_depth: int: defaults to 1,
#                                   The DenseLevels will be randomly connected
#                                   to others in Levels below it. These
#                                   connections will not be made exclusively to
#                                   DenseUnits in its immediate successor Level.
#                                   Some may be 2,3,4,5, ... Levels down. This
#                                   parameter controls the minimum depth of
#                                   connection allowed for connections not at
#                                   the immediate successor Level.
#                  maximum_skip_connection_depth: int: defaults to 7,
#                                   The DenseLevels will be randomly connected
#                                   to others in Levels below it. These
#                                   connections will not be made exclusively to
#                                   DenseUnits in its immediate successor Level.
#                                   Some may be 2,3,4,5, ... Levels down. This
#                                   parameter controls the minimum depth of
#                                   connection allowed for connections not at
#                                   the immediate successor Level.
#                  predecessor_level_connection_affinity_factor_first: int: defaults to 5,
#                                   The number of randomly selected Input Unit
#                                   connestions which the first Dense Level's
#                                   Units will make to randomly selected Input
#                                   units per input unit. In other words if
#                                   there are n input units, each first Level
#                                   Dense unit will select predecessor_level_connection_affinity_factor_first * n
#                                   InputUnits to connect to (WITH REPLACEMENT).
#                                   Hence this may be > 1, as a given
#                                   InputUnit may be chosen multiple times.
#                                   For example, with this set to 5, and where
#                                   were 2 connections, then each Dense
#                                   unit will make 10 connections to randomly
#                                   selected Input Units. If there were
#                                   10 inputs, and this were set to 0.5, then
#                                   the each Dense unit on the first layer would
#                                   connect to 5 randomly selected InputUnits.
#                  predecessor_level_connection_affinity_factor_first_rounding_rule: str: defaults to 'ceil',
#                                   Since the numnber of randomly selected
#                                   connections to an InputUnit that a given
#                                   Dense unit on the first layer makes is
#                                   [number of InputUnits] *  predecessor_level_connection_affinity_factor_first,
#                                   then this will usually come out to a floating
#                                   point number. The number of upstream
#                                   connections to make is a discrete variable,
#                                   so this float must be cast as an integer.
#                                   This supports "floor" and "ceil" as the
#                                   options with "ceil" as the recommended
#                                   option, because it can never ruturn 0,
#                                   which would throw and error
#                                   (It would create a disjointed graph).
#                  predecessor_level_connection_affinity_factor_main: float: defaults to 0.7,
#                                   For the second and subsequent DenseLevels,
#                                   if its immediate predecessor level has
#                                   n DenseUnits, the DenseUnits on this layer will randomly select
#                                   predecessor_level_connection_affinity_factor_main * n
#                                   DenseUnits form its immediate predecessor
#                                   Level to connect to. The selection is
#                                   WITH REPLACEMENT and a given DenseUnit may
#                                   be selected multiple times.
#                  predecessor_level_connection_affinity_factor_main_rounding_rule: str: defaults to 'ceil',
#                                  The number of connections calculated by
#                                  predecessor_level_connection_affinity_factor_main * n
#                                  will usually come out to a floating point
#                                  the number of DenseUnits  in its predecessor
#                                  Level to connect to is a discrete variable,
#                                  hence this must be cast to an integer. The
#                                  options to do this are "floor" and "ceil",
#                                  with "ceil" being the generally recommended
#                                  option, as it will never return 0. 0 will
#                                  create a disjointed graph and throw an error.
#                  predecessor_level_connection_affinity_factor_decay_main: object: defaults to zero_7_exp_decay,
#                                  It is likely that the connections with the
#                                  imemdiate predecessor, grandparent, and
#                                  great - grandparent predecessors will be more
#                                  (or less) infuential than more distant
#                                  predecessors. This parameter allows any
#                                  function that accepts an integer and returns
#                                  a floating point or integer to be used to
#                                  decrease (or increase) the number of
#                                  connections made between the k th layer and
#                                  its nthe predecessor, where n = 1 is its
#                                  GRANDPARENT predecesor level, and n-2 is its great - grandparent level. What this does exactly: The kth layer will make
#                                  predecessor_level_connection_affinity_factor_main
#                  seed: int: defaults to 8675309,
#                  max_consecutive_lateral_connections: int: defaults to 7,
#                                  The maximum number of consecutive Units on a Level that make
#                                  a lateral connection. Setting this too high can cause exploding /
#                                  vanishing gradients and internal covariate shift. Setting this too
#                                  low may miss some patterns the network may otherwise pick up.
#                  gate_after_n_lateral_connections: int: defaults to 3,
#                                  After this many kth consecutive lateral connecgtions, gate the
#                                  output of the k + 1 th DenseUnit before creating a subsequent
#                                  lateral connection.
#                  gate_activation_function: object: defaults to simple_sigmoid,
#                                  Which activation function to gate the output of one DenseUnit
#                                  before making a lateral connection.
#                  p_lateral_connection: float: defaults to 0.97,
#                                  The probability of the first given DenseUnit on a level making a
#                                  lateral connection with the second DenseUnit.
#                  p_lateral_connection_decay: object: defaults to zero_95_exp_decay,
#                                  A function that descreases or increases the probability of a
#                                  lateral connection being made with a subsequent DenseUnit.
#                                  Accepts an unsigned integer x. returns a floating point number that
#                                  will be multiplied by p_lateral_connection where x is the number
#                                  of subsequent connections after the first
#                 num_lateral_connection_tries_per_unit: int: defaults to 1,
#                  *args,
#                  **kwargs

#     """

#     def __init__(
#                  self,
#                  unit_type: Unit,
#                  input_shapes: list,
#                  output_shapes: list,
#                  training_data: list,
#                  labels: list,
#                  validation_split: float,
#                  direction: str,
#                  metric_to_rank_by: str,
#                  minimum_levels: int,
#                  maximum_levels: int,
#                  minimum_units_per_level: int,
#                  maximum_units_per_level: int,
#                  minimum_axon_neurons: int,
#                  maximum_axon_neurons: int,
#                  axon_activation: str,
#                  min_n_dendrites: int,
#                  max_n_dendrites: int,
#                  dendrite_activation: str,
#                  final_activation=None,
#                  number_of_architecture_moities_to_try=1,
#                  number_of_tries_per_architecture_moity=1,
#                  minimum_skip_connection_depth=1,
#                  maximum_skip_connection_depth=7,
#                  predecessor_level_connection_affinity_factor_first=5,
#                  predecessor_level_connection_affinity_factor_first_rounding_rule='ceil',
#                  predecessor_level_connection_affinity_factor_main=0.7,
#                  predecessor_level_connection_affinity_factor_main_rounding_rule='ceil',
#                  predecessor_level_connection_affinity_factor_decay_main=zero_7_exp_decay,
#                  seed=8675309,
#                  max_consecutive_lateral_connections=7,
#                  gate_after_n_lateral_connections=3,
#                  gate_activation_function=simple_sigmoid,
#                  p_lateral_connection=.97,
#                  p_lateral_connection_decay=zero_95_exp_decay,
#                  num_lateral_connection_tries_per_unit=1,
#                  learning_rate=0.005,
#                  loss="mse",
#                  metrics=[tf.keras.metrics.RootMeanSquaredError()],
#                  epochs=7,
#                  patience=7,
#                  project_name='cerebros-auto-ml-test',
#                  batch_size=200,
#                  meta_trial_number=0,
#                  base_models=[''],
#                  *args,
#                  **kwargs):

#         # self.num_processes = int(np.max([1, np.ceil(cpu_count() * .3)]))
#         #self.db_name = f"sqlite:///{project_name}/oracles.sqlite"
#         # self.conn = sqlite3.connect(db_name)
#         if not os.path.exists(project_name):
#             os.makedirs(project_name)
#         models_dir = f"{project_name}/models"
#         if not os.path.exists(models_dir):
#             os.makedirs(models_dir)
#         model_graphs_dir = f"{project_name}/model_graphs"
#         if not os.path.exists(model_graphs_dir):
#             os.makedirs(model_graphs_dir)
#         model_architectures_dir = f"{project_name}/model_architectures"
#         if not os.path.exists(model_architectures_dir):
#             os.makedirs(model_architectures_dir)
#         if minimum_levels > maximum_levels:
#             raise ValueError("It doesn't make sense to have minimum_levels > "
#                              "maximum_levels ")
#         if minimum_levels < 1:
#             raise ValueError("It doesn't make sense to have less than one "
#                              "DenseLevel. minimum_levels "
#                              "should be at least 1.")
#         # Constant throughout a Talos Scon sessionor Keras Tuner search session
#         self.input_shapes = input_shapes
#         self.inputs = [{'1': InputUnit} for _ in input_shapes]
#         self.output_shapes = output_shapes
#         self.outputs = [{str(s): FinalDenseUnit} for s in output_shapes]
#         self.direction = direction
#         self.metric_to_rank_by = metric_to_rank_by
#         self.minimum_levels = minimum_levels
#         self.maximum_levels = maximum_levels
#         self.minimum_units_per_level = minimum_units_per_level
#         self.maximum_units_per_level = maximum_units_per_level
#         self.minimum_neurons_per_unit = minimum_axon_neurons
#         self.maximum_neurons_per_unit = maximum_axon_neurons
#         self.axon_activation = axon_activation
#         self.min_n_dendrites = min_n_dendrites
#         self.max_n_dendrites = max_n_dendrites
#         self.dendrite_activation = dendrite_activation
#         self.final_activation = final_activation
#         self.unit_type = unit_type
#         self.number_of_architecture_moities_to_try = number_of_architecture_moities_to_try
#         self.number_of_tries_per_architecture_moity = number_of_tries_per_architecture_moity
#         self.training_data = training_data
#         self.labels = labels

#         self.validation_split = validation_split
#         self.patience = patience
#         self.project_name = project_name
#         self.oracle_table = f'{self.project_name}_oracle'
#         self.learning_rate = learning_rate
#         self.loss = loss
#         self.metrics = metrics
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.meta_trial_number = meta_trial_number
#         self.best_model_path = ""
#         # Can be varied throughout the serch session;
#         # must be controlled internally
#         DenseAutoMlStructuralComponent.__init__(
#             self,
#             minimum_skip_connection_depth=minimum_skip_connection_depth,
#             maximum_skip_connection_depth=maximum_skip_connection_depth,
#             predecessor_level_connection_affinity_factor_first=predecessor_level_connection_affinity_factor_first,
#             predecessor_level_connection_affinity_factor_first_rounding_rule=predecessor_level_connection_affinity_factor_first_rounding_rule,
#             predecessor_level_connection_affinity_factor_main=predecessor_level_connection_affinity_factor_main,
#             predecessor_level_connection_affinity_factor_main_rounding_rule=predecessor_level_connection_affinity_factor_main_rounding_rule,
#             predecessor_level_connection_affinity_factor_decay_main=predecessor_level_connection_affinity_factor_decay_main,
#             seed=seed,
#             *args,
#             **kwargs)

#         DenseLateralConnectivity.__init__(
#             self,
#             max_consecutive_lateral_connections=max_consecutive_lateral_connections,
#             gate_after_n_lateral_connections=gate_after_n_lateral_connections,
#             gate_activation_function=gate_activation_function,
#             p_lateral_connection=p_lateral_connection,
#             p_lateral_connection_decay=p_lateral_connection_decay,
#             num_lateral_connection_tries_per_unit=num_lateral_connection_tries_per_unit)
#         self.trial_number = 0
#         self.subtrial_number = 0
#         self.base_models = base_models
#         self.neural_network_specs = []
#         self.neural_network_futures = []
#         self.needs_oracle_header = True

#     def pick_num_units(self):
#         return np.random.choice(
#             jnp.arange(self.minimum_units_per_level,
#                        self.maximum_units_per_level + 1))

#     def parse_prototype_for_level(self,
#                                   num_units_this_level):
#         units_neuron_choices =\
#             [np.random.choice(jnp.arange(self.minimum_neurons_per_unit,
#                                          self.maximum_neurons_per_unit + 1))
#              for _ in jnp.arange(num_units_this_level)]
#         level_prototype =\
#             [{f"{units}": self.unit_type}
#              for units in units_neuron_choices]
#         return level_prototype

#     def parse_neural_network_structural_spec_random(self):

#         __neural_network_spec = {"0": self.inputs}
#         self.num_levels =\
#             np.random.choice(jnp.arange(self.minimum_levels,
#                                         self.maximum_levels + 1))
#         # __oracle_entry['num_levels'] = self.num_levels
#         last_level = int(jnp.arange(self.num_levels)[-1])
#         for i in jnp.arange(self.num_levels):
#             level_index = str(i + 1)  # 0 is taken by InputLevel
#             num_units_this_level =\
#                 self.pick_num_units()
#             for j in jnp.arange(num_units_this_level):
#                 if int(i) != last_level:
#                     __neural_network_spec[str(level_index)] =\
#                         self.parse_prototype_for_level(num_units_this_level)
#                 else:
#                     __neural_network_spec[str(int(i) + 1)] =\
#                         self.outputs

#         self.__neural_network_spec = __neural_network_spec

#     def get_neural_network_spec(self):
#         return self.__neural_network_spec

#     def run_moity_permutations(self, spec, subtrial_number, lock):
#         model_graph_file = f"{self.project_name}/model_graphs/tr_{str(self.trial_number).zfill(16)}_subtrial_{str(subtrial_number).zfill(16)}.html"
#         #with STRATEGY.scope():
#         nnf = RealNeuronNeuralNetworkFuture(
#             input_shapes=self.input_shapes,
#             output_shapes=self.output_shapes,
#             neural_network_spec=spec,
#             axon_activation=self.axon_activation,
#             min_n_dendrites=self.min_n_dendrites,
#             max_n_dendrites=self.max_n_dendrites,
#             dendrite_activation=self.dendrite_activation,
#             project_name=self.project_name,
#             trial_number=self.trial_number,
#             subtrial_number=self.subtrial_number,
#             base_models=self.base_models,
#             final_activation=self.final_activation,
#             minimum_skip_connection_depth=self.minimum_skip_connection_depth,
#             maximum_skip_connection_depth=self.maximum_skip_connection_depth,
#             predecessor_level_connection_affinity_factor_first=self.predecessor_level_connection_affinity_factor_first,
#             predecessor_level_connection_affinity_factor_first_rounding_rule=self.predecessor_level_connection_affinity_factor_first_rounding_rule,
#             predecessor_level_connection_affinity_factor_main=self.predecessor_level_connection_affinity_factor_main,
#             predecessor_level_connection_affinity_factor_main_rounding_rule=self.predecessor_level_connection_affinity_factor_main_rounding_rule,
#             predecessor_level_connection_affinity_factor_decay_main=self.predecessor_level_connection_affinity_factor_decay_main,
#             seed=self.seed,
#             max_consecutive_lateral_connections=self.max_consecutive_lateral_connections,
#             gate_after_n_lateral_connections=self.gate_after_n_lateral_connections,
#             gate_activation_function=self.gate_activation_function,
#             p_lateral_connection=self.p_lateral_connection,
#             p_lateral_connection_decay=self.p_lateral_connection_decay,
#             num_lateral_connection_tries_per_unit=self.num_lateral_connection_tries_per_unit,
#             learning_rate=self.learning_rate,
#             loss=self.loss,
#             metrics=self.metrics,
#             model_graph_file=model_graph_file
#             )
#         nnf.materialize()
#         nnf.compile_neural_network()
#         neural_network = nnf.materialized_neural_network
#         print(nnf.materialized_neural_network.summary())
#         nnf.get_graph()

#         history = neural_network.fit(x=self.training_data,
#                                      y=self.labels,
#                                      epochs=self.epochs,
#                                      batch_size=self.batch_size,
#                                      # callbacks=[early_stopping,
#                                      #            tensor_board],
#                                      validation_split=self.validation_split)
#         oracle_0 = pd.DataFrame(history.history)

#         model_architectures_folder = f"{self.project_name}/model_architectures"
#         neural_network_spec_file = f"{model_architectures_folder}/tr_{str(self.trial_number).zfill(16)}_subtrial_{str(subtrial_number).zfill(16)}.txt"
#         print(f"this is neural_network_spec_file {neural_network_spec_file}")
#         with open(neural_network_spec_file, 'w') as f:
#             f.write(str(spec))
#         next_model_name =\
#             f"{self.project_name}/models/tr_{str(self.trial_number).zfill(16)}_subtrial_{str(subtrial_number).zfill(16)}"\
#             .lower()
#         neural_network.save(next_model_name)
#         oracle_0['trial_number'] = self.trial_number
#         oracle_0['subtrial_number'] = subtrial_number
#         oracle_0['model_name'] = next_model_name
#         print(f"returning trial {self.trial_number} oracles")
#         print(oracle_0)
#         oracle_0.to_csv(f'{self.project_name}/oracle.csv',
#                         index=False,
#                         header=self.needs_oracle_header,
#                         mode='a')
#         self.needs_oracle_header = False
#         return 0

#     def run_random_search(self):
#         processes = []
#         for i in np.arange(self.number_of_architecture_moities_to_try):
#             self.parse_neural_network_structural_spec_random()
#             spec = self.get_neural_network_spec()

#             oracles = []
#             processes = []
#             subtrial_number = 0
#             lock = Lock()
#             for i in np.arange(
#                     self.number_of_tries_per_architecture_moity):
#                 processes.append(
#                     Process(target=self.run_moity_permutations(spec=spec, subtrial_number=subtrial_number, lock=lock)))
#                 subtrial_number += 1
#                 self.seed += 1
#             for p in processes:
#                 p.start()
#             for p in processes:
#                 p.join()
#             # final_oracles = pd.concat(oracles, ignore_index=False)
#             # if self.direction == "maximize":
#             #     return float(final_oracles[self.metric_to_rank_by].values.max())
#             # elif self.direction == "minimize":
#             #     return float(final_oracles[self.metric_to_rank_by].values.min())
#             # raise ValueError("direction must be 'maximize' or 'minimize'.")

#             self.trial_number += 1
#         oracles = pd.read_csv(f'{self.project_name}/oracle.csv')
#         print(oracles.columns)
#         print(f"metric_to_rank_by is: '{self.metric_to_rank_by}'")
#         print(
#             f"Type of metric_to_rank_by is: {str(type(self.metric_to_rank_by))}")
#         if self.direction == "maximize" or self.direction == "max":

#             best = float(oracles[oracles[self.metric_to_rank_by]
#                          != self.metric_to_rank_by]
#                          [self.metric_to_rank_by].astype(float).max())
#         else:
#             print(f"metric_to_rank_by is: '{self.metric_to_rank_by}'")
#             print(
#                 f"Type of metric_to_rank_by is: {str(type(self.metric_to_rank_by))}")
#             best = float(oracles[oracles[self.metric_to_rank_by]
#                                  != self.metric_to_rank_by]
#                          [self.metric_to_rank_by].astype(float).min())
#         print(f"Best result this trial was: {best}")
#         print(f"Type of best result: {type(best)}")
#         self.best_model_path =\
#             oracles[oracles[self.metric_to_rank_by]
#                     == best]['model_name'].values[0]
#         print(f"Best medel name: {self.best_model_path}")
#         return best

#     def get_best_model(self):
#         best_model = tf.keras.models.load_model(self.best_model_path)
#         return best_model
