"""A furutes object that coordinates the neural network's high-level
architecture"""
import warnings
from cerebros.nnfuturecomponent.neural_network_future_component \
    import NeuralNetworkFutureComponent
from cerebros.levels.levels import InputLevel, DenseLevel, FinalDenseLevel, \
    RealLevel, FinalRealLevel
import numpy as np


from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component \
    import DenseAutoMlStructuralComponent, DenseLateralConnectivity, \
    zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid

import pandas as pd
import tensorflow as tf
from pyvis.network import Network


class NeuralNetworkFuture(NeuralNetworkFutureComponent,
                          DenseAutoMlStructuralComponent,
                          DenseLateralConnectivity):
    """Takes a list of input_shapes and a neural_network_spec """

    def __init__(
             self,
             input_shapes: list,
             output_shapes: list,
             neural_network_spec: dict,
             project_name: str,
             trial_number: int,
             subtrial_number: int,
             base_models=[''],
             level_number='nan',
             activation='elu',
             final_activation=None,
             merging_strategy="concatenate",
             minimum_skip_connection_depth=1,
             maximum_skip_connection_depth=7,
             predecessor_level_connection_affinity_factor_first=5,
             predecessor_level_connection_affinity_factor_first_rounding_rule='ceil',
             predecessor_level_connection_affinity_factor_main=0.7,
             predecessor_level_connection_affinity_factor_main_rounding_rule='ceil',
             predecessor_level_connection_affinity_factor_decay_main=zero_7_exp_decay,
             predecessor_level_connection_affinity_factor_final_to_kminus1=2,
             seed=8675309,
             max_consecutive_lateral_connections=7,
             gate_after_n_lateral_connections=3,
             gate_activation_function=simple_sigmoid,
             p_lateral_connection=.97,
             p_lateral_connection_decay=zero_95_exp_decay,
             num_lateral_connection_tries_per_unit=1,
             #merging_strategy=
             learning_rate=0.005,
             loss="mse",
             metrics=[tf.keras.metrics.RootMeanSquaredError()],
             model_graph_file='test_model_graph.html',
             train_data_dtype=tf.float32,
             gradient_accumulation_steps=1,
             *args,
             **kwargs):
        print(level_number)
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.neural_network_spec = neural_network_spec
        self.base_models = base_models
        self.name = f"{project_name}_trial_{str(trial_number).zfill(16)}_subtrial_{str(subtrial_number).zfill(16)}"
        self.activation = activation
        self.final_activation = final_activation
        self.merging_strategy = merging_strategy
        self.predecessor_level_connection_affinity_factor_final_to_kminus1 =\
            predecessor_level_connection_affinity_factor_final_to_kminus1
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.uncompiled_materialized_neural_network = []
        self.compiled_materialized_neural_network = []
        self.model_graph_file = model_graph_file
        self.train_data_dtype = train_data_dtype
        self.gradient_accumulation_steps = gradient_accumulation_steps    

        # super().__init__(self,
        #                 *args,
        #                 **kwargs)
        NeuralNetworkFutureComponent.__init__(self,
                                              trial_number,
                                              level_number,
                                              *args,
                                              **kwargs)
        DenseAutoMlStructuralComponent.__init__(
            self,
            minimum_skip_connection_depth,
            maximum_skip_connection_depth,
            predecessor_level_connection_affinity_factor_first,
            predecessor_level_connection_affinity_factor_first_rounding_rule,
            predecessor_level_connection_affinity_factor_main,
            predecessor_level_connection_affinity_factor_main_rounding_rule,
            predecessor_level_connection_affinity_factor_decay_main,
            seed,
            *args,
            **kwargs)
        DenseLateralConnectivity.__init__(
            self,
            max_consecutive_lateral_connections=max_consecutive_lateral_connections,
            gate_after_n_lateral_connections=gate_after_n_lateral_connections,
            gate_actlearning_rateivation_function=gate_activation_function,
            p_lateral_connection=p_lateral_connection,
            p_lateral_connection_decay=zero_95_exp_decay,
            num_lateral_connection_tries_per_unit=num_lateral_connection_tries_per_unit)

# input_shapes: list,
# :  has_predecessors: str, has_successors: str, neural_network_future_name: str, trial_number: int, level_number: int

        levels = [InputLevel(input_shapes=self.input_shapes,
                             level_prototype=[self.neural_network_spec['0']],
                             predecessor_levels=[],
                             has_predecessors="no",
                             has_successors="yes",
                             neural_network_future_name=self.name,
                             trial_number=self.trial_number,
                             base_models=self.base_models,
                             level_number=0,
                             train_data_dtype=self.train_data_dtype)]
        print(
            f">nnf>{self.predecessor_level_connection_affinity_factor_first_rounding_rule}")
        max_k = int(np.max([int(k)
                            for k, _ in self.neural_network_spec.items()]))
        for k, v in self.neural_network_spec.items():
            print(f"k is: {k} value is: {v}")
            k1 = int(k)
            print(k1)
            if k1 == 0:
                pass
            elif k1 != max_k:
                print(f"Trying to create level {k1}")
                predecessor_levels = levels[:k1]
                print(
                    f"We think level {k1}'s predecessors are: {[l.level_number for l in predecessor_levels]}")
                ## activation: str, ##merging_strategy: str, ## level_prototype: list,
                ## predecessor_levels: list, has_predecessors: str, has_successors: str,
                # neural_network_future_name: str, trial_number: int,
                # level_number: int, minimum_skip_connection_depth=1,
                # maximum_skip_connection_depth=7,
                # predecessor_level_connection_affinity_factor_first=5,
                # predecessor_level_connection_affinity_factor_first_rounding_rule='ceil',
                # predecessor_level_connection_affinity_factor_main=0.7,
                # predecessor_level_connection_affinity_factor_main_rounding_rule='ceil',
                # predecessor_level_connection_affinity_factor_decay_main=lambda x: 0.7 * x,
                # seed=8675309

                level = DenseLevel(
                                   merging_strategy='concatenate',
                                   level_prototype=v,
                                   predecessor_levels=levels[:k1],
                                   has_predecessors="yes",
                                   has_successors='yes',
                                   neural_network_future_name=self.name,
                                   trial_number=self.trial_number,
                                   level_number=k1,
                                   activation=self.activation,
                                   minimum_skip_connection_depth=self.minimum_skip_connection_depth,
                                   maximum_skip_connection_depth=self.maximum_skip_connection_depth,
                                   predecessor_level_connection_affinity_factor_first=self.predecessor_level_connection_affinity_factor_first,
                                   predecessor_level_connection_affinity_factor_first_rounding_rule=self.predecessor_level_connection_affinity_factor_first_rounding_rule,
                                   predecessor_level_connection_affinity_factor_main=self.predecessor_level_connection_affinity_factor_main,
                                   predecessor_level_connection_affinity_factor_main_rounding_rule=self.predecessor_level_connection_affinity_factor_main_rounding_rule,
                                   predecessor_level_connection_affinity_factor_decay_main=self.predecessor_level_connection_affinity_factor_decay_main,
                                   seed=self.seed + 1,
                                   max_consecutive_lateral_connections=self.max_consecutive_lateral_connections,
                                   gate_after_n_lateral_connections=self.gate_after_n_lateral_connections,
                                   gate_activation_function=self.gate_activation_function,
                                   p_lateral_connection=self.p_lateral_connection,
                                   p_lateral_connection_decay=self.p_lateral_connection_decay,
                                   num_lateral_connection_tries_per_unit=self.num_lateral_connection_tries_per_unit)
                self.seed += 1
                level.set_possible_predecessor_connections()
                levels.append(level)  # insert(0, level)
            else:
                print(f"Trying to create Final level {k1}")
                print(f"Trying to create level {k1}")
                predecessor_levels = levels[:k1]
                print(
                    f"We think level final level {k1}'s predecessors are: {[l.level_number for l in predecessor_levels]}")
                final_level =\
                    FinalDenseLevel(
                        output_shapes=self.output_shapes,
                        merging_strategy=self.merging_strategy,
                        level_prototype=v,
                        predecessor_levels=predecessor_levels,
                        neural_network_future_name=self.name,
                        trial_number=self.trial_number,
                        level_number=k1,
                        final_activation=self.final_activation,
                        minimum_skip_connection_depth=self.maximum_skip_connection_depth,
                        maximum_skip_connection_depth=self.maximum_skip_connection_depth,
                        predecessor_level_connection_affinity_factor_first=self.predecessor_level_connection_affinity_factor_first,
                        predecessor_level_connection_affinity_factor_first_rounding_rule=self.predecessor_level_connection_affinity_factor_first_rounding_rule,
                        predecessor_level_connection_affinity_factor_main=self.predecessor_level_connection_affinity_factor_main,
                        predecessor_level_connection_affinity_factor_main_rounding_rule=self.predecessor_level_connection_affinity_factor_main_rounding_rule,
                        predecessor_level_connection_affinity_factor_decay_main=self.predecessor_level_connection_affinity_factor_decay_main,
                        predecessor_level_connection_affinity_factor_final_to_kminus1=self.predecessor_level_connection_affinity_factor_final_to_kminus1,
                        seed=self.seed + 1,
                        max_consecutive_lateral_connections=self.max_consecutive_lateral_connections,
                        gate_after_n_lateral_connections=self.gate_after_n_lateral_connections,
                        gate_activation_function=self.gate_activation_function,
                        p_lateral_connection=self.p_lateral_connection,
                        p_lateral_connection_decay=self.p_lateral_connection_decay,
                        num_lateral_connection_tries_per_unit=self.num_lateral_connection_tries_per_unit)
                self.seed += 1
                final_level.set_possible_predecessor_connections()
                levels.append(final_level)
                print("levels:")
                print([l.level_number for l in levels])
            self.levels_unmaterialized = levels

        self.predecessor_level_connection_affinity_factor_final_to_kminus1 =\
            predecessor_level_connection_affinity_factor_final_to_kminus1

        self.final_activation = final_activation

    def parse_units(self):
        last_level_number =\
            int(np.max([self.levels_unmaterialized[t].level_number
                        for t in np.arange(len(self.levels_unmaterialized))]))
        for i in np.arange(len(self.levels_unmaterialized)):
            level_0 = self.levels_unmaterialized[int(i)]
            if level_0.level_number != last_level_number:
                level_0.parse_units()
            else:
                level_0.parse_final_units()

    def set_connectivity_future_prototype(self):
        last_level_number =\
            int(np.max([self.levels_unmaterialized[t].level_number
                        for t in np.arange(len(self.levels_unmaterialized))]))
        for i in np.arange(len(self.levels_unmaterialized)):
            level_0 = self.levels_unmaterialized[int(i)]
            if level_0.level_number != last_level_number:
                level_0.set_connectivity_future_prototype()
            else:
                level_0.set_final_connectivity_future_prototype()

    def parse_meta_predecessor_connectivity(self):
        for level_0 in self.levels_unmaterialized:
            if level_0.level_number != 0:
                level_0.parse_meta_predecessor_connectivity()

    def set_successor_levels(self):
        level_id_number_index = {}
        for i in np.arange(len(self.levels_unmaterialized)):
            level_id_number_index[str(
                self.levels_unmaterialized[int(i)].level_number)] = i
        last_id_level_number = self.levels_unmaterialized[-1].level_number
        for i in np.arange(len(self.levels_unmaterialized)):
            level_0 = self.levels_unmaterialized[int(i)]
            next_level_id_number = str(level_0.level_number + 1)
            if next_level_id_number in level_id_number_index.keys():
                first_successor_level_index = level_id_number_index[next_level_id_number]
                print(
                    f"Setting levels_unmaterialized[{i}] level_number {level_0.level_number} to have first successor: levels_unmaterialized[:{first_successor_level_index}], having level_numbers of {[s.level_number for s in self.levels_unmaterialized[first_successor_level_index:]]}")
                level_0.set_successor_levels(
                    self.levels_unmaterialized[first_successor_level_index:])
            else:
                self.levels_unmaterialized[i].set_successor_levels([])
            #if level_0.level_number != last_level_number:
            #    first_successor_level = int(i + 1)

            #self.levels_unmaterialized[i].set_successor_levels(
            #    self.levels_unmaterialized[first_successor_level:])

    def detect_successor_connectivity_errors(self):
        for level_0 in self.levels_unmaterialized:
            level_0.detect_successor_connectivity_errors()

    def resolve_successor_connectivity_errors(self):
        for level_0 in self.levels_unmaterialized:
            level_0.resolve_successor_connectivity_errors()

    def util_set_predecessor_connectivity_metadata(self):
        for level_0 in self.levels_unmaterialized:
            if level_0.level_number != 0:
                level_0.util_set_predecessor_connectivity_metadata()

    def materialize(self):
        self.parse_units()
        self.set_connectivity_future_prototype()
        self.parse_meta_predecessor_connectivity()
        self.set_successor_levels()
        self.detect_successor_connectivity_errors()
        self.resolve_successor_connectivity_errors()
        self.util_set_predecessor_connectivity_metadata()

        for level in self.levels_unmaterialized:
            level.materialize()
        if len(self.levels_unmaterialized[0].parallel_units) > 1:
            materialized_neural_network_inputs =\
                [unit_0.raw_input
                 for unit_0 in self.levels_unmaterialized[0].parallel_units]
        else:
            materialized_neural_network_inputs =\
                self.levels_unmaterialized[0]\
                    .parallel_units[0].raw_input

        if len(self.levels_unmaterialized[-1].parallel_units) > 1:
            materialized_neural_network_outputs =\
                [unit_0.neural_network_layer
                 for unit_0 in self.levels_unmaterialized[-1].parallel_units]
        else:
            materialized_neural_network_outputs =\
                self.levels_unmaterialized[-1].parallel_units[0].neural_network_layer

        print("inputs")
        print(materialized_neural_network_inputs)
        print("")
        print("outputs")
        print(materialized_neural_network_outputs)
        if self.base_models == ['']:
            self.materialized_neural_network =\
                tf.keras.Model(inputs=materialized_neural_network_inputs,
                               outputs=materialized_neural_network_outputs,
                               name=f"{self.name}_nn_materialized")
        else:
            self.materialized_neural_network =\
                tf.keras.Model(inputs=materialized_neural_network_inputs,
                               outputs=materialized_neural_network_outputs,
                               name=f"{self.name}_nn_materialized")

    def compile_neural_network(self):
        if self.base_models == ['']:
            jit_compile = True
        else:
            jit_compile = False
        if not isinstance(self.gradient_accumulation_steps, int):     
            raise ValueError("gradient_accumulation_steps must be an int >= 0. You set it as {self.gradient_accumulation_steps} type {type(self.gradient_accumulation_steps)}")
        if self.gradient_accumulation_steps > 1:
            self.materialized_neural_network.compile(
                    loss=self.loss,
                    metrics=self.metrics,
                    optimizer=tf.keras.optimizers.AdamW(
                        learning_rate=self.learning_rate,
                        weight_decay=0.004,  # Add weight decay parameter
                        gradient_accumulation_steps=self.gradient_accumulation_steps
                    ),
                    jit_compile=jit_compile)
        elif self.gradient_accumulation_steps == 1:
            self.materialized_neural_network.compile(
                    loss=self.loss,
                    metrics=self.metrics,
                    optimizer=tf.keras.optimizers.AdamW(
                        learning_rate=self.learning_rate,
                        weight_decay=0.004,  # Add weight decay parameter
                    ),
                    jit_compile=jit_compile)
        else:
            raise ValueError("gradient_accumulation_steps must be an int >= 0. You set it as {self.gradient_accumulation_steps} type {type(self.gradient_accumulation_steps)}")


    def util_parse_connectivity_csv(self):

        lateral_conn_data_df =\
            pd.DataFrame({'my_name_is': [],
                          'my_level_number_is': [],
                          'my_unit_id_is': [],
                          'one_of_my_lateral_inputs_level_number_is_and_should_equal_mine': [],
                          'this_input_has_as_unit_id_should_be_less_than_mine': []})
        for level_0 in self.levels_unmaterialized:
            for unit_0 in level_0.parallel_units:
                if unit_0.level_number != 0:
                    for unit_0_0 in unit_0.lateral_connectivity_future:
                        lateral_conn_data_df_0 =\
                            pd.DataFrame({'my_name_is': [unit_0.name],
                                          'my_level_number_is': [unit_0.level_number],
                                          'my_unit_id_is': [unit_0.unit_id],
                                          'one_of_my_lateral_inputs_level_number_is_and_should_equal_mine': [unit_0_0.level_number],
                                          'this_input_has_as_unit_id_should_be_less_than_mine': [unit_0_0.unit_id]})
                        lateral_conn_data_df =\
                            pd.concat(
                                [lateral_conn_data_df, lateral_conn_data_df_0])
        lateral_conn_data_df.to_csv('lateral_conn_data_df.csv')

        predecessor_conn_data_df =\
            pd.DataFrame({'my_name_is': [],
                          'my_level_number_is': [],
                          'my_unit_id_is': [],
                          'one_of_my_predecessor_inputs_level_number_is_and_should_be_less_than_mine': [],
                          'this_input_has_as_unit_id_may_be_anything': []})
        for level_0 in self.levels_unmaterialized:
            for unit_0 in level_0.parallel_units:
                if unit_0.level_number != 0:
                    for unit_0_0 in unit_0.predecessor_connectivity_future:
                        predecessor_conn_data_df_0 =\
                            pd.DataFrame({'my_name_is': [unit_0.name],
                                          'my_level_number_is': [unit_0.level_number],
                                          'my_unit_id_is': [unit_0.unit_id],
                                          'one_of_my_predecessor_inputs_level_number_is_and_should_be_less_than_mine': [unit_0_0.level_number],
                                          'this_input_has_as_unit_id_may_be_anything': [unit_0_0.unit_id]})
                        predecessor_conn_data_df =\
                            pd.concat([predecessor_conn_data_df,
                                       predecessor_conn_data_df_0])
        predecessor_conn_data_df.to_csv('predecessor_conn_data_df.csv')

    def get_graph(self):
        net = Network(height="2000px",
                      width="1200px",
                      directed=True,
                      bgcolor="#ffffff",
                      font_color="#08fcf8")
        node_name_to_id = {}
        i = 0
        for level_0 in self.levels_unmaterialized:
            if level_0.level_number == 0:
                for unit_0 in level_0.parallel_units:
                    net.add_node(i,
                                 label=unit_0.name,
                                 shape="triangle",
                                 color="#6af784")
                    node_name_to_id[unit_0.name] = i
                    i += 1
            else:
                for unit_0 in level_0.parallel_units:
                    if unit_0.merging_strategy == "concatenate":
                        m = "cat"
                    elif unit_0.merging_strategy == "add":
                        m = "add"
                    else:
                        raise ValueError("Only add and concat are supported "
                                         "as merging_strategy at this time.")
                    merger = f"{unit_0.name}_{m}_{int(np.round(np.random.random(1)[0]*10**12))}"
                    net.add_node(i, label=merger, shape='square')
                    node_name_to_id[merger] = i
                    for unit_0_0 in [*unit_0.predecessor_connectivity_future,
                                     *unit_0.lateral_connectivity_future]:
                        net.add_edge(node_name_to_id[unit_0_0.name], i)
                    i += 1
                    if level_0 is self.levels_unmaterialized[-1]:
                        color = "#fa1919"
                        shape = "octagon"
                    else:
                        color = "#fa02b8"
                        shape = 'oval'
                    net.add_node(i,
                                 label=unit_0.name,
                                 shape=shape,
                                 color=color,
                                 size=35)
                    node_name_to_id[unit_0.name] = i
                    net.add_edge(i - 1, i)
                    i += 1
        net.toggle_physics(False)
        net.save_graph(self.model_graph_file)

# ->


class RealNeuronNeuralNetworkFuture(NeuralNetworkFutureComponent,
                                    DenseAutoMlStructuralComponent,
                                    DenseLateralConnectivity):
    """Takes a list of input_shapes and a neural_network_spec """

    def __init__(
             self,
             input_shapes: list,
             output_shapes: list,
             neural_network_spec: dict,
             axon_activation: str,
             min_n_dendrites: int,
             max_n_dendrites: int,
             dendrite_activation: str,
             project_name: str,
             trial_number: int,
             subtrial_number: int,
             base_models=[''],
             level_number='nan',
             final_activation=None,
             merging_strategy="concatenate",
             minimum_skip_connection_depth=1,
             maximum_skip_connection_depth=7,
             predecessor_level_connection_affinity_factor_first=5,
             predecessor_level_connection_affinity_factor_first_rounding_rule='ceil',
             predecessor_level_connection_affinity_factor_main=0.7,
             predecessor_level_connection_affinity_factor_main_rounding_rule='ceil',
             predecessor_level_connection_affinity_factor_decay_main=zero_7_exp_decay,
             predecessor_level_connection_affinity_factor_final_to_kminus1=2,
             seed=8675309,
             max_consecutive_lateral_connections=7,
             gate_after_n_lateral_connections=3,
             gate_activation_function=simple_sigmoid,
             p_lateral_connection=.97,
             p_lateral_connection_decay=zero_95_exp_decay,
             num_lateral_connection_tries_per_unit=1,
             #merging_strategy=
             learning_rate=0.005,
             loss="mse",
             metrics=[tf.keras.metrics.RootMeanSquaredError()],
             model_graph_file='test_model_graph.html',
             *args,
             **kwargs):
        print(level_number)
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.neural_network_spec = neural_network_spec
        self.name = f"{project_name}_trial_{str(trial_number).zfill(16)}_subtrial_{str(subtrial_number).zfill(16)}"
        self.base_models = base_models
        self.axon_activation = axon_activation
        self.min_n_dendrites = min_n_dendrites
        self.max_n_dendrites = max_n_dendrites
        self.dendrite_activation = dendrite_activation
        self.final_activation = final_activation
        self.merging_strategy = merging_strategy
        self.predecessor_level_connection_affinity_factor_final_to_kminus1 =\
            predecessor_level_connection_affinity_factor_final_to_kminus1
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.uncompiled_materialized_neural_network = []
        self.compiled_materialized_neural_network = []
        self.model_graph_file = model_graph_file

        # super().__init__(self,
        #                 *args,
        #                 **kwargs)
        NeuralNetworkFutureComponent.__init__(self,
                                              trial_number,
                                              level_number,
                                              *args,
                                              **kwargs)
        DenseAutoMlStructuralComponent.__init__(
            self,
            minimum_skip_connection_depth,
            maximum_skip_connection_depth,
            predecessor_level_connection_affinity_factor_first,
            predecessor_level_connection_affinity_factor_first_rounding_rule,
            predecessor_level_connection_affinity_factor_main,
            predecessor_level_connection_affinity_factor_main_rounding_rule,
            predecessor_level_connection_affinity_factor_decay_main,
            seed,
            *args,
            **kwargs)
        DenseLateralConnectivity.__init__(
            self,
            max_consecutive_lateral_connections=max_consecutive_lateral_connections,
            gate_after_n_lateral_connections=gate_after_n_lateral_connections,
            gate_actlearning_rateivation_function=gate_activation_function,
            p_lateral_connection=p_lateral_connection,
            p_lateral_connection_decay=zero_95_exp_decay,
            num_lateral_connection_tries_per_unit=num_lateral_connection_tries_per_unit)

# input_shapes: list,
# :  has_predecessors: str, has_successors: str, neural_network_future_name: str, trial_number: int, level_number: int

        levels = [InputLevel(input_shapes=self.input_shapes,
                             level_prototype=[self.neural_network_spec['0']],
                             predecessor_levels=[],
                             has_predecessors="no",
                             has_successors="yes",
                             neural_network_future_name=self.name,
                             trial_number=self.trial_number,
                             base_models=self.base_models,
                             level_number=0)]
        print(
            f">nnf>{self.predecessor_level_connection_affinity_factor_first_rounding_rule}")
        max_k = int(np.max([int(k)
                            for k, _ in self.neural_network_spec.items()]))
        for k, v in self.neural_network_spec.items():
            print(f"k is: {k} value is: {v}")
            k1 = int(k)
            print(k1)
            if k1 == 0:
                pass
            elif k1 != max_k:
                print(f"Trying to create level {k1}")
                predecessor_levels = levels[:k1]
                print(
                    f"We think level {k1}'s predecessors are: {[l.level_number for l in predecessor_levels]}")
                ## activation: str, ##merging_strategy: str, ## level_prototype: list,
                ## predecessor_levels: list, has_predecessors: str, has_successors: str,
                # neural_network_future_name: str, trial_number: int,
                # level_number: int, minimum_skip_connection_depth=1,
                # maximum_skip_connection_depth=7,
                # predecessor_level_connection_affinity_factor_first=5,
                # predecessor_level_connection_affinity_factor_first_rounding_rule='ceil',
                # predecessor_level_connection_affinity_factor_main=0.7,
                # predecessor_level_connection_affinity_factor_main_rounding_rule='ceil',
                # predecessor_level_connection_affinity_factor_decay_main=lambda x: 0.7 * x,
                # seed=8675309

                level = RealLevel(axon_activation=self.axon_activation,
                                  min_n_dendrites=self.min_n_dendrites,
                                  max_n_dendrites=self.max_n_dendrites,
                                  dendrite_activation=self.dendrite_activation,
                                  merging_strategy='concatenate',
                                  level_prototype=v,
                                  predecessor_levels=levels[:k1],
                                  has_predecessors="yes",
                                  has_successors='yes',
                                  neural_network_future_name=self.name,
                                  trial_number=self.trial_number,
                                  level_number=k1,
                                  minimum_skip_connection_depth=self.minimum_skip_connection_depth,
                                  maximum_skip_connection_depth=self.maximum_skip_connection_depth,
                                  predecessor_level_connection_affinity_factor_first=self.predecessor_level_connection_affinity_factor_first,
                                  predecessor_level_connection_affinity_factor_first_rounding_rule=self.predecessor_level_connection_affinity_factor_first_rounding_rule,
                                  predecessor_level_connection_affinity_factor_main=self.predecessor_level_connection_affinity_factor_main,
                                  predecessor_level_connection_affinity_factor_main_rounding_rule=self.predecessor_level_connection_affinity_factor_main_rounding_rule,
                                  predecessor_level_connection_affinity_factor_decay_main=self.predecessor_level_connection_affinity_factor_decay_main,
                                  seed=self.seed + 1,
                                  max_consecutive_lateral_connections=self.max_consecutive_lateral_connections,
                                  gate_after_n_lateral_connections=self.gate_after_n_lateral_connections,
                                  gate_activation_function=self.gate_activation_function,
                                  p_lateral_connection=self.p_lateral_connection,
                                  p_lateral_connection_decay=self.p_lateral_connection_decay,
                                  num_lateral_connection_tries_per_unit=self.num_lateral_connection_tries_per_unit)
                self.seed += 1
                level.set_possible_predecessor_connections()
                levels.append(level)  # insert(0, level)
            else:
                print(f"Trying to create Final level {k1}")
                print(f"Trying to create level {k1}")
                predecessor_levels = levels[:k1]
                print(
                    f"We think level final level {k1}'s predecessors are: {[l.level_number for l in predecessor_levels]}")
                final_level =\
                    FinalRealLevel(
                        axon_activation=self.axon_activation,
                        min_n_dendrites=self.min_n_dendrites,
                        max_n_dendrites=self.max_n_dendrites,
                        output_shapes=self.output_shapes,
                        merging_strategy=self.merging_strategy,
                        level_prototype=v,
                        predecessor_levels=predecessor_levels,
                        neural_network_future_name=self.name,
                        trial_number=self.trial_number,
                        level_number=k1,
                        final_activation=self.final_activation,
                        minimum_skip_connection_depth=self.maximum_skip_connection_depth,
                        maximum_skip_connection_depth=self.maximum_skip_connection_depth,
                        predecessor_level_connection_affinity_factor_first=self.predecessor_level_connection_affinity_factor_first,
                        predecessor_level_connection_affinity_factor_first_rounding_rule=self.predecessor_level_connection_affinity_factor_first_rounding_rule,
                        predecessor_level_connection_affinity_factor_main=self.predecessor_level_connection_affinity_factor_main,
                        predecessor_level_connection_affinity_factor_main_rounding_rule=self.predecessor_level_connection_affinity_factor_main_rounding_rule,
                        predecessor_level_connection_affinity_factor_decay_main=self.predecessor_level_connection_affinity_factor_decay_main,
                        predecessor_level_connection_affinity_factor_final_to_kminus1=self.predecessor_level_connection_affinity_factor_final_to_kminus1,
                        seed=self.seed + 1,
                        max_consecutive_lateral_connections=self.max_consecutive_lateral_connections,
                        gate_after_n_lateral_connections=self.gate_after_n_lateral_connections,
                        gate_activation_function=self.gate_activation_function,
                        p_lateral_connection=self.p_lateral_connection,
                        p_lateral_connection_decay=self.p_lateral_connection_decay,
                        num_lateral_connection_tries_per_unit=self.num_lateral_connection_tries_per_unit)
                self.seed += 1
                final_level.set_possible_predecessor_connections()
                levels.append(final_level)
                print("levels:")
                print([l.level_number for l in levels])
            self.levels_unmaterialized = levels

        self.predecessor_level_connection_affinity_factor_final_to_kminus1 =\
            predecessor_level_connection_affinity_factor_final_to_kminus1

        self.final_activation = final_activation

    def parse_units(self):
        last_level_number =\
            int(np.max([self.levels_unmaterialized[t].level_number
                        for t in np.arange(len(self.levels_unmaterialized))]))
        for i in np.arange(len(self.levels_unmaterialized)):
            level_0 = self.levels_unmaterialized[int(i)]
            if level_0.level_number != last_level_number:
                level_0.parse_units()
            else:
                level_0.parse_final_units()

    def set_connectivity_future_prototype(self):
        last_level_number =\
            int(np.max([self.levels_unmaterialized[t].level_number
                        for t in np.arange(len(self.levels_unmaterialized))]))
        for i in np.arange(len(self.levels_unmaterialized)):
            level_0 = self.levels_unmaterialized[int(i)]
            if level_0.level_number != last_level_number:
                level_0.set_connectivity_future_prototype()
            else:
                level_0.set_final_connectivity_future_prototype()

    def parse_meta_predecessor_connectivity(self):
        for level_0 in self.levels_unmaterialized:
            if level_0.level_number != 0:
                level_0.parse_meta_predecessor_connectivity()

    def set_successor_levels(self):
        level_id_number_index = {}
        for i in np.arange(len(self.levels_unmaterialized)):
            level_id_number_index[str(
                self.levels_unmaterialized[int(i)].level_number)] = i
        last_id_level_number = self.levels_unmaterialized[-1].level_number
        for i in np.arange(len(self.levels_unmaterialized)):
            level_0 = self.levels_unmaterialized[int(i)]
            next_level_id_number = str(level_0.level_number + 1)
            if next_level_id_number in level_id_number_index.keys():
                first_successor_level_index = level_id_number_index[next_level_id_number]
                print(
                    f"Setting levels_unmaterialized[{i}] level_number {level_0.level_number} to have first successor: levels_unmaterialized[:{first_successor_level_index}], having level_numbers of {[s.level_number for s in self.levels_unmaterialized[first_successor_level_index:]]}")
                level_0.set_successor_levels(
                    self.levels_unmaterialized[first_successor_level_index:])
            else:
                self.levels_unmaterialized[i].set_successor_levels([])
            #if level_0.level_number != last_level_number:
            #    first_successor_level = int(i + 1)

            #self.levels_unmaterialized[i].set_successor_levels(
            #    self.levels_unmaterialized[first_successor_level:])

    def detect_successor_connectivity_errors(self):
        for level_0 in self.levels_unmaterialized:
            level_0.detect_successor_connectivity_errors()

    def resolve_successor_connectivity_errors(self):
        for level_0 in self.levels_unmaterialized:
            level_0.resolve_successor_connectivity_errors()

    def util_set_predecessor_connectivity_metadata(self):
        for level_0 in self.levels_unmaterialized:
            if level_0.level_number != 0:
                level_0.util_set_predecessor_connectivity_metadata()

    def materialize(self):
        self.parse_units()
        self.set_connectivity_future_prototype()
        self.parse_meta_predecessor_connectivity()
        self.set_successor_levels()
        self.detect_successor_connectivity_errors()
        self.resolve_successor_connectivity_errors()
        self.util_set_predecessor_connectivity_metadata()

        for level in self.levels_unmaterialized:
            level.materialize()
        if len(self.levels_unmaterialized[0].parallel_units) > 1:
            materialized_neural_network_inputs =\
                [unit_0.neural_network_layer
                 for unit_0 in self.levels_unmaterialized[0].parallel_units]
        else:
            materialized_neural_network_inputs =\
                self.levels_unmaterialized[0]\
                    .parallel_units[0].neural_network_layer

        if len(self.levels_unmaterialized[-1].parallel_units) > 1:
            materialized_neural_network_outputs = []
            for unit_0 in self.levels_unmaterialized[-1].parallel_units:
                materialized_neural_network_outputs += unit_0.dendrites
        else:
            materialized_neural_network_outputs =\
                self.levels_unmaterialized[-1].parallel_units[0].dendrites[0]

        print("inputs")
        print(materialized_neural_network_inputs)
        print("")
        print("outputs")
        print(materialized_neural_network_outputs)

        self.materialized_neural_network =\
            tf.keras.Model(inputs=materialized_neural_network_inputs,
                           outputs=materialized_neural_network_outputs,
                           name=f"{self.name}_nn_materialized")

    def compile_neural_network(self):
        self.materialized_neural_network.compile(
            loss=self.loss,
            metrics=self.metrics,
            optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.learning_rate),
            jit_compile=True)

    def util_parse_connectivity_csv(self):

        lateral_conn_data_df =\
            pd.DataFrame({'my_name_is': [],
                          'my_level_number_is': [],
                          'my_unit_id_is': [],
                          'one_of_my_lateral_inputs_level_number_is_and_should_equal_mine': [],
                          'this_input_has_as_unit_id_should_be_less_than_mine': []})
        for level_0 in self.levels_unmaterialized:
            for unit_0 in level_0.parallel_units:
                if unit_0.level_number != 0:
                    for unit_0_0 in unit_0.lateral_connectivity_future:
                        lateral_conn_data_df_0 =\
                            pd.DataFrame({'my_name_is': [unit_0.name],
                                          'my_level_number_is': [unit_0.level_number],
                                          'my_unit_id_is': [unit_0.unit_id],
                                          'one_of_my_lateral_inputs_level_number_is_and_should_equal_mine': [unit_0_0.level_number],
                                          'this_input_has_as_unit_id_should_be_less_than_mine': [unit_0_0.unit_id]})
                        lateral_conn_data_df =\
                            pd.concat(
                                [lateral_conn_data_df, lateral_conn_data_df_0])
        lateral_conn_data_df.to_csv('lateral_conn_data_df.csv')

        predecessor_conn_data_df =\
            pd.DataFrame({'my_name_is': [],
                          'my_level_number_is': [],
                          'my_unit_id_is': [],
                          'one_of_my_predecessor_inputs_level_number_is_and_should_be_less_than_mine': [],
                          'this_input_has_as_unit_id_may_be_anything': []})
        for level_0 in self.levels_unmaterialized:
            for unit_0 in level_0.parallel_units:
                if unit_0.level_number != 0:
                    for unit_0_0 in unit_0.predecessor_connectivity_future:
                        predecessor_conn_data_df_0 =\
                            pd.DataFrame({'my_name_is': [unit_0.name],
                                          'my_level_number_is': [unit_0.level_number],
                                          'my_unit_id_is': [unit_0.unit_id],
                                          'one_of_my_predecessor_inputs_level_number_is_and_should_be_less_than_mine': [unit_0_0.level_number],
                                          'this_input_has_as_unit_id_may_be_anything': [unit_0_0.unit_id]})
                        predecessor_conn_data_df =\
                            pd.concat([predecessor_conn_data_df,
                                       predecessor_conn_data_df_0])
        predecessor_conn_data_df.to_csv('predecessor_conn_data_df.csv')

    def get_graph(self):
        net = Network(height="2000px",
                      width="1200px",
                      directed=True,
                      bgcolor="#ffffff",
                      font_color="#08fcf8")
        node_name_to_id = {}
        i = 0
        for level_0 in self.levels_unmaterialized:
            if level_0.level_number == 0:
                for unit_0 in level_0.parallel_units:
                    net.add_node(i,
                                 label=unit_0.name,
                                 shape="triangle",
                                 color="#6af784")
                    node_name_to_id[unit_0.name] = i
                    i += 1
            else:
                for unit_0 in level_0.parallel_units:
                    if unit_0.merging_strategy == "concatenate":
                        m = "cat"
                    elif unit_0.merging_strategy == "add":
                        m = "add"
                    else:
                        raise ValueError("Only add and concat are supported "
                                         "as merging_strategy at this time.")
                    merger = f"{unit_0.name}_{m}_{int(np.round(np.random.random(1)[0]*10**12))}"
                    net.add_node(i, label=merger, shape='square')
                    node_name_to_id[merger] = i
                    for unit_0_0 in [*unit_0.predecessor_connectivity_future,
                                     *unit_0.lateral_connectivity_future]:
                        net.add_edge(node_name_to_id[unit_0_0.name], i)
                    i += 1
                    if level_0 is self.levels_unmaterialized[-1]:
                        color = "#fa1919"
                        shape = "octagon"
                    else:
                        color = "#fa02b8"
                        shape = 'oval'
                    net.add_node(i,
                                 label=unit_0.name,
                                 shape=shape,
                                 color=color,
                                 size=35)
                    node_name_to_id[unit_0.name] = i
                    net.add_edge(i - 1, i)
                    i += 1
        net.toggle_physics(False)
        net.save_graph(self.model_graph_file)
