import numpy as np
import jax.numpy as jnp
import tensorflow as tf
from cerebros.nnfuturecomponent.neural_network_future_component import \
    NeuralNetworkFutureComponent

from cerebros.denseautomlstructuralcomponent.\
    dense_automl_structural_component \
    import zero_7_exp_decay, zero_95_exp_decay, \
    simple_sigmoid, \
    DenseAutoMlStructuralComponent, DenseLateralConnectivity


class Unit(NeuralNetworkFutureComponent):
    def __init__(self,
                 n_neurons: int,
                 predecessor_levels: list,
                 unit_id: int,
                 level_name: str,
                 trial_number: int,
                 level_number: int,
                 * args,
                 **kwargs):
        super().__init__(trial_number, level_number, *args, **kwargs)
        self.n_neurons = n_neurons
        self.predecessor_levels = predecessor_levels
        self.unit_id = unit_id
        self.level_name = level_name
        self.name = f"{self.level_name}_{self.name}_{self.unit_id}"
        self.successor_connectivity_errors_2d = []

    def set_successor_levels(self, successor_levels: list = []):
        self.successor_levels = successor_levels

    def detect_single_unit_errors(self,
                                  level_numbers: jnp.array,
                                  unit_ids: jnp.array):

        jnp_level_number = jnp.int32(self.level_number)
        jnp_unit_id = jnp.int32(self.unit_id)
        successor_connection_index =\
            jnp.logical_and(
                jnp.equal(level_numbers, jnp_level_number),
                jnp.equal(unit_ids, jnp_unit_id))
        if successor_connection_index.any():
            return True  # Successfully found a connection, good, termniate search
        else:
            return False  # Not found yet. Continue search if Units left to search

    def detect_successor_connectivity_errors(self):
        if self.successor_levels == []:
            print(
                f"Debug: successor_connectivity_errors_2d {self.successor_connectivity_errors_2d}")
            return True
        num_successors = len(self.successor_levels)
        for i in np.arange(num_successors):
            level_0 = self.successor_levels[i]
            for j in np.arange(len(level_0.parallel_units)):
                unit_0 = level_0.parallel_units[j]
                match_found = self.detect_single_unit_errors(
                       unit_0.meta_predecessor_connectivity_level_number,
                       unit_0.meta_predecessor_connectivity_unit_id)
                if match_found:  # and i != len(self.successor_levels) - 1:
                    return True

        print(self.level_number)
        print(self.unit_id)
        self.successor_connectivity_errors_2d.append(
            jnp.array([self.level_number,
                       self.unit_id]))


class InputUnit(Unit):
    def __init__(self,
                 input_shape: tuple,
                 unit_id: int,
                 level_name: str,
                 trial_number: int,
                 predecessor_levels=[],
                 n_neurons=1,
                 level_number=0,
                 base_models=[''],
                 train_data_dtype=tf.float32,
                 *args,
                 **kwargs):
        if isinstance(input_shape, int):
            self.input_shape = (input_shape,)
        elif isinstance(input_shape, str):
            self.input_shape = (int(input_shape),)
        else:
            _input_shape = [int(ax) for ax in input_shape]
            self.input_shape = tuple(_input_shape)
        self.neural_network_layer = []
        self.base_models = base_models
        self.train_data_dtype = train_data_dtype

        super().__init__(n_neurons,
                         predecessor_levels,
                         unit_id,
                         level_name,
                         trial_number,
                         level_number,
                         *args,
                         **kwargs)

    def materialize(self):

        self.raw_input = tf.keras.layers.Input(self.input_shape,
                                               name=f"{self.name}_inp",
                                               dtype=self.train_data_dtype)
        print(f"$$$$$$>>>>> Base model: {self.base_models[self.unit_id]}")
        print(f"InputUnit.input_shape: {self.input_shape}")
        if self.base_models != [''] and self.base_models[self.unit_id] != "":
            self.neural_network_layer =\
                self.base_models[self.unit_id](self.raw_input)
        else:
            self.neural_network_layer =\
                self.raw_input


class DenseUnit(Unit,
                DenseLateralConnectivity):

    # Add to base unit
    #         __predecessor_connections_future: list:
    #                              List of dictioaries:
    #                              The keys for each dict will be the index
    #                              position of which CURRENT layer dense_unit is
    #                              being connected upstream.
    #                              The value in the dictionary is a 2d list,
    #                              each listing an upstream dense_unit to which
    #                              it connects, represented in the format:
    #                              [layer_number of the layer the dense_unit,
    #                              index of dense_unit ]
    #                              are the index positions of the parallel dense
    #                              unit on the respective downstream layer to which
    #                              it connects.
    #

    """
    A future that will parse a tf.keras.layers.Dense object, once it
    determines which tf.keras.layers.___ objects upstream in the neural
    network which it will ingest as its as its predecessor.

    Args:
        n_neurons: int:
                    number of dense units the tf.keras.layers.Dense object
                    parsed by this object will consist of.
        activation: str:
                    The string representation of the activation
                    function to be used in the tf.keras.layers.Dense object
                    returned by this object.
        level_number: int, the layer number of the DenseLevels it belongs to (0 - indexed starting with the first Input layer).
        predecessor_levels: list: A list of DenseLevel objects that preceed
                                  the DenseLevel which this DenseUnitModule
                                  being instantiated belongs to.
        dense_unit_module_id: list, [layer_number, 0-indexed serial number of
                              DenseUnitModule objects in the DenseLevel
                              object to which this given DenseUnitModule
                              is a member of]
        maximum_skip_connection_depth: int:
                                       Maximum number of Levels above
                                       the current Level which the
                                       DenseUnitModule may directly ingest
                                       a DenseUnitModule as a predecessor.
        predecessor_level_connection_affinity_factor_first: float:
                                       range supported: [0.000...1,inf]:
                                       The approximate probability of this
                                       DenseUnitModule connecting directly to
                                       a given Input layer. This number will
                                       be divided by the number of inputs.
                                       The rounding rule will selected will
                                       either floor or ceiling the quotient.
                                       The resulting unsigned integer will
                                       be the number of random connections
                                       made to randomly selected members of
                                       the Inputs level. Many models will
                                       have 1 input, and each will will
                                       connect to
                                       all inputs. Setting this parameter to
                                       2 will make at least 2 connections to
                                       each input.
        predecessor_level_connection_affinity_factor_first_rounding_rule: str:
                                       options are "floor" and "ciling":
                                       whether to use np.floor or np.ceil to
                                       cast the quotient of predecessor_level_connection_affinity_factor_first
                                       and the number of model inputs to an
                                       unsigned integer.
        predecessor_level_connection_affinity_factor_body: float:
                                       range supported: [0.000...1,inf]:
                                       The approximate probability of this
                                       DenseUnitModule connecting directly to
                                       a given DenseUnitModule on the
                                       immediately preceeding DenseLevel.
                                       This number will be divided by the
                                       number of DenseUnitModules on the
                                       preceeding layer. The rounding rule
                                       selected will either floor or ceiling
                                       the quotient. The resulting unsigned
                                       integer will be the number of random
                                       connections made to randomly selected
                                       members of preceeding DenseLevel.
        predecessor_level_connection_affinity_factor_body_rounding_rule: str:
                                       options are "floor" and "ciling":
                                       whether to use np.floor or np.ceil to
                                       cast the quotient of predecessor_level_connection_affinity_factor_body
                                       and the number of model inputs to an
                                       unsigned integer.
        predecessor_level_connection_affinity_factor_decay_body: function:
                                       Function that alters the predecessor_level_connection_affinity_factor_body
                                       with regard to connections to
                                       grandparent, great grandparent
                                       predecessor levels, usually a decay
                                       function that decreases the
                                       connectivity with distant predecessors
                                       compared to recent predecessor levels.
         num_lateral_connection_tries_per_unit: int: defaults to 1,

        ###!! dense_unit: tf.keras.layers.Dense
          *args,
          **kwargs

    """

    def __init__(
             self,
             n_neurons: int,
             predecessor_levels: list,
             possible_predecessor_connections: dict,
             parallel_units: list,
             unit_id: int,
             level_name: str,
             trial_number: int,
             level_number: int,
             activation='elu',
             # minimum_skip_connection_depth = 1,
             merging_strategy="concatenate",
             maximum_skip_connection_depth=7,
             predecessor_level_connection_affinity_factor_first=5,
             predecessor_level_connection_affinity_factor_first_rounding_rule='ceil',
             predecessor_level_connection_affinity_factor_main=0.7,
             predecessor_level_connection_affinity_factor_main_rounding_rule='ceil',
             predecessor_level_connection_affinity_factor_decay_main=zero_7_exp_decay,
             max_consecutive_lateral_connections=7,
             gate_after_n_lateral_connections=3,
             gate_activation_function=simple_sigmoid,
             p_lateral_connection=.97,
             p_lateral_connection_decay=zero_95_exp_decay,
             num_lateral_connection_tries_per_unit=1,
             bnorm_or_dropout='bnorm',
             dropout_rate=0.2,
             *args,
             **kwargs):

        Unit.__init__(self,
                      n_neurons=n_neurons,
                      predecessor_levels=predecessor_levels,
                      unit_id=unit_id,
                      level_name=level_name,
                      trial_number=trial_number,
                      level_number=level_number,
                      *args,
                      **kwargs)

        DenseLateralConnectivity.__init__(
              self,
              max_consecutive_lateral_connections=max_consecutive_lateral_connections,
              gate_after_n_lateral_connections=gate_after_n_lateral_connections,
              gate_activation_function=gate_activation_function,
              p_lateral_connection=p_lateral_connection,
              p_lateral_connection_decay=p_lateral_connection_decay,
              num_lateral_connection_tries_per_unit=num_lateral_connection_tries_per_unit,
              * args,
              **kwargs)

        self.activation = activation
        self.merging_strategy = merging_strategy
        if self.level_number < 1:
            raise ValueError("Level number 0 is reserved for "
                             "tf.keras.layers.Input objects, and negative "
                             "level numbers don't make any sense. level_number must be at least 1")
        self.predecessor_level_number = self.level_number - 1
        # add validation for range of next param
        self.maximum_skip_connection_depth = maximum_skip_connection_depth
        # add validation for range of next param
        self.predecessor_level_connection_affinity_factor_first =\
            predecessor_level_connection_affinity_factor_first
        if predecessor_level_connection_affinity_factor_first_rounding_rule\
                == "floor":
            self.predecessor_level_connection_affinity_factor_first_rounding_rule = np.floor
        elif predecessor_level_connection_affinity_factor_first_rounding_rule\
                == "ceil":
            self.predecessor_level_connection_affinity_factor_first_rounding_rule = np.ceil
        else:
            raise ValueError("predecessor_level_connection_affinity_factor_first_rounding_rule "
                             "must be 'floor' or 'ceil'")
        # add validation for range of next param
        self.predecessor_level_connection_affinity_factor_main =\
            predecessor_level_connection_affinity_factor_main
        if predecessor_level_connection_affinity_factor_main_rounding_rule\
                == "floor":
            self.predecessor_level_connection_affinity_factor_main_rounding_rule = np.floor
        elif predecessor_level_connection_affinity_factor_main_rounding_rule\
                == "ceil":
            self.predecessor_level_connection_affinity_factor_main_rounding_rule = np.ceil
        else:
            raise ValueError("predecessor_level_connection_affinity_factor_body_rounding_rule "
                             "must be 'floor' or 'ceiling'")
        self.predecessor_level_connection_affinity_factor_decay_main =\
            predecessor_level_connection_affinity_factor_decay_main

        self.parallel_units = parallel_units

        self.bnorm_or_dropout = bnorm_or_dropout
        self.dropout_rate = dropout_rate
        self.predecessor_connectivity_future = []
        self.lateral_connectivity_future = []
        # Which lateral connections will be gated
        self.lateral_connectivity_gating_index = []
        self.consolidated_connectivity_future = []
        self.meta_predecessor_connectivity_level_number = []
        self.meta_predecessor_connectivity_unit_id = []
        self.successor_levels = []
        self.materialized = False

    #def set_lateral_dense_unit_modules(self,
    #                                   lateral_dense_unit_modules):
    #    self.lateral_dense_unit_modules = lateral_dense_unit_modules

    def set_possible_predecessor_connections(self):
        self.possible_predecessor_connections =\
             self.predecessor_levels[:-1].possible_predecessor_connections
        self.possible_predecessor_connections[
            f"{self.predecessor_levels[:-1].level_number}"] =\
            self.predecessor_levels[:-1].prototype

    def parse_predecessor_connections_to_level(self,
                                               level_num_to_process_int: int):

        # a string representation of the level number we are processing
        # level_num_to_process_str = str(level_num_to_process_int)

        # A list of the units that are possible connections for this level.
        units_from_level_to_process =\
            self.predecessor_levels[level_num_to_process_int].parallel_units
        len_of_units_from_level_to_process = len(units_from_level_to_process)
        level_num_from_level_0 = units_from_level_to_process[0].level_number
        # Determine the number of connections that this level should make to
        # units on the level above:
        # Formula is affinity coecciient * number of units on the k-1 th level
        # If this is not the input level, then this is also multiplied by
        # the decay function(self.level_number - level_num_to_process_int)
        k_minus_n = self.level_number - level_num_from_level_0
        if level_num_to_process_int == 0:
            pass
        if self.level_number != 1:
            num_predecessor_connections_unrounded =\
                self.predecessor_level_connection_affinity_factor_main *\
                self.predecessor_level_connection_affinity_factor_decay_main(
                    k_minus_n) *\
                len_of_units_from_level_to_process
            num_predecessor_connections =\
                int(
                    self.predecessor_level_connection_affinity_factor_main_rounding_rule(
                        num_predecessor_connections_unrounded))
        if self.level_number == 1:
            num_predecessor_connections_unrounded =\
                self.predecessor_level_connection_affinity_factor_first *\
                len_of_units_from_level_to_process
            num_predecessor_connections =\
                int(
                    self.predecessor_level_connection_affinity_factor_first_rounding_rule(
                        num_predecessor_connections_unrounded))

        predecessor_connection_index_options =\
            np.arange(len_of_units_from_level_to_process)

        predecessor_connection_picks_by_index =\
            np.random.choice(predecessor_connection_index_options,
                             size=(num_predecessor_connections))
        connections_to_this_level = [units_from_level_to_process[j]
                                     for j in
                                     predecessor_connection_picks_by_index]
        for unit_0 in connections_to_this_level:
            self.predecessor_connectivity_future.append(unit_0)

    def set_predecessor_connectiivty(self):
        for i in np.arange(self.level_number):
            self.parse_predecessor_connections_to_level(i)

    def set_lateral_connectivity_future(self):
        # Empty list to be populated with the index positions of the DenseUnit
        # objects on the list self.lateral_connections
        connection_index = []
        gated_bool = []

        # once for each unit having a unit_id lower than k, [repeat once for]
        for i in np.arange(self.unit_id):
            if i != 0:
                k_minus_n = self.unit_id - i
                for _ in np.arange(self.num_lateral_connection_tries_per_unit):
                    add_connection = self.select_connection_or_not(k_minus_n)
                    gate_if_connected = self.gate_or_not()
                    if add_connection:
                        connection_index.append(int(k_minus_n))
                        gated_bool.append(gate_if_connected)

        self.lateral_connectivity_future =\
            [self.parallel_units[index_number_0]
             for index_number_0 in connection_index]
        self.lateral_connectivity_gating_index =\
            [p for
             p in gated_bool]

    def parse_meta_predecessor_connectivity(self):
        """The purpose of this class is to refactor the 6 - dimentional
        breadth first search necessary for Predecessors to validate that they
        have at least one Successor connection (no disjointed graph).
        Without this, each Unit (that isn't one of the level's layer's Units,
        would need to query 1. each successor Level, in it, each successor Unit,
        and in it, each Level in its' predecessor_conenctivity_future, in range
        minimum_skip_connection_depth, maximum_skip_connection_depth, then each
        level therein for any unit having the same level_number and unit_id as
        self. Obviously this has numerous problems. 1. It is a 6 dimentional
        bredth first search. (Refactoring can reduce the problem some).
        2. Iterating through large Units objects is not a computationally
        efficient way to do a traversal that may consist of more than a bilion
        individual comparisons. It is beter to extract the metadata from each of
        these Units and make a list at the units level (Then merge this list at
        the Levels level)."""
        meta_level_number = []
        meta_unit_id = []
        for unit_0 in self.predecessor_connectivity_future:
            meta_level_number.append(
                unit_0.level_number)
            meta_unit_id.append(
                unit_0.unit_id)
        self.meta_predecessor_connectivity_level_number =\
            jnp.array(meta_level_number, dtype=jnp.int32)
        self.meta_predecessor_connectivity_unit_id =\
            jnp.array(meta_unit_id, dtype=jnp.int32)

    def set_connectivity_future_prototype(self):
        self.set_predecessor_connectiivty()
        self.set_lateral_connectivity_future()
        self.parse_meta_predecessor_connectivity()

    def parse_dense_layer_object(self):
        return "under construction"

    def get_predecessor_connectivity_future(self):
        return self.predecessor_connectivity_future

    def get_lateral_connectivity_future(self):
        return self.lateral_connectivity_future

    def get_consolidated_connectivity_future(self):
        return self.consolidated_connectivity_future

    def get_lateral_connectivity_gating_index(self):
        return self.lateral_connectivity_gating_index

    # call only after detect_successor_connectivity_errors(), inherted
    # from Unit is called
    def resolve_successor_connectivity_errors(self, unselected_unit):
        map_predecessor_level = {}
        for i in np.arange(len(self.predecessor_levels)):
            map_predecessor_level[
                str(self.predecessor_levels[i].level_number)] = i
        print(
            f"I am: {self.level_number}: My predecessors are {[pl.level_number for pl in self.predecessor_levels]}")
        self.predecessor_connectivity_future.append(
            self.predecessor_levels[map_predecessor_level[str(
                unselected_unit[0])]]
            .parallel_units[unselected_unit[1]])

    def util_set_predecessor_connectivity_metadata(self):
        self.consolidated_connectivity_future =\
            self.predecessor_connectivity_future +\
            self.lateral_connectivity_future
        self.util_predecessor_connectivity_metadata =\
            [f"level_number:{unit_0.level_number},unit_id:{unit_0.unit_id}"
             for unit_0 in self.consolidated_connectivity_future]

    def materialize(self):
        if not self.materialized:
            print(f"materialize:_{self.name} called")
            un_materilized_predecessor_units =\
                self.lateral_connectivity_future + self.predecessor_connectivity_future
            materialized_predecessor_units =\
                [unit_0.neural_network_layer for unit_0
                    in un_materilized_predecessor_units]
            print("materialized network layers")
            print(materialized_predecessor_units)
            if self.merging_strategy == "concatenate":
                # rn_1 = int(np.round(np.random.random(1)[0]*10**12))
                rn_1 = ""
                print(
                    "materialized_predecessor_units "
                    f"{materialized_predecessor_units}")
                unprocessed_merged_nn_layer_input = tf.keras.layers.Concatenate(
                    axis=1, name=f"{self.name}_cat_{rn_1}")(materialized_predecessor_units)
            elif self.merging_strategy == "add":
                # rn_2 = int(np.round(np.random.random(1)[0]*10**12))
                rn_2 = ''
                unprocessed_merged_nn_layer_input = tf.keras.layers.Add(
                    name=f"{self.name}_add_{rn_2}")(materialized_predecessor_units)
            else:
                raise ValueError("The only supported arguments for "
                                 "merging_strategy are 'concatenate' and add")

            if self.bnorm_or_dropout == "bnorm":
                # rn_3 = int(np.round(np.random.random(1)[0]*10**12))
                rn_3 = ''
                merged_neural_network_layer_input = tf.keras.layers.BatchNormalization(
                    name=f"{self.name}_btn_{rn_3}")(unprocessed_merged_nn_layer_input)
            elif self.bnorm_or_dropout == 'dropout':
                # rn_4 = int(np.round(np.random.random(1)[0]*10**12))
                rn_4 = ''
                merged_neural_network_layer_input =\
                    tf.keras.layers.Dropout(
                        dropout_rate=self.dropout_rate,
                        name=f"{self.name}_drp_{rn_4}")(unprocessed_merged_nn_layer_input)
            else:
                raise ValueError("The only arguments supported by the parameter "
                                 "'bnorm_or_dropout' are 'bnorm' and 'dropout'")
            rn_5 = int(np.round(np.random.random(1)[0]*10**12))
            rn_5 = ''
            self.neural_network_layer =\
                tf.keras.layers.Dense(
                    self.n_neurons,
                    self.activation,
                    name=f"{self.name}_dns_{rn_5}")(merged_neural_network_layer_input)
            self.materialized = True
        # refactor the lagic below and this class is complete
        # self.dense_unit_module_id = dense_unit_module_id
        # self.upstream_levels = upstream_levels
        # # Refactor to only take the actual Dense objects as the
        # # argument
        # if len(upstream_connections) > 1 and level_number != 1:
        #     self.dense_unit_arg =\
        #         tf.leras.layers.Concatenate(axis=1)(
        #             [self.
        #                 upstream_levels[i].parallel_dense_layers[j].dense_unit
        #              for j in
        #                 tf.range(
        #                     len(upstream_levels[i].parallel_dense_units))
        #              for i in tf.range(len(upstream_levels))])
        # elif len(upstream_connections) == 1 and own_level_number != 1:
        #     self.dense_unit_arg =\
        #         self.upstream_levels[0].parallel_dense_layers[0].dense_unit
        # elif len(upstream_connections) > 1 and own_level_number == 1:
        #     self.dense_unit_arg =\
        #         tf.keras.layers.Concatenate(axis=1)(self.inputs)
        # elif len(upstream_connections) == 1 and own_level_number == 1:
        #     self.dense_unit_arg = self.inputs[0]
        # self.dense_unit =\
        #     tf.keras.layers.\
        #     Dense(units=self.units,
        #           activation=self.activation)(self.dense_unit_arg)

        # for i in tf.range(len(upstream_connections)):
        #     if upstream_connections[i][0] >= own_level_number:
        #         raise ValueError("Level numbers in upstream layer connections "
        #                          " > a unit's own level doesn't make sense.")


class FinalDenseUnit(DenseUnit):
    """docstring for FinalDenseUnit."""

    def __init__(
            self,
            output_shape: int,
            predecessor_levels: list,
            possible_predecessor_connections: dict,
            parallel_units: list,
            unit_id: int,
            level_name: str,
            trial_number: int,
            level_number: int,
            final_activation=None,
            maximum_skip_connection_depth=7,
            predecessor_level_connection_affinity_factor_first=5,
            predecessor_level_connection_affinity_factor_first_rounding_rule='ceil',
            predecessor_level_connection_affinity_factor_main=0.7,
            predecessor_level_connection_affinity_factor_main_rounding_rule='ceil',
            predecessor_level_connection_affinity_factor_decay_main=zero_7_exp_decay,
            predecessor_level_connection_affinity_factor_final_to_kminus1=2,
            max_consecutive_lateral_connections=7,
            gate_after_n_lateral_connections=3,
            gate_activation_function=simple_sigmoid,
            p_lateral_connection=.97,
            p_lateral_connection_decay=zero_95_exp_decay,
            num_lateral_connection_tries_per_unit=1,
            *args,
            **kwargs
            ):

        activation = final_activation
        n_neurons = output_shape

        super().__init__(n_neurons=n_neurons,
                         predecessor_levels=predecessor_levels,
                         possible_predecessor_connections=possible_predecessor_connections,
                         parallel_units=parallel_units,
                         unit_id=unit_id,
                         level_name=level_name,
                         trial_number=trial_number,
                         level_number=level_number,
                         activation=activation,
                         maximum_skip_connection_depth=maximum_skip_connection_depth,
                         predecessor_level_connection_affinity_factor_first=predecessor_level_connection_affinity_factor_first,
                         predecessor_level_connection_affinity_factor_first_rounding_rule=predecessor_level_connection_affinity_factor_first_rounding_rule,
                         predecessor_level_connection_affinity_factor_main=predecessor_level_connection_affinity_factor_main,
                         predecessor_level_connection_affinity_factor_main_rounding_rule=predecessor_level_connection_affinity_factor_main_rounding_rule,
                         predecessor_level_connection_affinity_factor_decay_main=predecessor_level_connection_affinity_factor_decay_main,
                         max_consecutive_lateral_connections=max_consecutive_lateral_connections,
                         gate_after_n_lateral_connections=gate_after_n_lateral_connections,
                         gate_activation_function=gate_activation_function,
                         p_lateral_connection=p_lateral_connection,
                         p_lateral_connection_decay=p_lateral_connection_decay,
                         num_lateral_connection_tries_per_unit=num_lateral_connection_tries_per_unit,
                         *args,
                         **kwargs)

        self.predecessor_level_connection_affinity_factor_final_to_kminus1 =\
            predecessor_level_connection_affinity_factor_final_to_kminus1

    def set_final_connectivity_future_prototype(self):
        last_level_units = self.predecessor_levels[-1].parallel_units
        print(
            f"Debug: I am {self.level_number} selecting {last_level_units[0].level_number}")
        num_units_0 = len(last_level_units)
        indexes_oflast_level_units = np.arange(num_units_0)

        num_to_pick = self.predecessor_level_connection_affinity_factor_final_to_kminus1 *\
            num_units_0
        units_chosen_by_index =\
            np.random.choice(indexes_oflast_level_units,
                             size=num_to_pick)
        for i in units_chosen_by_index:
            self.predecessor_connectivity_future.append(last_level_units[i])
        self.set_connectivity_future_prototype()
        self.set_lateral_connectivity_future()
        self.parse_meta_predecessor_connectivity()

### ->


class RealNeuron(Unit,
                 DenseLateralConnectivity):

    # Add to base unit
    #         __predecessor_connections_future: list:
    #                              List of dictioaries:
    #                              The keys for each dict will be the index
    #                              position of which CURRENT layer dense_unit is
    #                              being connected upstream.
    #                              The value in the dictionary is a 2d list,
    #                              each listing an upstream dense_unit to which
    #                              it connects, represented in the format:
    #                              [layer_number of the layer the dense_unit,
    #                              index of dense_unit ]
    #                              are the index positions of the parallel dense
    #                              unit on the respective downstream layer to which
    #                              it connects.
    #

    """
    A future that will parse a tf.keras.layers.Dense object, once it
    determines which tf.keras.layers.___ objects upstream in the neural
    network which it will ingest as its as its predecessor.

    Args:
        n_neurons: int:
                    number of dense units the tf.keras.layers.Dense object
                    parsed by this object will consist of.
        axon_activation: str:
                    The string representation of the activation
                    function to be used in the tf.keras.layers.Dense object
                    returned by this object.
        self.dendrite_activation: str:

        level_number: int, the layer number of the DenseLevels it belongs to (0 - indexed starting with the first Input layer).
        predecessor_levels: list: A list of DenseLevel objects that preceed
                                  the DenseLevel which this DenseUnitModule
                                  being instantiated belongs to.
        dense_unit_module_id: list, [layer_number, 0-indexed serial number of
                              DenseUnitModule objects in the DenseLevel
                              object to which this given DenseUnitModule
                              is a member of]
        maximum_skip_connection_depth: int:
                                       Maximum number of Levels above
                                       the current Level which the
                                       DenseUnitModule may directly ingest
                                       a DenseUnitModule as a predecessor.
        predecessor_level_connection_affinity_factor_first: float:
                                       range supported: [0.000...1,inf]:
                                       The approximate probability of this
                                       DenseUnitModule connecting directly to
                                       a given Input layer. This number will
                                       be divided by the number of inputs.
                                       The rounding rule will selected will
                                       either floor or ceiling the quotient.
                                       The resulting unsigned integer will
                                       be the number of random connections
                                       made to randomly selected members of
                                       the Inputs level. Many models will
                                       have 1 input, and each will will
                                       connect to
                                       all inputs. Setting this parameter to
                                       2 will make at least 2 connections to
                                       each input.
        predecessor_level_connection_affinity_factor_first_rounding_rule: str:
                                       options are "floor" and "ciling":
                                       whether to use np.floor or np.ceil to
                                       cast the quotient of predecessor_level_connection_affinity_factor_first
                                       and the number of model inputs to an
                                       unsigned integer.
        predecessor_level_connection_affinity_factor_body: float:
                                       range supported: [0.000...1,inf]:
                                       The approximate probability of this
                                       DenseUnitModule connecting directly to
                                       a given DenseUnitModule on the
                                       immediately preceeding DenseLevel.
                                       This number will be divided by the
                                       number of DenseUnitModules on the
                                       preceeding layer. The rounding rule
                                       selected will either floor or ceiling
                                       the quotient. The resulting unsigned
                                       integer will be the number of random
                                       connections made to randomly selected
                                       members of preceeding DenseLevel.
        predecessor_level_connection_affinity_factor_body_rounding_rule: str:
                                       options are "floor" and "ciling":
                                       whether to use np.floor or np.ceil to
                                       cast the quotient of predecessor_level_connection_affinity_factor_body
                                       and the number of model inputs to an
                                       unsigned integer.
        predecessor_level_connection_affinity_factor_decay_body: function:
                                       Function that alters the predecessor_level_connection_affinity_factor_body
                                       with regard to connections to
                                       grandparent, great grandparent
                                       predecessor levels, usually a decay
                                       function that decreases the
                                       connectivity with distant predecessors
                                       compared to recent predecessor levels.
         num_lateral_connection_tries_per_unit: int: defaults to 1,

        ###!! dense_unit: tf.keras.layers.Dense
          *args,
          **kwargs

    """

    def __init__(
             self,
             n_axon_nuerons: int,
             axon_activation: str,
             n_dendrites: int,
             dendrite_activation: str,
             predecessor_levels: list,
             possible_predecessor_connections: dict,
             parallel_units: list,
             unit_id: int,
             level_name: str,
             trial_number: int,
             level_number: int,
             # minimum_skip_connection_depth = 1,
             dendrite_units=1,
             merging_strategy="concatenate",
             maximum_skip_connection_depth=7,
             predecessor_level_connection_affinity_factor_first=5,
             predecessor_level_connection_affinity_factor_first_rounding_rule='ceil',
             predecessor_level_connection_affinity_factor_main=0.7,
             predecessor_level_connection_affinity_factor_main_rounding_rule='ceil',
             predecessor_level_connection_affinity_factor_decay_main=zero_7_exp_decay,
             max_consecutive_lateral_connections=7,
             gate_after_n_lateral_connections=3,
             gate_activation_function=simple_sigmoid,
             p_lateral_connection=.97,
             p_lateral_connection_decay=zero_95_exp_decay,
             num_lateral_connection_tries_per_unit=1,
             bnorm_or_dropout='bnorm',
             dropout_rate=0.2,
             *args,
             **kwargs):

        Unit.__init__(self,
                      n_neurons=n_axon_nuerons,
                      predecessor_levels=predecessor_levels,
                      unit_id=unit_id,
                      level_name=level_name,
                      trial_number=trial_number,
                      level_number=level_number,
                      *args,
                      **kwargs)

        DenseLateralConnectivity.__init__(
              self,
              max_consecutive_lateral_connections=max_consecutive_lateral_connections,
              gate_after_n_lateral_connections=gate_after_n_lateral_connections,
              gate_activation_function=gate_activation_function,
              p_lateral_connection=p_lateral_connection,
              p_lateral_connection_decay=p_lateral_connection_decay,
              num_lateral_connection_tries_per_unit=num_lateral_connection_tries_per_unit,
              * args,
              **kwargs)

        self.n_axon_neurons = self.n_neurons
        self.axon_activation = axon_activation
        self.n_dendrites = n_dendrites
        self.dendrite_activation = dendrite_activation
        self.merging_strategy = merging_strategy
        self.dendrite_units = dendrite_units
        if self.level_number < 1:
            raise ValueError("Level number 0 is reserved for "
                             "tf.keras.layers.Input objects, and negative "
                             "level numbers don't make any sense. level_number must be at least 1")
        self.predecessor_level_number = self.level_number - 1
        # add validation for range of next param
        self.maximum_skip_connection_depth = maximum_skip_connection_depth
        # add validation for range of next param
        self.predecessor_level_connection_affinity_factor_first =\
            predecessor_level_connection_affinity_factor_first
        if predecessor_level_connection_affinity_factor_first_rounding_rule\
                == "floor":
            self.predecessor_level_connection_affinity_factor_first_rounding_rule = np.floor
        elif predecessor_level_connection_affinity_factor_first_rounding_rule\
                == "ceil":
            self.predecessor_level_connection_affinity_factor_first_rounding_rule = np.ceil
        else:
            raise ValueError("predecessor_level_connection_affinity_factor_first_rounding_rule "
                             "must be 'floor' or 'ceil'")
        # add validation for range of next param
        self.predecessor_level_connection_affinity_factor_main =\
            predecessor_level_connection_affinity_factor_main
        if predecessor_level_connection_affinity_factor_main_rounding_rule\
                == "floor":
            self.predecessor_level_connection_affinity_factor_main_rounding_rule = np.floor
        elif predecessor_level_connection_affinity_factor_main_rounding_rule\
                == "ceil":
            self.predecessor_level_connection_affinity_factor_main_rounding_rule = np.ceil
        else:
            raise ValueError("predecessor_level_connection_affinity_factor_body_rounding_rule "
                             "must be 'floor' or 'ceiling'")
        self.predecessor_level_connection_affinity_factor_decay_main =\
            predecessor_level_connection_affinity_factor_decay_main

        self.parallel_units = parallel_units

        self.bnorm_or_dropout = bnorm_or_dropout
        self.dropout_rate = dropout_rate
        self.predecessor_connectivity_future = []
        self.lateral_connectivity_future = []
        # Which lateral connections will be gated
        self.lateral_connectivity_gating_index = []
        self.consolidated_connectivity_future = []
        self.meta_predecessor_connectivity_level_number = []
        self.meta_predecessor_connectivity_unit_id = []
        self.successor_levels = []
        self.materialized = False

    #def set_lateral_dense_unit_modules(self,
    #                                   lateral_dense_unit_modules):
    #    self.lateral_dense_unit_modules = lateral_dense_unit_modules

    def set_possible_predecessor_connections(self):
        self.possible_predecessor_connections =\
             self.predecessor_levels[:-1].possible_predecessor_connections
        self.possible_predecessor_connections[
            f"{self.predecessor_levels[:-1].level_number}"] =\
            self.predecessor_levels[:-1].prototype

    def parse_predecessor_connections_to_level(self,
                                               level_num_to_process_int: int):

        # a string representation of the level number we are processing
        # level_num_to_process_str = str(level_num_to_process_int)

        # A list of the units that are possible connections for this level.
        units_from_level_to_process =\
            self.predecessor_levels[level_num_to_process_int].parallel_units
        len_of_units_from_level_to_process = len(units_from_level_to_process)
        level_num_from_level_0 = units_from_level_to_process[0].level_number
        # Determine the number of connections that this level should make to
        # units on the level above:
        # Formula is affinity coecciient * number of units on the k-1 th level
        # If this is not the input level, then this is also multiplied by
        # the decay function(self.level_number - level_num_to_process_int)
        k_minus_n = self.level_number - level_num_from_level_0
        if level_num_to_process_int == 0:
            pass
        if self.level_number != 1:
            num_predecessor_connections_unrounded =\
                self.predecessor_level_connection_affinity_factor_main *\
                self.predecessor_level_connection_affinity_factor_decay_main(
                    k_minus_n) *\
                len_of_units_from_level_to_process
            num_predecessor_connections =\
                int(
                    self.predecessor_level_connection_affinity_factor_main_rounding_rule(
                        num_predecessor_connections_unrounded))
        if self.level_number == 1:
            num_predecessor_connections_unrounded =\
                self.predecessor_level_connection_affinity_factor_first *\
                len_of_units_from_level_to_process
            num_predecessor_connections =\
                int(
                    self.predecessor_level_connection_affinity_factor_first_rounding_rule(
                        num_predecessor_connections_unrounded))

        predecessor_connection_index_options =\
            np.arange(len_of_units_from_level_to_process)

        predecessor_connection_picks_by_index =\
            np.random.choice(predecessor_connection_index_options,
                             size=(num_predecessor_connections))
        connections_to_this_level = [units_from_level_to_process[j]
                                     for j in
                                     predecessor_connection_picks_by_index]
        for unit_0 in connections_to_this_level:
            self.predecessor_connectivity_future.append(unit_0)

    def set_predecessor_connectiivty(self):
        for i in np.arange(self.level_number):
            self.parse_predecessor_connections_to_level(i)

    def set_lateral_connectivity_future(self):
        # Empty list to be populated with the index positions of the DenseUnit
        # objects on the list self.lateral_connections
        connection_index = []
        gated_bool = []

        # once for each unit having a unit_id lower than k, [repeat once for]
        for i in np.arange(self.unit_id):
            if i != 0:
                k_minus_n = self.unit_id - i
                for _ in np.arange(self.num_lateral_connection_tries_per_unit):
                    add_connection = self.select_connection_or_not(k_minus_n)
                    gate_if_connected = self.gate_or_not()
                    if add_connection:
                        connection_index.append(int(k_minus_n))
                        gated_bool.append(gate_if_connected)

        self.lateral_connectivity_future =\
            [self.parallel_units[index_number_0]
             for index_number_0 in connection_index]
        self.lateral_connectivity_gating_index =\
            [p for
             p in gated_bool]

    def parse_meta_predecessor_connectivity(self):
        """The purpose of this class is to refactor the 6 - dimentional
        breadth first search necessary for Predecessors to validate that they
        have at least one Successor connection (no disjointed graph).
        Without this, each Unit (that isn't one of the level's layer's Units,
        would need to query 1. each successor Level, in it, each successor Unit,
        and in it, each Level in its' predecessor_conenctivity_future, in range
        minimum_skip_connection_depth, maximum_skip_connection_depth, then each
        level therein for any unit having the same level_number and unit_id as
        self. Obviously this has numerous problems. 1. It is a 6 dimentional
        bredth first search. (Refactoring can reduce the problem some).
        2. Iterating through large Units objects is not a computationally
        efficient way to do a traversal that may consist of more than a bilion
        individual comparisons. It is beter to extract the metadata from each of
        these Units and make a list at the units level (Then merge this list at
        the Levels level)."""
        meta_level_number = []
        meta_unit_id = []
        for unit_0 in self.predecessor_connectivity_future:
            meta_level_number.append(
                unit_0.level_number)
            meta_unit_id.append(
                unit_0.unit_id)
        self.meta_predecessor_connectivity_level_number =\
            jnp.array(meta_level_number, dtype=jnp.int32)
        self.meta_predecessor_connectivity_unit_id =\
            jnp.array(meta_unit_id, dtype=jnp.int32)

    def set_connectivity_future_prototype(self):
        self.set_predecessor_connectiivty()
        self.set_lateral_connectivity_future()
        self.parse_meta_predecessor_connectivity()

    def parse_dense_layer_object(self):
        return "under construction"

    def get_predecessor_connectivity_future(self):
        return self.predecessor_connectivity_future

    def get_lateral_connectivity_future(self):
        return self.lateral_connectivity_future

    def get_consolidated_connectivity_future(self):
        return self.consolidated_connectivity_future

    def get_lateral_connectivity_gating_index(self):
        return self.lateral_connectivity_gating_index

    # call only after detect_successor_connectivity_errors(), inherted
    # from Unit is called
    def resolve_successor_connectivity_errors(self, unselected_unit):
        map_predecessor_level = {}
        for i in np.arange(len(self.predecessor_levels)):
            map_predecessor_level[
                str(self.predecessor_levels[i].level_number)] = i
        print(
            f"I am: {self.level_number}: My predecessors are {[pl.level_number for pl in self.predecessor_levels]}")
        self.predecessor_connectivity_future.append(
            self.predecessor_levels[map_predecessor_level[str(
                unselected_unit[0])]]
            .parallel_units[unselected_unit[1]])

    def util_set_predecessor_connectivity_metadata(self):
        self.consolidated_connectivity_future =\
            self.predecessor_connectivity_future +\
            self.lateral_connectivity_future
        self.util_predecessor_connectivity_metadata =\
            [f"level_number:{unit_0.level_number},unit_id:{unit_0.unit_id}"
             for unit_0 in self.consolidated_connectivity_future]

    def materialize(self):
        if not self.materialized:
            print(f"materialize:_{self.name} called")
            un_materilized_predecessor_units =\
                self.lateral_connectivity_future + self.predecessor_connectivity_future
            materialized_predecessor_units = []
            for unit_0 in un_materilized_predecessor_units:
                if isinstance(unit_0, InputUnit):
                    materialized_predecessor_units.append(
                        unit_0.neural_network_layer)
                else:
                    print(f"Trying unit (should be input):{unit_0}")
                    materialized_predecessor_units += unit_0.dendrites
                    print("materialized network layers: case RealNeuron")
                    print(materialized_predecessor_units)
                    print("materialized network layers case: Input")
                    print(materialized_predecessor_units)
            print("materialized network layers")
            print(materialized_predecessor_units)
            if self.merging_strategy == "concatenate":
                # rn_1 = int(np.round(np.random.random(1)[0]*10**12))
                rn_1 = ""
                unprocessed_merged_nn_layer_input = tf.keras.layers.Concatenate(
                    axis=1, name=f"{self.name}_cat_{rn_1}")(materialized_predecessor_units)
            elif self.merging_strategy == "add":
                # rn_2 = int(np.round(np.random.random(1)[0]*10**12))
                rn_2 = ''
                unprocessed_merged_nn_layer_input = tf.keras.layers.Add(
                    name=f"{self.name}_add_{rn_2}")(materialized_predecessor_units)
            else:
                raise ValueError("The only supported arguments for "
                                 "merging_strategy are 'concatenate' and add")

            if self.bnorm_or_dropout == "bnorm":
                # rn_3 = int(np.round(np.random.random(1)[0]*10**12))
                rn_3 = ''
                merged_neural_network_layer_input = tf.keras.layers.BatchNormalization(
                    name=f"{self.name}_btn_{rn_3}")(unprocessed_merged_nn_layer_input)
            elif self.bnorm_or_dropout == 'dropout':
                # rn_4 = int(np.round(np.random.random(1)[0]*10**12))
                rn_4 = ''
                merged_neural_network_layer_input =\
                    tf.keras.layers.Dropout(
                        dropout_rate=self.dropout_rate,
                        name=f"{self.name}_drp_{rn_4}")(unprocessed_merged_nn_layer_input)
            else:
                raise ValueError("The only arguments supported by the parameter "
                                 "'bnorm_or_dropout' are 'bnorm' and 'dropout'")
            rn_5 = int(np.round(np.random.random(1)[0]*10**12))
            self.axon =\
                tf.keras.layers.Dense(
                    self.n_axon_neurons,
                    self.axon_activation,
                    name=f"{self.name}_axn")(merged_neural_network_layer_input)
            self.dendrites =\
                [
                    tf.keras.layers.Dense(self.dendrite_units,
                                          activation=self.dendrite_activation,
                                          name=f"{self.name}_dend-{int(i)}")(self.axon)
                    for i in np.arange(self.n_dendrites)]

            self.materialized = True


class FinalRealNeuron(RealNeuron):
    """docstring for FinalDenseUnit."""

    def __init__(
            self,
            n_axon_nuerons: int,
            axon_activation: str,
            output_shape: int,
            predecessor_levels: list,
            possible_predecessor_connections: dict,
            parallel_units: list,
            unit_id: int,
            level_name: str,
            trial_number: int,
            level_number: int,
            final_activation=None,
            maximum_skip_connection_depth=7,
            predecessor_level_connection_affinity_factor_first=5,
            predecessor_level_connection_affinity_factor_first_rounding_rule='ceil',
            predecessor_level_connection_affinity_factor_main=0.7,
            predecessor_level_connection_affinity_factor_main_rounding_rule='ceil',
            predecessor_level_connection_affinity_factor_decay_main=zero_7_exp_decay,
            predecessor_level_connection_affinity_factor_final_to_kminus1=2,
            max_consecutive_lateral_connections=7,
            gate_after_n_lateral_connections=3,
            gate_activation_function=simple_sigmoid,
            p_lateral_connection=.97,
            p_lateral_connection_decay=zero_95_exp_decay,
            num_lateral_connection_tries_per_unit=1,
            *args,
            **kwargs
            ):

        # n_axon_nuerons: int, axon_activation: str, n_dendrites: int, dendrite_activation: str

        super().__init__(n_axon_nuerons=n_axon_nuerons,
                         axon_activation=axon_activation,
                         n_dendrites=1,
                         dendrite_activation=final_activation,
                         predecessor_levels=predecessor_levels,
                         possible_predecessor_connections=possible_predecessor_connections,
                         parallel_units=parallel_units,
                         unit_id=unit_id,
                         level_name=level_name,
                         trial_number=trial_number,
                         level_number=level_number,
                         dendrite_units=output_shape,
                         maximum_skip_connection_depth=maximum_skip_connection_depth,
                         predecessor_level_connection_affinity_factor_first=predecessor_level_connection_affinity_factor_first,
                         predecessor_level_connection_affinity_factor_first_rounding_rule=predecessor_level_connection_affinity_factor_first_rounding_rule,
                         predecessor_level_connection_affinity_factor_main=predecessor_level_connection_affinity_factor_main,
                         predecessor_level_connection_affinity_factor_main_rounding_rule=predecessor_level_connection_affinity_factor_main_rounding_rule,
                         predecessor_level_connection_affinity_factor_decay_main=predecessor_level_connection_affinity_factor_decay_main,
                         max_consecutive_lateral_connections=max_consecutive_lateral_connections,
                         gate_after_n_lateral_connections=gate_after_n_lateral_connections,
                         gate_activation_function=gate_activation_function,
                         p_lateral_connection=p_lateral_connection,
                         p_lateral_connection_decay=p_lateral_connection_decay,
                         num_lateral_connection_tries_per_unit=num_lateral_connection_tries_per_unit,
                         *args,
                         **kwargs)

        self.predecessor_level_connection_affinity_factor_final_to_kminus1 =\
            predecessor_level_connection_affinity_factor_final_to_kminus1

    def set_final_connectivity_future_prototype(self):
        last_level_units = self.predecessor_levels[-1].parallel_units
        print(
            f"Debug: I am {self.level_number} selecting {last_level_units[0].level_number}")
        num_units_0 = len(last_level_units)
        indexes_oflast_level_units = np.arange(num_units_0)

        num_to_pick = self.predecessor_level_connection_affinity_factor_final_to_kminus1 *\
            num_units_0
        units_chosen_by_index =\
            np.random.choice(indexes_oflast_level_units,
                             size=num_to_pick)
        for i in units_chosen_by_index:
            self.predecessor_connectivity_future.append(last_level_units[i])
        self.set_connectivity_future_prototype()
        self.set_lateral_connectivity_future()
        self.parse_meta_predecessor_connectivity()
