"""

Objects of class Level.
    Represents a level of tf.keras.layers.[...] units which are in
    parallel in the hirearchy of the network. For example, there will be
    DenseLevel and InputLevel objects which will inheret from this class.
    As we extend the functionality of the NAS, additional Level objects
    will be added to add support for other types of layers, such as
    Conv2D, Conv1D, Conv3D, GRUs, ... and any avante garde layer objects
"""
from cerebros.units.units import InputUnit, DenseUnit, FinalDenseUnit,\
    RealNeuron, FinalRealNeuron
from cerebros.nnfuturecomponent.neural_network_future_component\
    import NeuralNetworkFutureComponent
from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component \
    import DenseAutoMlStructuralComponent, DenseLateralConnectivity, \
    zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid
import jax.numpy as jnp
import numpy as np
from tensorflow import float32


class Level(NeuralNetworkFutureComponent,
            DenseAutoMlStructuralComponent):
    """
    Represents a level of tf.keras.layers.[...] units which are in
    parallel in the hirearchy of the network. For example, there will be
    DenseLevel and InputLevel objects which will inheret from this class.
    As we extend the functionality of the NAS, additional Level objects
    will be added to add support for other types of layers, such as
    Conv2D, Conv1D, Conv3D, GRUs, ... and any avante garde layer objects

    Args:
        level_prototype: list:
                  A 1d list of dictionaries where each dictionary
                  represents a tf.keras.layers.[?] object (or
                  mimetic thereof) which the Level consists of:
        key:
                  unsigned integers where each integer
                  represents the first argument to the
                  respective tf.keras.layers object
                  (or mimetic thereof) that it represents:
        val:
                  the string representation of the
                  name of the class of which the
                  layers.layers.Layers
            example:
                  The prototype of a layer consisting of a
                  Dense(5) unit and a Dense(3) unit would look
                  like this:
                  ['5':units.units.Dense,
                  '3':units.units.Dense]
        predecessor_levels: list:
                  List of objects of type Layers, list of
                  layers that are predecessors in the chain of
                  Levels.
        has_predecessors: str:
            options:
                  "yes", "no", "indeterminate"
        has_successors: str:
            options:
                  "yes", "no", "indeterminate"
        neural_network_future_name: str:
        trial_number: int:
                  Which sequential trial or neural architecture being
                  tried that this object is a member of e.g. if this Level
                  belonged to the 5th neural network that the AutoMl tried,
                  this would be 5.
        level_number: int:
                  Which 0 indexed level in the neural network this Level is
                  (Input level is 0, first DenseLevel is 1, second
                  DenseLevel is 2)
        minimum_skip_connection_depth=1:
                  The DenseLevels will be randomly connected
                  to others in Levels below it. These
                  connections will not be made exclusively to
                  DenseUnits in its immediate successor Level.
                  Some may be 2,3,4,5, ... Levels down. This
                  parameter controls the minimum depth of
                  connection allowed for connections not at
                  the immediate successor Level.
        maximum_skip_connection_depth=7:
                  The DenseLevels will be randomly connected
                  to others in Levels below it. These
                  connections will not be made exclusively to
                  DenseUnits in its immediate successor Level.
                  Some may be 2,3,4,5, ... Levels down. This
                  parameter controls the minimum depth of
                  connection allowed for connections not at
                  the immediate successor Level.
        predecessor_level_connection_affinity_factor_first=5:
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
        predecessor_level_connection_affinity_factor_first_rounding_rule='ceil':
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
        predecessor_level_connection_affinity_factor_main=0.7:
                  For the second and subsequent DenseLevels,
                  if its immediate predecessor level has
                  n DenseUnits, the DenseUnits on this layer will randomly select
                  predecessor_level_connection_affinity_factor_main * n
                  DenseUnits form its immediate predecessor
                  Level to connect to. The selection is
                  WITH REPLACEMENT and a given DenseUnit may
                  be selected multiple times.
        predecessor_level_connection_affinity_factor_main_rounding_rule='ceil':
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
        predecessor_level_connection_affinity_factor_decay_main=lambda x: 0.7 * x:
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
        seed=8675309:
                     *args,
                     **kwargs)

    private variables (to be set internally only but read externally):

        parallel_units:
                  A list of _Units objects
                  (See Unit, DenseUnit, etc)
    Private variables (set and accessed internally only):

        possible_predecessor_connections: list:
                  A 2d list.
                  # k is the level_number of this class
                  # i is the index position on the 2d list.
                  Each i th 1d list within the 2d list is a
                  prototype of the k - i th level.

    public methods:
        set_possible_predecessor_connections:
                  parses the variable __possible_predecessor_connections.
        get_possible_predecessor_connections:
                  Returns a copy of __possible_predecessor_connections.
        parse_units:
                  From the level_prototype and predecessor_levels, parses a Unit
                  object for each Unit prescribed in its level_prototype.

        validate_all_updtream_connected:
                  Checks each dense_unit in the layer's
                  prototype, verifies there is a key for
                  that dense_unit in listed in
                  __predecessor_connections_future. If not,
                  it randomly selects a value from
                  __availible_connections (where
                  level_number is <= max_skip_connection_depth)
                  and creates a dictionary for the dense_unit
                  in __predecessor_connections_future

    """

    def __init__(self,
                 level_prototype: list,
                 predecessor_levels: list,
                 has_predecessors: str,
                 has_successors: str,
                 neural_network_future_name: str,
                 trial_number: int,
                 level_number: int,
                 minimum_skip_connection_depth=1,
                 maximum_skip_connection_depth=7,
                 predecessor_level_connection_affinity_factor_first=5,
                 predecessor_level_connection_affinity_factor_first_rounding_rule='ceil',
                 predecessor_level_connection_affinity_factor_main=0.7,
                 predecessor_level_connection_affinity_factor_main_rounding_rule='ceil',
                 predecessor_level_connection_affinity_factor_decay_main=zero_7_exp_decay,
                 seed=8675309,
                 train_data_dtype=float32,
                 *args,
                 **kwargs):
        # inbound_connections now kept at the DenseUnit level.
        NeuralNetworkFutureComponent.__init__(self,
                                              trial_number=trial_number,
                                              level_number=level_number,
                                              *args,
                                              **kwargs)
        DenseAutoMlStructuralComponent.__init__(self,
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

        # super().__init__(self, *args, **kwargs)

        self.level_prototype = level_prototype
        self.predecessor_levels = predecessor_levels
        self.has_predecessors = has_predecessors
        self.has_successors = has_successors
        self.parallel_units = []
        self.__predecessor_connections_future = []
        self.name = f"{neural_network_future_name}_{self.name}"
        if self.level_number < 0:
            raise ValueError("A negative number ith Level in a hirearchy of "
                             "real objects just doesn't make any sense.")
        self.successor_levels = []
        self.successor_connectivity_errors_2d = jnp.array([])
        self.train_data_dtype = train_data_dtype

    def set_possible_predecessor_connections(self):
        """
        Append's the immediate predecessor level's prototype to the same
        immediate oredecessor's possible_predecessor_connections and sets the
        this Level objet's possible_predecessor_connections to the result of
        this operation.
        """
        # for i, j in zip(np.arange(self.level_number),
        #                 np.arange(self.level_number)[::-1]):
        #     # if len(self.predecessor_levels) != 0:
        #     #     if self.predecessor_levels[int(i)].level_number != int(j):
        #     #         raise ValueError("The list of predecessor_levels passed "
        #     #                          "here is not sequential. The Level "
        #     #                          "instances should be in reverse order "
        #     #                          "starting from level_number = k - 1 "
        #     #                          "going to level_number =  0, where k "
        #     #                          "is the level_number of this layer that "
        #     #                          "raised this exception. In other words, "
        #     #                          "the list goes in parent, grandparent, "
        #     #                          "greatgrandparent, greatgreatgrandparent "
        #     #                          "order.")

        if not self.has_predecessors:
            self.possible_predecessor_connections = []
        else:
            self.possible_predecessor_connections =\
                {f"{self.predecessor_levels[i].level_number}":
                    self.predecessor_levels[i].level_prototype
                 for i in np.arange(len(self.predecessor_levels))}

    def get_possible_predecessor_connections(self):
        return self.possible_predecessor_connections

    def add_unit(self,
                 unit_id_0: int,
                 k: int,
                 v: list):
        k1 = int(k)
        if self.level_number != 0:
            unit_0 =\
                DenseUnit(
                  n_neurons=k1,
                  predecessor_levels=self.predecessor_levels[:self.level_number],
                  possible_predecessor_connections=self.possible_predecessor_connections,
                  parallel_units=self.parallel_units,
                  unit_id=unit_id_0,
                  level_name=self.name,
                  trial_number=self.trial_number,
                  level_number=self.level_number,
                  activation=self.activation,
                  maximum_skip_connection_depth=self.maximum_skip_connection_depth,
                  predecessor_level_connection_affinity_factor_first=self.predecessor_level_connection_affinity_factor_first,
                  predecessor_level_connection_affinity_factor_first_rounding_rule=self.predecessor_level_connection_affinity_factor_first_rounding_rule,
                  predecessor_level_connection_affinity_factor_main=self.predecessor_level_connection_affinity_factor_main,
                  predecessor_level_connection_affinity_factor_main_rounding_rule=self.predecessor_level_connection_affinity_factor_main_rounding_rule,
                  predecessor_level_connection_affinity_factor_decay_main=self.predecessor_level_connection_affinity_factor_decay_main,
                  max_consecutive_lateral_connections=self.max_consecutive_lateral_connections,
                  gate_after_n_lateral_connections=self.gate_after_n_lateral_connections,
                  gate_activation_function=self.gate_activation_function,
                  p_lateral_connection=self.p_lateral_connection,
                  p_lateral_connection_decay=self.p_lateral_connection_decay,
                  num_lateral_connection_tries_per_unit=self.num_lateral_connection_tries_per_unit)
        else:
            print(f"InputLevel.input_shapes {self.input_shapes}")
            unit_0 = \
                InputUnit(input_shape=self.input_shapes[unit_id_0],
                          unit_id=unit_id_0,
                          level_name=self.name,
                          trial_number=self.trial_number,
                          base_models=self.base_models,
                          train_data_dtype=self.train_data_dtype)

        if unit_0.name not in [u.name for u in self.parallel_units]:
            self.parallel_units.append(unit_0)

    def parse_units(self):

        for i in np.arange(len(self.level_prototype)):
            unit_id_num = int(i)
            unit_1_dict = self.level_prototype[unit_id_num]
            print(unit_1_dict)
            for k, v in unit_1_dict.items():
                if "Final" not in self.name:
                    self.add_unit(unit_id_num, k, v)

    def get_parallel_units(self):
        return self.parallel_units

    def set_connectivity_future_prototype(self):
        for unit in self.parallel_units:
            if "InputLevel" not in self.name and self.level_number != 0:
                unit.set_connectivity_future_prototype()

    # Call only filter all levels are instantiated and have their connectivity
    # future and units have been parsed!
    def set_successor_levels(self, successor_levels: list = []):
        successor_levels = self.successor_levels = successor_levels
        for unit_0 in self.parallel_units:
            unit_0.set_successor_levels(self.successor_levels)

    def detect_successor_connectivity_errors(self):
        for unit_0 in self.parallel_units:
            unit_0.detect_successor_connectivity_errors()

        self.successor_connectivity_errors_2d =\
            jnp.array([unit_0.successor_connectivity_errors_2d[0]
                       for unit_0 in self.parallel_units
                       if len(unit_0.successor_connectivity_errors_2d) != 0])

        self.successor_connectivity_errors_2d =\
            jnp.unique(self.successor_connectivity_errors_2d, axis=0)

    def resolve_successor_connectivity_errors(self):
        # Map which element on self.successor_levels (key in this dict)
        # belongs to which level number on the levels themselves
        # (value in this dict)
        if self.successor_connectivity_errors_2d.shape[0] > 0\
                and len(self.successor_levels) > 0:
            successor_map = {}
            for i in np.arange(len(self.successor_levels)):
                successor_map[
                    self.successor_levels[int(i)].level_number] =\
                    str(i)
            print("Successor map:")
            print(successor_map)

            for error_0 in self.successor_connectivity_errors_2d:

                print(
                    f"I am: level #: {self.level_number} calling a correction to an error in:")
                print(f"error_0:{error_0}")
            #
            # for error_0 in self.successor_connectivity_errors_2d:
            #     print(f"error_0:{error_0}")
            #     print(f"I am: level #: {self.level_number}")

                unselected_unit = self.parallel_units[int(error_0[1])]
                num_unselecteds_successor_levels =\
                    len(unselected_unit.successor_levels)
                eligible_successor_level_indexes =\
                    np.arange(jnp.min(jnp.array([num_unselecteds_successor_levels - 1,
                                                 self.minimum_skip_connection_depth]), axis=0),
                              jnp.min(jnp.array(
                                  [num_unselecteds_successor_levels,
                                   self.maximum_skip_connection_depth]), axis=0))
                level_to_assign_to_idx =\
                    np.random.choice([int(i)
                                      for i in eligible_successor_level_indexes])
                level_to_assign_to =\
                    unselected_unit.successor_levels[level_to_assign_to_idx]
                unit_to_assign_idx =\
                    np.random.choice([i for i in np.arange(
                                len(level_to_assign_to.parallel_units))])
                unit_to_assign = level_to_assign_to\
                    .parallel_units[unit_to_assign_idx]
                unit_to_assign\
                    .resolve_successor_connectivity_errors(
                        [unselected_unit.level_number,
                         unselected_unit.unit_id])
                print(
                    f"asigning unit level {unselected_unit.level_number}, unit: {unselected_unit.unit_id} to be the input of: level: {unit_to_assign.level_number} unit: {unit_to_assign.unit_id}")


class DenseLevel(Level,
                 DenseLateralConnectivity):
    """
    Params:
        merging_strategy: str,
        level_prototype: list,
        predecessor_levels: list,
        has_predecessors: str,
        has_successors: str,
        neural_network_future_name: str,
        trial_number: int,
        level_number: int,
        activation: str: defaults to 'elu',
        minimum_skip_connection_depth: int: defaults to 1,
        maximum_skip_connection_depth: int: defaults to 7,
        predecessor_level_connection_affinity_factor_first: int: defauts to 5,
        predecessor_level_connection_affinity_factor_first_rounding_rule: str: defaults to 'ceil',
        predecessor_level_connection_affinity_factor_main: float: defaults to 0.7,
        predecessor_level_connection_affinity_factor_main_rounding_rule: str: defaults to 'ceil',
        predecessor_level_connection_affinity_factor_decay_main: object: defaults to zero_7_exp_decay,
        seed=8675309,
        max_consecutive_lateral_connections: int: defaults to 7,
        gate_after_n_lateral_connections: int: defaults to 3,
        gate_activation_function: object: defaults to simple_sigmoid,
        p_lateral_connection: float: defaults to .97,
        p_lateral_connection_decay: object: defaults to zero_95_exp_decay,
        num_lateral_connection_tries_per_unit: int: defaults to 1,
    public methods:
        set_possible_predecessor_connections:
            parses the variable __possible_predecessor_connections.
        get_possible_predecessor_connections:
            Returns a copy of __possible_predecessor_connections.
        parse_units:
            From the level_prototype and predecessor_levels, parses a Unit
            object for each Unit prescribed in its level_prototype.
    build_level_future:
        sets the variable:
            parallel_units by instantiating the selected number of _Units futures (instantiated, but not materialized)
        materialize:
            calls the .materialize() method on each of its_units objects,
            creating their real respective tf.keras.layers.? object.
         *args,
         **kwargs):
    """

    def __init__(self,
                 level_prototype: list,
                 predecessor_levels: list,
                 has_predecessors: str,
                 has_successors: str,
                 neural_network_future_name: str,
                 trial_number: int,
                 level_number: int,
                 activation='elu',
                 merging_strategy="concatenate",
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
                 *args,
                 **kwargs):
        Level.__init__(
           self,
           level_prototype=level_prototype,
           predecessor_levels=predecessor_levels,
           has_predecessors=has_predecessors,
           has_successors=has_successors,
           neural_network_future_name=neural_network_future_name,
           trial_number=trial_number,
           level_number=level_number,
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
        if self.level_number < 1:
            raise ValueError("Level number 0 is reserved for "
                             "tf.keras.layers.Input objects, and negative "
                             "level numbers don't make any sense. "
                             "level_number must be at least 1")
        self.activation = activation
        self.merging_strategy = merging_strategy
        self.meta_predecessor_connectivity_level_number = jnp.array([])
        self.meta_predecessor_connectivity_unit_id = jnp.array([])

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
        meta_level_number = jnp.array([], dtype=jnp.int32)
        meta_unit_id = jnp.array([], dtype=jnp.int32)
        for unit_0 in self.parallel_units:
            print("debug: meta_level_number")
            meta_level_number =\
                jnp.concatenate(
                    [meta_level_number,
                     unit_0.meta_predecessor_connectivity_level_number],
                    dtype=jnp.int32)
            meta_unit_id =\
                jnp.concatenate(
                    [meta_unit_id,
                     unit_0.meta_predecessor_connectivity_unit_id],
                    dtype=jnp.int32)
        check_unique_units = jnp.column_stack(
            [meta_level_number, meta_unit_id])
        unique_units = jnp.unique(check_unique_units, axis=0)

        self.meta_predecessor_connectivity_level_number = unique_units[:, 0]
        self.meta_predecessor_connectivity_unit_id = unique_units[:, 1]

    def util_set_predecessor_connectivity_metadata(self):
        for unit_0 in self.parallel_units:
            if unit_0.unit_id != 0:
                unit_0.util_set_predecessor_connectivity_metadata()

    def materialize(self):
        self.parse_units()
        self.set_connectivity_future_prototype()
        self.parse_meta_predecessor_connectivity()
        self.set_successor_levels()
        self.detect_successor_connectivity_errors()
        self.resolve_successor_connectivity_errors()
        self.util_set_predecessor_connectivity_metadata()

        for unit_0 in self.parallel_units:
            unit_0.materialize()

        # def build_level_future(self):
        #    self.__predecessor_connections_future = []

        # Test cases passing # Case 1: Instantiate a DenseLevel,
        # see that it inherits from Level
# level_prototype: list, predecessor_levels: list, has_predecessors: str, has_successors: str, *, trial_number, level_number


class InputLevel(Level):
    """Level object for tf.keras.Input layers
    Params:
        input_shapes: list,
                     List of tuples representing the shape
                     of input(s) to the network.
        level_prototype: list,
                        A 1d list of dictionaries where each dictionary
                        represents a tf.keras.layers.[?] object (or
                        mimetic thereof) which the Level consists of:
                example:
                The prototype of a layer consisting of a
                Dense(5) unit and a Dense(3) unit would look
                like this:
                ['5':units.units.Dense,
                '3':units.units.Dense]
         predecessor_levels: list,
                            List of objects of type Layers, list of
                            layers that are predecessors in the chain of
                            Levels.
           has_predecessors: str,
                            options:
                            "yes", "no", "indeterminate"
           has_successors: str,
                            options:
                            "yes", "no", "indeterminate"
           neural_network_future_name: str,
           trial_number: int,
                        Which sequential trial or neural architecture being
                        tried that this object is a member of e.g. if this Level
                        belonged to the 5th neural network that the AutoMl tried,
                        this would be 5.
           level_number: int,
           minimum_skip_connection_depth: int: defaults to 1,
           maximum_skip_connection_depth: int: defaults to 7,
           predecessor_level_connection_affinity_factor_first: int: defualts to 5,
           predecessor_level_connection_affinity_factor_first_rounding_rule: str: defaults to 'ceil',
           predecessor_level_connection_affinity_factor_main: float: defaults to 0.7,
           predecessor_level_connection_affinity_factor_main_rounding_rule: str: defaults to 'ceil',
           predecessor_level_connection_affinity_factor_decay_main: object: defaults to zero_7_exp_decay,
           seed: int: defaults to 8675309,
        """

    def __init__(self,
                 input_shapes: list,
                 level_prototype: list,
                 predecessor_levels: list,
                 has_predecessors: str,
                 has_successors: str,
                 neural_network_future_name: str,
                 trial_number: int,
                 level_number: int,
                 base_models=[''],
                 minimum_skip_connection_depth=1,
                 maximum_skip_connection_depth=7,
                 predecessor_level_connection_affinity_factor_first=5,
                 predecessor_level_connection_affinity_factor_first_rounding_rule='ceil',
                 predecessor_level_connection_affinity_factor_main=0.7,
                 predecessor_level_connection_affinity_factor_main_rounding_rule='ceil',
                 predecessor_level_connection_affinity_factor_decay_main=zero_7_exp_decay,
                 seed=8675309,
                 *args,
                 **kwargs):
        level_prototype = []
        predecessor_levels = []
        has_predecessors = "no"
        has_successors = "yes"
        level_number = 0
        # level_prototype: list, predecessor_levels: list, has_predecessors: str, has_successors: str, trial_number, level_number, /, *, trial_number, level_number
        super().__init__(level_prototype,
                         predecessor_levels,
                         has_predecessors,
                         has_successors,
                         neural_network_future_name,
                         trial_number,
                         level_number,
                         minimum_skip_connection_depth,
                         maximum_skip_connection_depth,
                         predecessor_level_connection_affinity_factor_first,
                         predecessor_level_connection_affinity_factor_first_rounding_rule,
                         predecessor_level_connection_affinity_factor_main,
                         predecessor_level_connection_affinity_factor_main_rounding_rule,
                         predecessor_level_connection_affinity_factor_decay_main,
                         seed=seed,
                         *args,
                         **kwargs)

        self.level_prototype = [{"0": 'InputUnitModule'}
                                for _ in input_shapes]
        self.level_number = 0
        self.base_models = base_models
        self.has_predecessors = "no"
        self.has_successors = "yes"
        self.input_shapes = input_shapes

    def materialize(self):
        for unit_0 in self.parallel_units:
            unit_0.materialize()


class FinalDenseLevel(DenseLevel):
    """docstring for FinalDenseLevel."""

    def __init__(
            self,
            output_shapes: list,
            level_prototype: list,
            predecessor_levels: list,
            neural_network_future_name: str,
            trial_number: int,
            level_number: int,
            # activation set by final_activation
            merging_strategy="concatenate",
            final_activation=None,
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
            *args,
            **kwargs):

        self.output_shapes = output_shapes
        self.predecessor_level_connection_affinity_factor_final_to_kminus1 =\
            predecessor_level_connection_affinity_factor_final_to_kminus1
        self.final_activation = final_activation
        activation = final_activation
        has_predecessors = True
        has_successors = False

        super().__init__(
                level_prototype=level_prototype,
                predecessor_levels=predecessor_levels,
                has_predecessors=has_predecessors,
                has_successors=has_successors,
                neural_network_future_name=neural_network_future_name,
                trial_number=trial_number,
                level_number=level_number,
                activation=activation,
                merging_strategy=merging_strategy,
                minimum_skip_connection_depth=minimum_skip_connection_depth,
                maximum_skip_connection_depth=maximum_skip_connection_depth,
                predecessor_level_connection_affinity_factor_first=predecessor_level_connection_affinity_factor_first,
                predecessor_level_connection_affinity_factor_first_rounding_rule=predecessor_level_connection_affinity_factor_first_rounding_rule,
                predecessor_level_connection_affinity_factor_main=predecessor_level_connection_affinity_factor_main,
                predecessor_level_connection_affinity_factor_main_rounding_rule=predecessor_level_connection_affinity_factor_main_rounding_rule,
                predecessor_level_connection_affinity_factor_decay_main=predecessor_level_connection_affinity_factor_decay_main,
                seed=seed,
                max_consecutive_lateral_connections=max_consecutive_lateral_connections,
                gate_after_n_lateral_connections=gate_after_n_lateral_connections,
                gate_activation_function=gate_activation_function,
                p_lateral_connection=p_lateral_connection,
                p_lateral_connection_decay=p_lateral_connection_decay,
                num_lateral_connection_tries_per_unit=num_lateral_connection_tries_per_unit,
                *args,
                **kwargs)

    def parse_final_units(self):
        for i in np.arange(len(self.output_shapes)):
            i_int = int(i)
            output_shape = self.output_shapes[i_int]
            unit_0 =\
                FinalDenseUnit(
                    output_shape=output_shape,
                    predecessor_levels=self.predecessor_levels,
                    possible_predecessor_connections=self.possible_predecessor_connections,
                    parallel_units=self.parallel_units,
                    unit_id=i_int,
                    level_name=self.name,
                    trial_number=self.trial_number,
                    level_number=self.level_number,
                    final_activation=self.final_activation,
                    maximum_skip_connection_depth=self.maximum_skip_connection_depth,
                    predecessor_level_connection_affinity_factor_first=self.predecessor_level_connection_affinity_factor_first,
                    predecessor_level_connection_affinity_factor_first_rounding_rule=self.predecessor_level_connection_affinity_factor_first_rounding_rule,
                    predecessor_level_connection_affinity_factor_main=self.predecessor_level_connection_affinity_factor_main,
                    predecessor_level_connection_affinity_factor_main_rounding_rule=self.predecessor_level_connection_affinity_factor_main_rounding_rule,
                    predecessor_level_connection_affinity_factor_decay_main=self.predecessor_level_connection_affinity_factor_decay_main,
                    predecessor_level_connection_affinity_factor_final_to_kminus1=self.predecessor_level_connection_affinity_factor_final_to_kminus1,
                    max_consecutive_lateral_connections=self.max_consecutive_lateral_connections,
                    gate_after_n_lateral_connections=self.gate_after_n_lateral_connections,
                    gate_activation_function=self.gate_activation_function,
                    p_lateral_connection=self.p_lateral_connection,
                    p_lateral_connection_decay=self.p_lateral_connection_decay,
                    num_lateral_connection_tries_per_unit=self.num_lateral_connection_tries_per_unit)
            self.parallel_units.append(unit_0)

    def set_final_connectivity_future_prototype(self):
        for unit_0 in self.parallel_units:
            unit_0.set_final_connectivity_future_prototype()

# ->


class RealLevel(NeuralNetworkFutureComponent,
                DenseAutoMlStructuralComponent,
                DenseLateralConnectivity):
    """
    Represents a level of tf.keras.layers.[...] units which are in
    parallel in the hirearchy of the network. For example, there will be
    DenseLevel and InputLevel objects which will inheret from this class.
    As we extend the functionality of the NAS, additional Level objects
    will be added to add support for other types of layers, such as
    Conv2D, Conv1D, Conv3D, GRUs, ... and any avante garde layer objects

    Args:
        level_prototype: list:
                  A 1d list of dictionaries where each dictionary
                  represents a tf.keras.layers.[?] object (or
                  mimetic thereof) which the Level consists of:
        key:
                  unsigned integers where each integer
                  represents the first argument to the
                  respective tf.keras.layers object
                  (or mimetic thereof) that it represents:
        val:
                  the string representation of the
                  name of the class of which the
                  layers.layers.Layers
            example:
                  The prototype of a layer consisting of a
                  Dense(5) unit and a Dense(3) unit would look
                  like this:
                  ['5':units.units.Dense,
                  '3':units.units.Dense]
        predecessor_levels: list:
                  List of objects of type Layers, list of
                  layers that are predecessors in the chain of
                  Levels.
        has_predecessors: str:
            options:
                  "yes", "no", "indeterminate"
        has_successors: str:
            options:
                  "yes", "no", "indeterminate"
        neural_network_future_name: str:
        trial_number: int:
                  Which sequential trial or neural architecture being
                  tried that this object is a member of e.g. if this Level
                  belonged to the 5th neural network that the AutoMl tried,
                  this would be 5.
        level_number: int:
                  Which 0 indexed level in the neural network this Level is
                  (Input level is 0, first DenseLevel is 1, second
                  DenseLevel is 2)
        minimum_skip_connection_depth=1:
                  The DenseLevels will be randomly connected
                  to others in Levels below it. These
                  connections will not be made exclusively to
                  DenseUnits in its immediate successor Level.
                  Some may be 2,3,4,5, ... Levels down. This
                  parameter controls the minimum depth of
                  connection allowed for connections not at
                  the immediate successor Level.
        maximum_skip_connection_depth=7:
                  The DenseLevels will be randomly connected
                  to others in Levels below it. These
                  connections will not be made exclusively to
                  DenseUnits in its immediate successor Level.
                  Some may be 2,3,4,5, ... Levels down. This
                  parameter controls the minimum depth of
                  connection allowed for connections not at
                  the immediate successor Level.
        predecessor_level_connection_affinity_factor_first=5:
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
        predecessor_level_connection_affinity_factor_first_rounding_rule='ceil':
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
        predecessor_level_connection_affinity_factor_main=0.7:
                  For the second and subsequent DenseLevels,
                  if its immediate predecessor level has
                  n DenseUnits, the DenseUnits on this layer will randomly select
                  predecessor_level_connection_affinity_factor_main * n
                  DenseUnits form its immediate predecessor
                  Level to connect to. The selection is
                  WITH REPLACEMENT and a given DenseUnit may
                  be selected multiple times.
        predecessor_level_connection_affinity_factor_main_rounding_rule='ceil':
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
        predecessor_level_connection_affinity_factor_decay_main=lambda x: 0.7 * x:
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
        seed=8675309:
                     *args,
                     **kwargs)

    private variables (to be set internally only but read externally):

        parallel_units:
                  A list of _Units objects
                  (See Unit, DenseUnit, etc)
    Private variables (set and accessed internally only):

        possible_predecessor_connections: list:
                  A 2d list.
                  # k is the level_number of this class
                  # i is the index position on the 2d list.
                  Each i th 1d list within the 2d list is a
                  prototype of the k - i th level.

    public methods:
        set_possible_predecessor_connections:
                  parses the variable __possible_predecessor_connections.
        get_possible_predecessor_connections:
                  Returns a copy of __possible_predecessor_connections.
        parse_units:
                  From the level_prototype and predecessor_levels, parses a Unit
                  object for each Unit prescribed in its level_prototype.

        validate_all_updtream_connected:
                  Checks each dense_unit in the layer's
                  prototype, verifies there is a key for
                  that dense_unit in listed in
                  __predecessor_connections_future. If not,
                  it randomly selects a value from
                  __availible_connections (where
                  level_number is <= max_skip_connection_depth)
                  and creates a dictionary for the dense_unit
                  in __predecessor_connections_future

    """

    def __init__(self,
                 axon_activation: str,
                 min_n_dendrites: int,
                 max_n_dendrites: int,
                 dendrite_activation: str,
                 level_prototype: list,
                 predecessor_levels: list,
                 has_predecessors: str,
                 has_successors: str,
                 neural_network_future_name: str,
                 trial_number: int,
                 level_number: int,
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
                 *args,
                 **kwargs):
        # inbound_connections now kept at the DenseUnit level.
        NeuralNetworkFutureComponent.__init__(self,
                                              trial_number=trial_number,
                                              level_number=level_number,
                                              *args,
                                              **kwargs)
        DenseAutoMlStructuralComponent.__init__(self,
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

        # super().__init__(self, *args, **kwargs)

        self.min_n_dendrites = min_n_dendrites
        self.max_n_dendrites = max_n_dendrites
        self.axon_activation = axon_activation
        self.dendrite_activation = dendrite_activation

        self.level_prototype = level_prototype
        self.predecessor_levels = predecessor_levels
        self.has_predecessors = has_predecessors
        self.has_successors = has_successors
        self.parallel_units = []
        self.__predecessor_connections_future = []
        self.name = f"{neural_network_future_name}_{self.name}"
        if self.level_number < 0:
            raise ValueError("A negative number ith Level in a hirearchy of "
                             "real objects just doesn't make any sense.")
        self.successor_levels = []
        self.successor_connectivity_errors_2d = jnp.array([])

    def set_possible_predecessor_connections(self):
        """
        Append's the immediate predecessor level's prototype to the same
        immediate oredecessor's possible_predecessor_connections and sets the
        this Level objet's possible_predecessor_connections to the result of
        this operation.
        """
        # for i, j in zip(np.arange(self.level_number),
        #                 np.arange(self.level_number)[::-1]):
        #     # if len(self.predecessor_levels) != 0:
        #     #     if self.predecessor_levels[int(i)].level_number != int(j):
        #     #         raise ValueError("The list of predecessor_levels passed "
        #     #                          "here is not sequential. The Level "
        #     #                          "instances should be in reverse order "
        #     #                          "starting from level_number = k - 1 "
        #     #                          "going to level_number =  0, where k "
        #     #                          "is the level_number of this layer that "
        #     #                          "raised this exception. In other words, "
        #     #                          "the list goes in parent, grandparent, "
        #     #                          "greatgrandparent, greatgreatgrandparent "
        #     #                          "order.")

        if not self.has_predecessors:
            self.possible_predecessor_connections = []
        else:
            self.possible_predecessor_connections =\
                {f"{self.predecessor_levels[i].level_number}":
                    self.predecessor_levels[i].level_prototype
                 for i in np.arange(len(self.predecessor_levels))}

    def get_possible_predecessor_connections(self):
        return self.possible_predecessor_connections

    def add_unit(self,
                 unit_id_0: int,
                 k: int,
                 v: list):
        k1 = int(k)
        n_dendrites = int(np.random.randint(self.min_n_dendrites,
                                            self.max_n_dendrites))
        if self.level_number != 0:
            unit_0 =\
                RealNeuron(
                  n_axon_nuerons=k1,
                  axon_activation=self.axon_activation,
                  n_dendrites=n_dendrites,
                  dendrite_activation=self.dendrite_activation,
                  predecessor_levels=self.predecessor_levels[:self.level_number],
                  possible_predecessor_connections=self.possible_predecessor_connections,
                  parallel_units=self.parallel_units,
                  unit_id=unit_id_0,
                  level_name=self.name,
                  trial_number=self.trial_number,
                  level_number=self.level_number,
                  maximum_skip_connection_depth=self.maximum_skip_connection_depth,
                  predecessor_level_connection_affinity_factor_first=self.predecessor_level_connection_affinity_factor_first,
                  predecessor_level_connection_affinity_factor_first_rounding_rule=self.predecessor_level_connection_affinity_factor_first_rounding_rule,
                  predecessor_level_connection_affinity_factor_main=self.predecessor_level_connection_affinity_factor_main,
                  predecessor_level_connection_affinity_factor_main_rounding_rule=self.predecessor_level_connection_affinity_factor_main_rounding_rule,
                  predecessor_level_connection_affinity_factor_decay_main=self.predecessor_level_connection_affinity_factor_decay_main,
                  max_consecutive_lateral_connections=self.max_consecutive_lateral_connections,
                  gate_after_n_lateral_connections=self.gate_after_n_lateral_connections,
                  gate_activation_function=self.gate_activation_function,
                  p_lateral_connection=self.p_lateral_connection,
                  p_lateral_connection_decay=self.p_lateral_connection_decay,
                  num_lateral_connection_tries_per_unit=self.num_lateral_connection_tries_per_unit)
        else:
            unit_0 = \
                InputUnit(input_shape=self.input_shapes[unit_id_0],
                          unit_id=unit_id_0,
                          level_name=self.name,
                          trial_number=self.trial_number,
                          base_models=self.base_models)

        if unit_0.name not in [u.name for u in self.parallel_units]:
            self.parallel_units.append(unit_0)

    def parse_units(self):

        for i in np.arange(len(self.level_prototype)):
            unit_id_num = int(i)
            unit_1_dict = self.level_prototype[unit_id_num]
            print(unit_1_dict)
            for k, v in unit_1_dict.items():
                if "Final" not in self.name:
                    self.add_unit(unit_id_num, k, v)

    def get_parallel_units(self):
        return self.parallel_units

    def set_connectivity_future_prototype(self):
        for unit in self.parallel_units:
            if "InputLevel" not in self.name and self.level_number != 0:
                unit.set_connectivity_future_prototype()

    # Call only filter all levels are instantiated and have their connectivity
    # future and units have been parsed!
    def set_successor_levels(self, successor_levels: list = []):
        successor_levels = self.successor_levels = successor_levels
        for unit_0 in self.parallel_units:
            unit_0.set_successor_levels(self.successor_levels)

    def detect_successor_connectivity_errors(self):
        for unit_0 in self.parallel_units:
            unit_0.detect_successor_connectivity_errors()

        self.successor_connectivity_errors_2d =\
            jnp.array([unit_0.successor_connectivity_errors_2d[0]
                       for unit_0 in self.parallel_units
                       if len(unit_0.successor_connectivity_errors_2d) != 0])

        self.successor_connectivity_errors_2d =\
            jnp.unique(self.successor_connectivity_errors_2d, axis=0)

    def resolve_successor_connectivity_errors(self):
        # Map which element on self.successor_levels (key in this dict)
        # belongs to which level number on the levels themselves
        # (value in this dict)
        if self.successor_connectivity_errors_2d.shape[0] > 0\
                and len(self.successor_levels) > 0:
            successor_map = {}
            for i in np.arange(len(self.successor_levels)):
                successor_map[
                    self.successor_levels[int(i)].level_number] =\
                    str(i)
            print("Successor map:")
            print(successor_map)

            for error_0 in self.successor_connectivity_errors_2d:

                print(
                    f"I am: level #: {self.level_number} calling a correction to an error in:")
                print(f"error_0:{error_0}")
            #
            # for error_0 in self.successor_connectivity_errors_2d:
            #     print(f"error_0:{error_0}")
            #     print(f"I am: level #: {self.level_number}")

                unselected_unit = self.parallel_units[int(error_0[1])]
                num_unselecteds_successor_levels =\
                    len(unselected_unit.successor_levels)
                eligible_successor_level_indexes =\
                    np.arange(jnp.min(jnp.array([num_unselecteds_successor_levels - 1,
                                                 self.minimum_skip_connection_depth]), axis=0),
                              jnp.min(jnp.array(
                                  [num_unselecteds_successor_levels,
                                   self.maximum_skip_connection_depth]), axis=0))
                level_to_assign_to_idx =\
                    np.random.choice([int(i)
                                      for i in eligible_successor_level_indexes])
                level_to_assign_to =\
                    unselected_unit.successor_levels[level_to_assign_to_idx]
                unit_to_assign_idx =\
                    np.random.choice([i for i in np.arange(
                                len(level_to_assign_to.parallel_units))])
                unit_to_assign = level_to_assign_to\
                    .parallel_units[unit_to_assign_idx]
                unit_to_assign\
                    .resolve_successor_connectivity_errors(
                        [unselected_unit.level_number,
                         unselected_unit.unit_id])
                print(
                    f"asigning unit level {unselected_unit.level_number}, unit: {unselected_unit.unit_id} to be the input of: level: {unit_to_assign.level_number} unit: {unit_to_assign.unit_id}")

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
        meta_level_number = jnp.array([], dtype=jnp.int32)
        meta_unit_id = jnp.array([], dtype=jnp.int32)
        for unit_0 in self.parallel_units:
            print("debug: meta_level_number")
            meta_level_number =\
                jnp.concatenate(
                    [meta_level_number,
                     unit_0.meta_predecessor_connectivity_level_number],
                    dtype=jnp.int32)
            meta_unit_id =\
                jnp.concatenate(
                    [meta_unit_id,
                     unit_0.meta_predecessor_connectivity_unit_id],
                    dtype=jnp.int32)
        check_unique_units = jnp.column_stack(
            [meta_level_number, meta_unit_id])
        unique_units = jnp.unique(check_unique_units, axis=0)

        self.meta_predecessor_connectivity_level_number = unique_units[:, 0]
        self.meta_predecessor_connectivity_unit_id = unique_units[:, 1]

    def util_set_predecessor_connectivity_metadata(self):
        for unit_0 in self.parallel_units:
            if unit_0.unit_id != 0:
                unit_0.util_set_predecessor_connectivity_metadata()

    def materialize(self):
        self.parse_units()
        self.set_connectivity_future_prototype()
        self.parse_meta_predecessor_connectivity()
        self.set_successor_levels()
        self.detect_successor_connectivity_errors()
        self.resolve_successor_connectivity_errors()
        self.util_set_predecessor_connectivity_metadata()

        for unit_0 in self.parallel_units:
            unit_0.materialize()


class FinalRealLevel(RealLevel):
    """docstring for FinalDenseLevel."""

    def __init__(
            self,
            axon_activation: str,
            min_n_dendrites: int,
            max_n_dendrites: int,
            output_shapes: list,
            level_prototype: list,
            predecessor_levels: list,
            neural_network_future_name: str,
            trial_number: int,
            level_number: int,
            merging_strategy="concatenate",
            final_activation=None,
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
            *args,
            **kwargs):

        self.output_shapes = output_shapes
        self.predecessor_level_connection_affinity_factor_final_to_kminus1 =\
            predecessor_level_connection_affinity_factor_final_to_kminus1
        self.final_activation = final_activation
        activation = final_activation
        has_predecessors = True
        has_successors = False

        super().__init__(
                axon_activation=axon_activation,
                min_n_dendrites=min_n_dendrites,
                max_n_dendrites=max_n_dendrites,
                dendrite_activation=final_activation,
                level_prototype=level_prototype,
                predecessor_levels=predecessor_levels,
                has_predecessors=has_predecessors,
                has_successors=has_successors,
                neural_network_future_name=neural_network_future_name,
                trial_number=trial_number,
                level_number=level_number,
                activation=activation,
                merging_strategy=merging_strategy,
                minimum_skip_connection_depth=minimum_skip_connection_depth,
                maximum_skip_connection_depth=maximum_skip_connection_depth,
                predecessor_level_connection_affinity_factor_first=predecessor_level_connection_affinity_factor_first,
                predecessor_level_connection_affinity_factor_first_rounding_rule=predecessor_level_connection_affinity_factor_first_rounding_rule,
                predecessor_level_connection_affinity_factor_main=predecessor_level_connection_affinity_factor_main,
                predecessor_level_connection_affinity_factor_main_rounding_rule=predecessor_level_connection_affinity_factor_main_rounding_rule,
                predecessor_level_connection_affinity_factor_decay_main=predecessor_level_connection_affinity_factor_decay_main,
                seed=seed,
                max_consecutive_lateral_connections=max_consecutive_lateral_connections,
                gate_after_n_lateral_connections=gate_after_n_lateral_connections,
                gate_activation_function=gate_activation_function,
                p_lateral_connection=p_lateral_connection,
                p_lateral_connection_decay=p_lateral_connection_decay,
                num_lateral_connection_tries_per_unit=num_lateral_connection_tries_per_unit,
                *args,
                **kwargs)

    def parse_final_units(self):
        for i in np.arange(len(self.output_shapes)):
            i_int = int(i)
            output_shape = self.output_shapes[i_int]
            final_n_axon_neurons = int(np.random.randint(self.min_n_dendrites,
                                                         self.max_n_dendrites))
            unit_0 =\
                FinalRealNeuron(
                    n_axon_nuerons=final_n_axon_neurons,
                    axon_activation=self.axon_activation,
                    output_shape=output_shape,
                    predecessor_levels=self.predecessor_levels,
                    possible_predecessor_connections=self.possible_predecessor_connections,
                    parallel_units=self.parallel_units,
                    unit_id=i_int,
                    level_name=self.name,
                    trial_number=self.trial_number,
                    level_number=self.level_number,
                    final_activation=self.final_activation,
                    maximum_skip_connection_depth=self.maximum_skip_connection_depth,
                    predecessor_level_connection_affinity_factor_first=self.predecessor_level_connection_affinity_factor_first,
                    predecessor_level_connection_affinity_factor_first_rounding_rule=self.predecessor_level_connection_affinity_factor_first_rounding_rule,
                    predecessor_level_connection_affinity_factor_main=self.predecessor_level_connection_affinity_factor_main,
                    predecessor_level_connection_affinity_factor_main_rounding_rule=self.predecessor_level_connection_affinity_factor_main_rounding_rule,
                    predecessor_level_connection_affinity_factor_decay_main=self.predecessor_level_connection_affinity_factor_decay_main,
                    predecessor_level_connection_affinity_factor_final_to_kminus1=self.predecessor_level_connection_affinity_factor_final_to_kminus1,
                    max_consecutive_lateral_connections=self.max_consecutive_lateral_connections,
                    gate_after_n_lateral_connections=self.gate_after_n_lateral_connections,
                    gate_activation_function=self.gate_activation_function,
                    p_lateral_connection=self.p_lateral_connection,
                    p_lateral_connection_decay=self.p_lateral_connection_decay,
                    num_lateral_connection_tries_per_unit=self.num_lateral_connection_tries_per_unit)
            self.parallel_units.append(unit_0)

    def set_final_connectivity_future_prototype(self):
        for unit_0 in self.parallel_units:
            unit_0.set_final_connectivity_future_prototype()
