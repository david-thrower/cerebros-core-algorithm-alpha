"""DenseAutoMlStructuralComponent: A base class that the NeuralNetworkFuture, Layers, Units, etc inherit from."""

import tensorflow as tf
import jax.numpy as jnp
from jax import jit
import numpy as np


@jit
def jit_zero_7_exp_decay(x):
    return 0.7 ** x


def zero_7_exp_decay(x):
    if x == 0:
        return 1
    return float(jit_zero_7_exp_decay(x))


@jit
def jit_zero_95_exp_decay(x):
    return 0.95 ** x


def zero_95_exp_decay(x):
    if x == 0:
        return 1
    return float(jit_zero_95_exp_decay(x))


@jit
def jit_sigmoid(x):
    s = 1/(1+jnp.exp(-x))
    return s


def simple_sigmoid(x):
    s = jit_sigmoid(x)
    return s.__float__()


# @jit


class DenseAutoMlStructuralComponent:
    """DenseAutoMlStructuralComponent: A base class that the NeuralNetworkFuture, Layers, Units, etc inherit from.
    Args:
                    minimum_skip_connection_depth=1:
                                      The DenseLevels will be randomly connected
                                      to others in Levels below it. These
                                      connections will not be made exclusively to
                                      DenseUnits in its immediate successor Level.
                                      Some may be 2,3,4,5, ... Levels down. This
                                      parameter controls the minimum depth of
                                      connection allowed for connections not at
                                      the immediate successor Level.
                     maximum_skip_connection_depth=7,
                                      The DenseLevels will be randomly connected
                                      to others in Levels below it. These
                                      connections will not be made exclusively to
                                      DenseUnits in its immediate successor Level.
                                      Some may be 2,3,4,5, ... Levels down. This
                                      parameter controls the minimum depth of
                                      connection allowed for connections not at
                                      the immediate successor Level.
                     predecessor_level_connection_affinity_factor_first=5,
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
                     predecessor_level_connection_affinity_factor_main=0.7,
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
                      seed: int: defaults to 8675309,
                     *args,
                     **kwargs
    """

    def __init__(self,
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

        self.minimum_skip_connection_depth = minimum_skip_connection_depth
        self.maximum_skip_connection_depth = maximum_skip_connection_depth
        self.predecessor_level_connection_affinity_factor_first = \
            predecessor_level_connection_affinity_factor_first
        self.predecessor_level_connection_affinity_factor_first_rounding_rule =\
            predecessor_level_connection_affinity_factor_first_rounding_rule
        self.predecessor_level_connection_affinity_factor_main = \
            predecessor_level_connection_affinity_factor_main
        self.predecessor_level_connection_affinity_factor_main_rounding_rule =\
            predecessor_level_connection_affinity_factor_main_rounding_rule
        self.predecessor_level_connection_affinity_factor_decay_main =\
            predecessor_level_connection_affinity_factor_decay_main
        self.seed = seed


#@jit
class DenseLateralConnectivity:
    """DenseLateralConnectivity:
    Params:
         max_consecutive_lateral_connections: int: defaults to: 7:
            The maximum number of consecutive Units on a Level that make
            a lateral connection. Setting this too high can cause exploding /
            vanishing gradients and internal covariate shift. Setting this too
            low may miss some patterns the network may otherwise pick up.
         gate_after_n_lateral_connections: int: defaults to: 3,
            After this many kth consecutive lateral connecgtions, gate the
            output of the k + 1 th DenseUnit before creating a subsequent
            lateral connection.
         gate_activation_function=tf.keras.activation.sigmoid:
            Which activation function to gate the output of one DenseUnit
            before making a lateral connection.
         p_lateral_connection: int: defaults to: 1:
            The probability of the first given DenseUnit on a level making a
            lateral connection with the second DenseUnit.
         p_lateral_connection_decay: object: defaults to: lambda x: 1 ** x
            A function that descreases or increases the probability of a
            lateral connection being made with a subsequent DenseUnit.
            Accepts an unsigned integer x. returns a floating point number that
            will be multiplied by p_lateral_connection where x is the number
            of subsequent connections after the first
         num_lateral_connection_tries_per_unit: int: defaults to 1,
          *args,
          **kwargs
         Public methods
    """

    def __init__(self,
                 max_consecutive_lateral_connections=7,
                 gate_after_n_lateral_connections=3,
                 gate_activation_function=simple_sigmoid,
                 p_lateral_connection=.97,
                 p_lateral_connection_decay=zero_95_exp_decay,
                 num_lateral_connection_tries_per_unit=1,
                 *args,
                 **kwargs):
        self.max_consecutive_lateral_connections =\
            max_consecutive_lateral_connections
        self.gate_after_n_lateral_connections =\
            gate_after_n_lateral_connections
        self.gate_activation_function = gate_activation_function
        self.__p_lateral_connection = p_lateral_connection  # not updated
        self.p_lateral_connection = p_lateral_connection  # Updated
        self.p_lateral_connection_decay = p_lateral_connection_decay
        self.num_lateral_connection_tries_per_unit = \
            num_lateral_connection_tries_per_unit
        self.n_consecutive_ungated_connections = 0
        self.n_consecutive_connections = 0

    def gate_or_not(self):
        if self.n_consecutive_ungated_connections % \
                self.gate_after_n_lateral_connections == 0 and\
                self.n_consecutive_ungated_connections != 0:
            self.n_consecutive_ungated_connections = 0
            return True
        self.n_consecutive_ungated_connections += 1
        return False

    def select_connection_or_not(self, k_minus_n):
        """Determines whether to make a lateral connection. This should be
        called once for each possible connection.
        Arg:
            k_minus_n: How many units to the left, a possible lateral
            connection is upstream the possible connection is
        Returns: A boolean where p(True) = self.p_lateral_connection *
                 self.p_lateral_connection_decay(k_minus_n). In other
                 words, if self.p_lateral_connection = .98 and
                 self.p_lateral_connection_decay is '0.9 ** x',
                 and k_minus_n is set to 3, then the probability of
                 True being returned is 0.98 * 0.9 ** 3, or 0.71"""
        p_connect_this_time = self.p_lateral_connection *\
            self.p_lateral_connection_decay(k_minus_n)
        ran_0_1 = float(np.random.random())
        conn_or_not = self.n_consecutive_connections <= \
            self.max_consecutive_lateral_connections - 1 and \
            ran_0_1 <= p_connect_this_time
        if conn_or_not:
            self.n_consecutive_connections += 1
        else:
            self.n_consecutive_connections = 0
        return conn_or_not
