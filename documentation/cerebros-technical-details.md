
# How Cerebros Works

## Summary

If the goal of MLPs was to mimic how a biological neuron works, why do we still build neural networks that are structurally similar to the first prototypes from 1989? At the time, it was the closest we could get, but both hardware and software have changed since.

In a biological brain, neurons connect in a multi-dimensional lattice of vertical and lateral connections, which may repeat. Why don't we try to mimic this? In recent years, we got a step closer to this by using single skip connections, but why not simply randomize the connectivity between Dense layers vertically to other Dense layers resting on numerous levels in the network's structure and add lateral connections that overlap and may repeat like a biological brain? (We presume God knew what He was doing, so why re-invent the wheel.)

That is in summary, what we did here. We built a neural architecture search that connects Dense layers in this manner.

See below:

![documentation-assets/Cerebros.png](documentation-assets/Cerebros.png)



The goal here is to recursively generate models consisting of "Levels" which consist of of Dense Layers in parallel, where the Dense layers on one level randomly connect to layers on not only its subsequent Level, but multiple levels below. In addition to these randomized vertical connections, the Dense layers also connect **latrally** at random to not only their neighboring layer, but to layers multiple layers to the right of them (remember, this architectural pattern consists of "Levels" of Dense layers. The Dense layers make lateral connections to the other Dense layers in the same level, and vertial connections to Dense layers in their level's successor levels). There may also be more than one connection between a given Dense layer and another, both laterally and vertically, which if you have the patience to follow the example neural architecture created by the Ames housing data example below, (I dare you to try following the connectivity in that), you will probably see many instances where this occurs. This may allow more complex networks to gain deeper, more granular insight on smaller data sets before problems like internal covariate shift, vanishing gradients, and exploding gradients drive overfitting, zeroed out weights, and "predictions of [0 | infinity] for all samples". Bear in mind that the deepest layers of a Multi - Layer Perceptron will have the most granular and specific information about a given data set.  In recent years, we got a step closer that this does by using single skip connections, but why not simply randomize the connectivity to numerous levels in the network's structure altogether and add lateral connections that overlap like a biological brain? (We presume God knew what He was doing.)

## How this all works:

We start with some basic structural components:

The SimpleCerebrosRandomSearch: The core auto-ML that recursively creates the neural networks, vicariously through the NeuralNetworkFuture object:

NeuralNetworkFuture:
  - A data structure that is essentially a wrapper around the whole neural network system.
  - You will note the word "Future" in the name of this data structure. This is for a reason. The solution to the problem of recursively parsing neural networks having topologies similar to the connectivity between biological neurons involves a chicken before the egg problem. Specifically this was that randomly assigning neural connectivity will create some errors and disjointed graphs, which once created is impractical correct without starting over. Additionally, I have found that for some reason, graphs having some dead ends, but not completely disjointed train very slowly.
  - The probability of a given graph passing on the first try is very low, especially if you are building a complex network, nonetheless, the randomized vertical and lateral, and potentially repeating connections are the key to making this all work.
  - The solution was to create a Futures objects that first planned how many levels of Dense layers would exist in the network, how many Dense layers each level would consist of, and how many neurons each Dense layer would consist of, allowing the random connections to be tentatively selected, but not materialized. Then it applies a protocol to detect and resolve any inconsistencies, disjointed connections, etc in the planned conductivities before any actual neural network components are actually materialized.
  - A list of errors is compiled and another protocol appends the planned connectivity with additional connections to fix the breaks in the connectivity.
  - Lastly, once the network's connectivity has been validated, it then materializes the Dense layers of the neural network per the conductivities planned, resulting in a neural network ready to be trained.

Level:
  - A data structure adding a new layer of abstraction above the concept of a Dense layer, which is a wrapper consisting of multiple instances of a future for what we historically called a Dense layer in a neural network. A level will consist of multiple Dense Units, which each will materialize to a Dense Layer. Since we are making both vertical and lateral connections, the term "Layer" loses relevance it has in the traditional sequential MLP context. A NeuralNetworkFuture has many Levels, and a Level belongs to a NeuralNetworkFuture. A Level has many Units, and a Unit belongs to a Level.

Unit:
  - A data structure which is a future for a single Dense layer. A Level has many Units, and a Unit belongs to a Level.

![documentation-assets/Cerebros.png](documentation-assets/Cerebros.png)

Here are the steps to the process:

0. Some nomenclature:
  1. k refers to the Level number being immediately discussed.
  2. l refers to the number of DenseUnits the kth Level has.
  3. k-1 refers to the immediate predecessor Level (Parent Level) number of the kth level of the level being discussed.
  4. n refers to the number of DenseUnits the kth Level's parent predecessor has.
1. SimpleCerebrosRandomSearch.run_random_search().
    1. This calls SimpleCerebrosRandomSearch.parse_neural_network_structural_spec_random() which chooses the following random unsigned integers:
        1. How many Levels, the archtecture will consist of;
        2. For each Level, how many Units the levl will consist of;
        3. For each unit, how many neurons the Dense layer it will materialize will consist of.
        4. This is parsed into a dictionary as a high-level specification for the nodes, but not edges called a neural_network_spec.
        5. This will instantiate a NeuralNetworkFuture of the selected specification for number of Levels, Units per Level, and neurons per Unit, and the NeuralNetworkFuture takes the neural_network_spec as an argument.
        6. This entire logic will repeat once for each number in range of: number_of_architecture_moities_to_try.
        7. Step 5 will repea multiple times, for each same neural_network_spec, once for each number in range: number_of_tries_per_architecture_moity.
        8. All replictions are done as separate Python multiprocessing proces (multiple workers in parallel on separate processor cores).
2. **(This is a top down operation starting with InputLevel and proceeding to the last hidden layer in the network)** In each NeuralNetworkFuture, the neural_network_spec will be iterated through, instantiating a DenseLevel object for each element in the dictionary , which will be passed as the argument level_prototype. Each will be linked to the last, and each will maintain access to the same chain of Levels as the list predecessor_levels (The whole thing is essentially like a list, having many necessary nested elements).
3. A dictionary of possible predecessor connections will also be parsed. This is a symbolic representation of the levels and units above it that is faster to iterate through than the actual Levels and Units objects themselves.
4. SimpleCerebrosRandomSearch calls each NeuralNetworkFuture.materialize() method which calls NeuralNetworkFuture.parse_units(), method, which recursively calls the .parse_units() belonging to each Levels object. Within each Levels object, this will iterate through its level_prototype list and will instantiate a DenseUnits object for each item and append DenseLevel.parallel_units with it.
5. NeuralNetworkFuture.materialize() calls each NeuralNetworkFuture object's NeuralNetworkFuture.set_connectivity_future_prototype() method. ... **(This is a bottom - up operation starting with the last hidden layer and and proceeding to the InputLevel)** This will recursively call each Levels object's determine_upstream_random_connectivity(). Which will trigger each layer to recursively call each of its constituent DenseUnit objects' set_connectivity_future_prototype() method. Each DenseUnit will calculate how many connections to make to DenseUnits in each predecessor Level and will select that many units from its possible_predecessor_connections dictionary, from each of those levels, appending each to its __predecessor_connections_future list.
7. Once the random predecessor connections have been selected, now, the system will then need to validate the DOWNSTREAM network connectivity of each Dense unit and repair any breaks that would cause a disjointed graph.  (verify that each DenseUnit has at least one connection to a SUCCESSOR Layer's DenseUnit). Here is why: If the random connections were chosen bottom - up, (which is the lesser of two evils) AND each DenseUnit will always select at least one PREDECESSOR connection (I ether validate this to throw an error if the number of connections to the immediate predecessor layer is less than 1 OR it just coerce it to 1 if 0 is calculated, with this said, then upstream connectivity is not possible to break, however, it is inherently possible that at least one predecessor unit was NOT selected by any of its successor's randomly selected connections. (especially if a low value is selected for the predecessor_level_connection_affinity_factor_main), something, I speculate may be advantageous to do, as some research has indicate that making sparse connections can outperform Dense connections, which is what we aim to do. We want not all connections to be made to the immediate successor Level. We want some to connect 2 Levels downstream, 3, ...  4, ... 5, Levels downstream ... The trouble is that the random connections that facilitate this can "leave out" a DenseUnit in a predecessor Level as not being picked at all by any DenseUnit in a SUCCESSOR level, leaving us with a dead end in the network, hence a disjointed graph. There are 3 rules this has to follow:
    1. **Rule 1:** Each DenseUnit must connect to SOMETHING upstream (PREDECESSORS) WITHIN max_skip_connection_depth layers of its immediate predecessor. (Random bottom - up assignment can't accidentally violate this rule if it always selects at least one selection, so nothing to worry about or validate here).
    2. **Rule 2:** Everything must connect to something something DOWNSTREAM (successors) within max_skip_connection_depth Levels of its level_number. **(Random bottom - up assignment will frequently leave violations of this rule behind. Where this happens, these should either be connected to a randomly selected DenseUnit max_skip_connection_depth layers below | **or a randomly selected DenseUnit residing a randomly chosen number of layers below in the range of [minimum_skip_connection_depth, maximum_skip_connection_depth] below when possible | and if necessary, to the last hidden DenseLevel or the output level)**.
    3. Now the third rule **Rule 3:** The connectivity must flow in only one direction vertically and one direction laterally (on a layer - by - layer basis). In other words, a kth Level's DenseUnit can't take its own output as one of its inputs. Nor can a kth Level's DenseUnit take its k+[any number]th successor's output as one of its inputs (because it is also a function of the kth Level's DenseUnit's own output). This would obviously be a contradiction, like filling the tank of an empty fire truck using the fire hose that draws water from its own tank.. or an empty fuel pump at a fuel station filling its own empty tank using the hose that goes in a car's gas tank, ... and gets that fuel from its own tank... Fortunately, both the logic setting the vertical connectivity and the logic setting the lateral connectivity both can't create this inconsistency, so there is nothing to worry about or validate here either. We only have to screen for and fix violations of rule 2, "Every DenseUnit must connect to some (DOWNSTREAM / SUCCESSOR) DenseUnit within max_skip_connection_depth of itself. Here is the logic for this validation and rectification:   
8. The test to determine whether there are violations of rule 2 desctibed above is done by DenseLevel objects:
  1. Scenario 1: **(If the kth DenseLayer is the last Layer)**:
    1. For each layer having a  **layer_number >= k - maximum_skip_connection_depth** (look at possible_predecessor_connections):
    2. For each DenseUnit in said layer, check each successor layers' Dense units' __predecessor_connections_future list for whether the DenseUnit making the check has been selected. If at least one connection is found: The check passes, else, the kth layer will add itself to the __predecessor_connections_future belonging to a successor meeting the constraint in 7.1.1.
    3. Scenario 2: **(If the kth DenseLayer is not the last Layer)** Do the same as scenario 1, EXCEPT, only check the layer where **its layer number == (k - maximum_skip_connection_depth) for a pre-existing selection for connection.**.    
9. Lateral connectivity: For each DesnseLayer:
  7.1 For each DenseUnit:
  7.2 Calculate the number of lateral connections to make.
  7.3 Randomly select a number of units from level_prototype where unit is > the the current unit's unit id and the connection doesn't violate the max_consecutive_lateral_connections setting. If there has been > gate_after_n_lateral_connections, then apply the gating function and restart the count for this rule. There is nothing to validate here. If this is set up to only allow right OR left connections, then this can't create a disjointed graph or contradiction. Now all connectivities are planned.
10. Materialize Dense layers. **(This is a top down operation starting with InputLevel and proceeding to the last hidden layer in the network)**
11. Create output layer.
12. Compile model.
13. Fit the model, save the oracles, and save the model as a saved Keras model.
14. Iterate through the results and find best the model ad metrics.  
