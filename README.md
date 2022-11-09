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
