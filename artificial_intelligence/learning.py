import numpy as np

def create_child(weights_network1, weights_network2):
    # take the two networks
    network_dad = weights_network1
    network_mom = weights_network2
    assert_length(network_dad, network_mom)
    network_child = []
    # iterate through layers
    for i_network in range(0, len(network_dad)):
        layer_dad = network_dad[i_network]
        layer_mom = network_mom[i_network]
        assert_length(layer_dad, layer_mom)
        layer_child = []
        # iterate through neurons
        for i_layer in range(0, len(layer_dad)):
            neuron_dad = layer_dad[i_layer]
            neuron_mom = layer_mom[i_layer]
            assert_length(neuron_dad, neuron_mom)
            neuron_child = []
            # iterate through weights
            for i_neuron in range(0, len(neuron_dad)):
                weight_dad = neuron_dad[i_neuron]
                weight_mom = neuron_mom[i_neuron]
                weight_child = pick_random(weight_dad, weight_mom)
                neuron_child.append(weight_child)
            assert_length(neuron_dad, neuron_child)
            layer_child.append(neuron_child)
        assert_length(layer_dad, layer_child)
        network_child.append(layer_child)
    assert_length(network_dad, network_child)
    return network_child

def introduce_mutation(weights):
    for index, elem in enumerate(weights):
        if isinstance(elem, list):
            introduce_mutation(elem)
        else:
            max_mutation_amount = 1
            mutation = np.random.rand() * 2 - 1
            mutation *= max_mutation_amount
            weights[index] += mutation

def assert_length(one, two):
    assert len(one) == len(two)

def pick_random(one, two):
    pick = np.random.randint(2)
    result = one if pick == 0 else two
    return result


if __name__ == "__main__":
    weights = [[[3, 2, 1],[1,2,3]],[[0,1,2], [-1, -2, -3]]]
    introduce_mutation(weights)
    print(weights)
    weights_dad = [[[3, 2, 1],[1,2,3]],[[0,1,2], [-1, -2, -3]]]
    weights_mom = [[[8,9,10],[4,5,6]],[[-10,-1,-2], [9,9,9]]]
    child = create_child(weights_dad, weights_mom)
    print(child)
