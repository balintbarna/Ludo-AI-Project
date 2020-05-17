import numpy as np
from artificial_intelligence.neuron_layer import neuron_layer
from artificial_intelligence.transfer_function import *

class neural_network():
    def __init__(self, number_of_inputs, layers_list):
        self.layers = layers_list
        self.number_of_inputs = number_of_inputs
    
    @classmethod
    def fromEmpty(cls, transfer_function, number_of_inputs, number_of_neurons_per_layer):
        layers_list = []
        number_of_layers = len(number_of_neurons_per_layer)
        previous_layer_neurons = number_of_inputs
        for i in range(0, number_of_layers):
            current_layer_neurons = number_of_neurons_per_layer[i]
            layers_list.append(neuron_layer.fromEmpty(current_layer_neurons, previous_layer_neurons, transfer_function))
            previous_layer_neurons = current_layer_neurons
        return cls(number_of_inputs, layers_list)
    
    @classmethod
    def fromRandom(cls, transfer_function, number_of_inputs, number_of_neurons_per_layer):
        layers_list = []
        number_of_layers = len(number_of_neurons_per_layer)
        previous_layer_neurons = number_of_inputs
        for i in range(0, number_of_layers):
            current_layer_neurons = number_of_neurons_per_layer[i]
            layers_list.append(neuron_layer.fromRandom(current_layer_neurons, previous_layer_neurons, transfer_function))
            previous_layer_neurons = current_layer_neurons
        return cls(number_of_inputs, layers_list)
    
    @classmethod
    def fromWeights(cls, transfer_function, weights_list_list_list):
        number_of_inputs = len(weights_list_list_list[0][0]) # number of weights in the first neuron in the first layer
        layers_list = []
        for weights_list_list in weights_list_list_list:
            layers_list.append(neuron_layer.fromWeights(weights_list_list, transfer_function))
        return cls(number_of_inputs, layers_list)
        
    def calculate_outputs(self, inputs):
        assert len(inputs) == self.number_of_inputs
        previous_layer_outputs = inputs
        for i in range(0, len(self.layers)):
            previous_layer_outputs = self.layers[i].calculate_outputs(previous_layer_outputs)
        return previous_layer_outputs
    
    def get_weights(self):
        '''
        returns a list of a list of a list of doubles, which are all the weights of all the neurons in all the layers
        '''
        weights_arrays_of_arrays = []
        for i in range(0, len(self.layers)):
            weights_arrays_of_arrays.append(self.layers[i].get_weights())
        return weights_arrays_of_arrays

    def randomize_weights(self, percentage):
        for i in range(0, len(self.layers)):
            self.layers[i].randomize_weights(percentage)

if __name__ == "__main__":
    # only one output
    inputs = 10
    outputs = 1
    neurons_per_layer = [outputs]
    ann = neural_network.fromEmpty(sigmoid, inputs, neurons_per_layer)
    # few hidden layers
    inputs = 10
    outputs = 1
    neurons_per_layer = [5,5,5] # 3 hidden layers of 5 neurons
    neurons_per_layer.append(outputs)
    ann = neural_network.fromEmpty(sigmoid, inputs, neurons_per_layer)
