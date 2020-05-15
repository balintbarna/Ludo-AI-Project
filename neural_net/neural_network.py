import numpy as np
from neural_net.neuron_layer import neuron_layer
from neural_net.transfer_function import *

class neural_network():
    def __init__(self, transfer_function, number_of_inputs, number_of_neurons_per_layer):
        number_of_layers = len(number_of_neurons_per_layer)
        self.layers = []
        self.number_of_inputs = number_of_inputs
        previous_layer_neurons = number_of_inputs
        for i in range(0, number_of_layers):
            current_layer_neurons = number_of_neurons_per_layer[i]
            self.layers.append(neuron_layer(current_layer_neurons, previous_layer_neurons, transfer_function))
            previous_layer_neurons = current_layer_neurons
        
    def calculate_outputs(self, inputs):
        assert len(inputs) == self.number_of_inputs
        previous_layer_outputs = inputs
        for i in range(0, len(self.layers)):
            previous_layer_outputs = self.layers[i].calculate_outputs(previous_layer_outputs)
        return previous_layer_outputs
    
    def get_weights(self):
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
    ann = neural_network(sigmoid, inputs, neurons_per_layer)
    # few hidden layers
    inputs = 10
    outputs = 1
    neurons_per_layer = [5,5,5] # 3 hidden layers of 5 neurons
    neurons_per_layer.append(outputs)
    ann = neural_network(sigmoid, inputs, neurons_per_layer)
