import numpy as np
from neural_net.neuron_layer import neuron_layer

class neural_network():
    def __init__(self, transfer_function, number_of_inputs, number_of_outputs, number_of_hidden_layers = 0, number_of_hidden_layer_neurons = 0):
        self.layers = []
        self.number_of_inputs = number_of_inputs
        previous_layer_neurons = number_of_inputs
        for i in range(0, number_of_hidden_layers + 1):
            current_layer_neurons = number_of_hidden_layer_neurons if i < number_of_hidden_layers else number_of_outputs
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