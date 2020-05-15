import numpy as np
from neural_net.neuron import neuron

class neuron_layer():
    def __init__(self, number_of_neurons, previous_layer_number_of_neurons, transfer_function):
        self.neurons = []
        self.calculated_outputs = np.zeros(number_of_neurons)
        for _ in range(0, number_of_neurons):
            self.neurons.append(neuron(previous_layer_number_of_neurons, transfer_function))

    def __len__(self):
        return len(self.calculated_outputs)

    def calculate_outputs(self, previous_layer_values):
        for i in range(0, len(self.calculated_outputs)):
            self.calculated_outputs[i] = self.neurons[i].calculate_output(previous_layer_values)
        return self.calculated_outputs

    def get_calculated_outputs(self):
        return self.calculated_outputs

    def get_weights(self):
        weights_arrays = []
        for i in range(0, len(self.neurons)):
            weights_arrays.append(self.neurons[i].get_weights())
        return weights_arrays
