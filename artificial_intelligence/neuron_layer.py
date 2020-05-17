import numpy as np
from artificial_intelligence.neuron import neuron
from artificial_intelligence.transfer_function import sigmoid

class neuron_layer():
    def __init__(self, neurons_list):
        self.neurons = neurons_list
        self.calculated_outputs = np.zeros(len(neurons_list))

    @classmethod
    def fromEmpty(cls, number_of_neurons, previous_layer_number_of_neurons, transfer_function):
        neurons_list = []
        for _ in range(0, number_of_neurons):
            neurons_list.append(neuron.fromEmpty(previous_layer_number_of_neurons, transfer_function))
        return cls(neurons_list)

    @classmethod
    def fromRandom(cls, number_of_neurons, previous_layer_number_of_neurons, transfer_function):
        neurons_list = []
        for _ in range(0, number_of_neurons):
            neurons_list.append(neuron.fromRandom(previous_layer_number_of_neurons, transfer_function))
        return cls(neurons_list)

    @classmethod
    def fromWeights(cls, weights_list_list, transfer_function):
        neurons_list = []
        for weights_list in weights_list_list:
            neurons_list.append(neuron.fromWeights(weights_list, transfer_function))
        return cls(neurons_list)

    def __len__(self):
        return len(self.calculated_outputs)

    def calculate_outputs(self, previous_layer_values):
        for i in range(0, len(self.calculated_outputs)):
            self.calculated_outputs[i] = self.neurons[i].calculate_output(previous_layer_values)
        return self.calculated_outputs

    def get_calculated_outputs(self):
        return self.calculated_outputs

    def get_weights(self):
        '''
        returns a list of a list of doubles, that are all the weights of all the neurons in this layer
        '''
        weights_arrays = []
        for i in range(0, len(self.neurons)):
            weights_arrays.append(self.neurons[i].get_weights())
        return weights_arrays

    def randomize_weights(self, percentage):
        for i in range(0, len(self.neurons)):
            self.neurons[i].randomize_weights(percentage)

if __name__ == "__main__":
    my_layer = neuron_layer.fromEmpty(5, 5, sigmoid)