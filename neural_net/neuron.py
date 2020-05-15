import numpy as np

class neuron():
    def __init__(self, previous_layer_number_of_neurons, transfer_function):
        self.calculated_output = 0.0
        self.weights = np.zeros(previous_layer_number_of_neurons)
        self.transfer_function = transfer_function

    def calculate_output(self, previous_layer_values):
        assert(len(previous_layer_values) == len(self.weights))
        weighted_sum = np.sum(np.dot(previous_layer_values, self.weights))
        self.calculated_output = self.transfer_function(weighted_sum)
        return self.calculated_output

    def get_weights(self):
        return self.weights.tolist()