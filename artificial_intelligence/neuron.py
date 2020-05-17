import numpy as np
from artificial_intelligence.transfer_function import sigmoid

class neuron():
    def __init__(self, previous_layer_number_of_neurons, transfer_function):
        self.calculated_output = 0.0
        self.weights = np.zeros(previous_layer_number_of_neurons + 1) # extra spot for bias
        self.transfer_function = transfer_function

    def calculate_output(self, previous_layer_values):
        bias_weight = self.weights[0]
        neuron_weights = self.weights[1:]
        assert(len(previous_layer_values) == len(neuron_weights))
        weighted_sum = np.sum(np.dot(previous_layer_values, neuron_weights)) + bias_weight
        self.calculated_output = self.transfer_function(weighted_sum)
        return self.calculated_output

    def get_weights(self):
        return self.weights.tolist()
    
    def randomize_weights(self, percentage):
        fraction = percentage / 100.0
        mask = np.random.rand(len(self.weights)) < fraction # weights length number of random doubles between 0 and 1 that are < franction; these should be changed
        new_weights = np.random.rand(len(self.weights))
        for i in range(0, len(self.weights)):
            if mask[i]:
                self.weights[i] = new_weights[i]

if __name__ == "__main__":
    my_neuron = neuron(5, sigmoid)