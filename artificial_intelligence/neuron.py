import numpy as np

try:
    from artificial_intelligence.transfer_function import sigmoid # "app" case
except:
    from transfer_function import sigmoid # "__main__" case

class Neuron():
    def __init__(self, weights_np_array, transfer_function):
        self.weights = weights_np_array
        self.transfer_function = transfer_function

    @classmethod
    def fromEmpty(cls, number_of_inputs, transfer_function):
        weights = np.zeros(number_of_inputs + 1) # extra spot for bias
        return cls(weights, transfer_function)
    
    @classmethod
    def fromRandom(cls, number_of_inputs, transfer_function):
        obj = cls.fromEmpty(number_of_inputs, transfer_function)
        obj.randomize_weights(100)
        return obj
    
    @classmethod
    def fromWeights(cls, weights_param, transfer_function):
        if isinstance(weights_param, np.ndarray):
            return cls(weights_param, transfer_function)
        if isinstance(weights_param, list):
            weights = np.array(weights_param)
            return cls(weights, transfer_function)
        raise TypeError
    
    def __len__(self):
        return len(self.weights)

    def calculate_output(self, input_values):
        bias_weight = self.weights[0]
        neuron_weights = self.weights[1:]
        assert(len(input_values) == len(neuron_weights))
        weighted_sum = np.sum(np.dot(input_values, neuron_weights)) + bias_weight
        calculated_output = self.transfer_function(weighted_sum)
        return calculated_output

    def get_weights(self):
        '''
        returns an array of doubles which are the weights belonging to the previous layer outputs
        '''
        return self.weights
    
    def randomize_weights(self, percentage):
        fraction = percentage / 100.0
        mask = np.random.rand(len(self.weights)) < fraction # weights length number of random doubles between 0 and 1 that are < franction; these should be changed
        new_weights = np.random.rand(len(self.weights))
        for i in range(0, len(self.weights)):
            if mask[i]:
                self.weights[i] = new_weights[i]

if __name__ == "__main__":
    my_neuron = Neuron.fromEmpty(5, sigmoid)

    weights = [0, 0, 1]
    weights_arr = np.array(weights)
    my_neuron = Neuron.fromWeights(weights, sigmoid)
    my_neuron = Neuron.fromWeights(weights_arr, sigmoid)
    my_neuron = Neuron.fromWeights(1, sigmoid)