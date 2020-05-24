import numpy as np
try:
    from artificial_intelligence.neuron_layer import NeuronLayer
    from artificial_intelligence.transfer_function import *
except:
    from neuron_layer import NeuronLayer
    from transfer_function import *

class NeuralNetwork():
    def __init__(self, input_array, layers_list):
        self._input = input_array
        self._layers = layers_list
    
    @classmethod
    def fromEmpty(cls, transfer_function, number_of_inputs, number_of_neurons_per_layer):
        layers_list = []
        number_of_layers = len(number_of_neurons_per_layer)
        for i in range(0, number_of_layers):
            current_layer_neurons = number_of_neurons_per_layer[i]
            layers_list.append(NeuronLayer.fromEmpty(current_layer_neurons, number_of_inputs, transfer_function))
            number_of_inputs = current_layer_neurons
        input_array = np.zeros(number_of_inputs)
        return cls(input_array, layers_list)
    
    @classmethod
    def fromRandom(cls, transfer_function, number_of_inputs, number_of_neurons_per_layer):
        layers_list = []
        number_of_layers = len(number_of_neurons_per_layer)
        for i in range(0, number_of_layers):
            current_layer_neurons = number_of_neurons_per_layer[i]
            layers_list.append(NeuronLayer.fromRandom(current_layer_neurons, number_of_inputs, transfer_function))
            number_of_inputs = current_layer_neurons
        input_array = np.zeros(number_of_inputs)
        return cls(input_array, layers_list)
    
    @classmethod
    def fromWeights(cls, transfer_function, weights_list_list_arr):
        number_of_inputs = len(weights_list_list_arr[0][0]) - 1 # number of weights in the first neuron in the first layer minus the bias weight
        input_array = np.zeros(number_of_inputs)
        layers_list = []
        for weights_list_list in weights_list_list_arr:
            layers_list.append(NeuronLayer.fromWeights(weights_list_list, transfer_function))
        return cls(input_array, layers_list)

    def __len__(self):
        return len(self._layers)
    
    def get_input_array(self):
        return self._input
        
    def calculate_outputs(self):
        previous_layer_outputs = self._input
        for i in range(0, len(self)):
            previous_layer_outputs = self.layers[i].calculate_outputs(previous_layer_outputs)
        return previous_layer_outputs
    
    def get_weights(self):
        '''
        returns a list of a list of a list of doubles, which are all the weights of all the neurons in all the layers
        '''
        weights_list_list_list = []
        for i in range(0, len(self)):
            weights_list_list_list.append(self.layers[i].get_weights())
        return weights_list_list_list

    def randomize_weights(self):
        for i in range(0, len(self)):
            self.layers[i].randomize_weights()

if __name__ == "__main__":
    # only one output
    inputs = 10
    outputs = 1
    neurons_per_layer = [outputs]
    ann = NeuralNetwork.fromEmpty(sigmoid, inputs, neurons_per_layer)
    # few hidden layers
    inputs = 10
    outputs = 1
    neurons_per_layer = [5,5,5] # 3 hidden layers of 5 neurons
    neurons_per_layer.append(outputs)
    ann = NeuralNetwork.fromEmpty(sigmoid, inputs, neurons_per_layer)
