import numpy as np
from player.abstract_player import *
from neural_net.transfer_function import sigmoid
from neural_net.neural_network import neural_network

class AiPlayer(AbstractPlayer):
    def __init__(self):
        self.number_of_inputs = 10
        self.ann = neural_network(transfer_function = sigmoid, number_of_inputs = self.number_of_inputs, number_of_outputs = 1, number_of_hidden_layers=0, number_of_hidden_layer_neurons=0)

    def select_piece_to_move(self, observation):
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner) = observation
        inputs = np.zeros(self.number_of_inputs)
        max_value = 0
        max_value_index = 0
        for i in range(0, len(move_pieces)):
            value = self.ann.calculate_outputs(inputs)[0]
            print('value'+str(value))
            if(value > max_value):
                max_value = value
                max_value_index = i
        piece_to_move = move_pieces[max_value_index]
        return piece_to_move
