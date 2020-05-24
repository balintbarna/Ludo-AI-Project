import numpy as np
from player.abstract_player import AbstractPlayer
import player.game_mechanics as gm
from artificial_intelligence.transfer_function import sigmoid
from artificial_intelligence.neural_network import NeuralNetwork

class DeepAiPlayer(AbstractPlayer):
    def __init__(self, ann: NeuralNetwork):
        super().__init__()
        self.ann = ann
    
    @classmethod
    def fromRandom(cls):
        number_of_inputs = 17
        layers = [17, 30, 30, 17, 4]
        ann = NeuralNetwork.fromRandom(transfer_function = sigmoid, number_of_inputs = number_of_inputs, number_of_neurons_per_layer = layers)
        return cls(ann)
    
    @classmethod
    def fromWeights(cls, weights_list_list_list):
        ann = NeuralNetwork.fromWeights(sigmoid, weights_list_list_list)
        return cls(ann)
    
    def getWeights(self):
        return self.ann.get_weights()

    def select_piece_to_move(self, observation):
        (_, moveable_pieces, _, _, _, _) = observation
        self.make_inputs(observation)
        outputs = self.ann.calculate_outputs()
        max_value = -1
        max_value_index = 0
        for i in range(0, len(moveable_pieces)):
            value = outputs[moveable_pieces[i]]
            if value > max_value:
                max_value = value
                max_value_index = i
        piece_to_move = moveable_pieces[max_value_index]
        return piece_to_move

    def make_inputs(self, observation):
        (dice, _, player_pieces, enemy_pieces, _, _) = observation

        inputs = self.ann.get_input_array()

        inputs[0] = dice / gm.dice_max
        inputs[1] = player_pieces[0] / gm.pos_max
        inputs[2] = player_pieces[1] / gm.pos_max
        inputs[3] = player_pieces[2] / gm.pos_max
        inputs[4] = player_pieces[3] / gm.pos_max
        inputs[5] = enemy_pieces[0][0] / gm.pos_max
        inputs[6] = enemy_pieces[0][1] / gm.pos_max
        inputs[7] = enemy_pieces[0][2] / gm.pos_max
        inputs[8] = enemy_pieces[0][3] / gm.pos_max
        inputs[9] = enemy_pieces[1][0] / gm.pos_max
        inputs[10] = enemy_pieces[1][1] / gm.pos_max
        inputs[11] = enemy_pieces[1][2] / gm.pos_max
        inputs[12] = enemy_pieces[1][3] / gm.pos_max
        inputs[13] = enemy_pieces[2][0] / gm.pos_max
        inputs[14] = enemy_pieces[2][1] / gm.pos_max
        inputs[15] = enemy_pieces[2][2] / gm.pos_max
        inputs[16] = enemy_pieces[2][3] / gm.pos_max
