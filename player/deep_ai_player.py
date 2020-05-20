import numpy as np
from player.abstract_player import AbstractPlayer
import player.game_mechanics as gm
from artificial_intelligence.transfer_function import sigmoid
from artificial_intelligence.neural_network import neural_network

class AiPlayer(AbstractPlayer):
    def __init__(self, number_of_inputs, ann):
        super().__init__()
        self.ann = ann
        self.inputs = np.zeros(number_of_inputs)
    
    @classmethod
    def fromRandom(cls):
        number_of_inputs = 23
        layers = [23, 30, 30, 23, 1]
        ann = neural_network.fromEmpty(transfer_function = sigmoid, number_of_inputs = number_of_inputs, number_of_neurons_per_layer = layers)
        ann.randomize_weights(100)
        return cls(number_of_inputs, ann)
    
    @classmethod
    def fromWeights(cls, weights_list_list_list):
        number_of_inputs = 23
        ann = neural_network.fromWeights(sigmoid, weights_list_list_list)
        return cls(number_of_inputs, ann)

    def select_piece_to_move(self, observation):
        (_, moveable_pieces, _, _, _, _) = observation
        max_value = 0
        max_value_index = 0
        for i in range(0, len(moveable_pieces)):
            self.make_inputs(observation, moveable_pieces[i])
            value = self.ann.calculate_outputs(self.inputs)[0]
            if(value > max_value):
                max_value = value
                max_value_index = i
        piece_to_move = moveable_pieces[max_value_index]
        return piece_to_move

    def make_inputs(self, observation, piece_index):
        (dice, _, player_pieces, enemy_pieces, _, _) = observation
        piece_pos = player_pieces[piece_index]
        dest = gm.piece_destination(piece_pos, dice)


        self.inputs[0] = piece_pos
        self.inputs[1] = dice
        self.inputs[2] = dest
        self.inputs[3] = player_pieces[0]
        self.inputs[4] = player_pieces[1]
        self.inputs[5] = player_pieces[2]
        self.inputs[6] = player_pieces[3]
        self.inputs[7] = enemy_pieces[0][0]
        self.inputs[8] = enemy_pieces[0][1]
        self.inputs[9] = enemy_pieces[0][2]
        self.inputs[10] = enemy_pieces[0][3]
        self.inputs[11] = enemy_pieces[1][0]
        self.inputs[12] = enemy_pieces[1][1]
        self.inputs[13] = enemy_pieces[1][2]
        self.inputs[14] = enemy_pieces[1][3]
        self.inputs[15] = enemy_pieces[2][0]
        self.inputs[16] = enemy_pieces[2][1]
        self.inputs[17] = enemy_pieces[2][2]
        self.inputs[18] = enemy_pieces[2][3]
        self.inputs[19] = enemy_pieces[3][0]
        self.inputs[20] = enemy_pieces[3][1]
        self.inputs[21] = enemy_pieces[3][2]
        self.inputs[22] = enemy_pieces[3][3]


        # print("piece_pos\tdest\tcan_enter\tcan_finish\ttower\tsafe\tkill\n"
        # +str(piece_pos)+'\t'+str(dest)+'\t'+str(can_enter)+'\t'+str(can_finish)+'\t'+str(tower)+'\t'+str(safe)+'\t'+str(kill))
        # print("inputs")
        # print(self.inputs)

