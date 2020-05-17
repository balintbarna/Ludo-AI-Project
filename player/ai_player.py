import numpy as np
from player.abstract_player import AbstractPlayer
import player.game_mechanics as gm
from artificial_intelligence.transfer_function import sigmoid
from artificial_intelligence.neural_network import neural_network

class AiPlayer(AbstractPlayer):
    def __init__(self, number_of_inputs, ann):
        self.ann = ann
        self.inputs = np.zeros(number_of_inputs)
    
    @classmethod
    def fromRandom(cls):
        number_of_inputs = 6
        layers = [1]
        ann = neural_network.fromEmpty(transfer_function = sigmoid, number_of_inputs = number_of_inputs, number_of_neurons_per_layer = layers)
        ann.randomize_weights(100)
        return cls(number_of_inputs, ann)
    
    @classmethod
    def fromWeights(cls, weights_list_list_list):
        number_of_inputs = 6
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
        can_enter = gm.can_enter_game(piece_pos, dice)
        can_finish = gm.is_goal(dest)
        tower = gm.can_build_tower(dest, player_pieces)
        safe = gm.is_safe(dest)
        kill = gm.can_kill_enemy(dest, enemy_pieces)
        progress = gm.potential_player_progress(player_pieces, piece_index, dice)

        self.inputs[0] = 1 if can_enter else 0
        self.inputs[1] = 1 if can_finish else 0
        self.inputs[2] = 1 if tower else 0
        self.inputs[3] = 1 if safe else 0
        self.inputs[4] = 1 if kill else 0
        self.inputs[5] = progress

        # print("piece_pos\tdest\tcan_enter\tcan_finish\ttower\tsafe\tkill\n"
        # +str(piece_pos)+'\t'+str(dest)+'\t'+str(can_enter)+'\t'+str(can_finish)+'\t'+str(tower)+'\t'+str(safe)+'\t'+str(kill))
        # print("inputs")
        # print(self.inputs)

