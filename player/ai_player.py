import numpy as np
from player.abstract_player import AbstractPlayer
import player.game_mechanics as gm
from artificial_intelligence.transfer_function import sigmoid
from artificial_intelligence.neural_network import neural_network

class AiPlayer(AbstractPlayer):
    def __init__(self):
        self.number_of_inputs = 10
        layers = [1]
        self.ann = neural_network(transfer_function = sigmoid, number_of_inputs = self.number_of_inputs, number_of_neurons_per_layer = layers)
        self.ann.randomize_weights(100)
        self.inputs = np.zeros(self.number_of_inputs)

    def select_piece_to_move(self, observation):
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner) = observation
        max_value = 0
        max_value_index = 0
        for i in range(0, len(move_pieces)):
            self.make_inputs(observation, move_pieces[i])
            value = self.ann.calculate_outputs(self.inputs)[0]
            if(value > max_value):
                max_value = value
                max_value_index = i
        piece_to_move = move_pieces[max_value_index]
        return piece_to_move

    def make_inputs(self, observation, piece_index):
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner) = observation
        piece_pos = player_pieces[piece_index]
        dest = gm.piece_destination(piece_pos, dice)
        at_start = gm.is_start(piece_pos)
        can_enter = gm.can_enter_game(piece_pos, dice)
        can_finish = gm.is_goal(dest)

