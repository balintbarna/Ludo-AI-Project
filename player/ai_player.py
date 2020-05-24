import numpy as np
from artificial_intelligence.transfer_function import sigmoid
from artificial_intelligence.neural_network import NeuralNetwork
import artificial_intelligence.learning as lrn
from player.abstract_player import AbstractPlayer
import player.game_mechanics as gm

class AiPlayer(AbstractPlayer):
    def __init__(self, ann: NeuralNetwork):
        super().__init__()
        self.ann = ann
    
    @classmethod
    def fromRandom(cls):
        number_of_inputs = 6
        layers = [1]
        ann = NeuralNetwork.fromRandom(transfer_function = sigmoid, number_of_inputs = number_of_inputs, number_of_neurons_per_layer = layers)
        return cls(ann)
    
    @classmethod
    def fromWeights(cls, weights_list_list_list):
        ann = NeuralNetwork.fromWeights(sigmoid, weights_list_list_list)
        return cls(ann)

    def select_piece_to_move(self, observation):
        (_, moveable_pieces, _, _, _, _) = observation
        max_value = 0
        max_value_index = 0
        for i in range(0, len(moveable_pieces)):
            self.make_inputs(observation, moveable_pieces[i])
            value = self.ann.calculate_outputs()[0]
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

        inputs = self.ann.get_input_array()

        inputs[0] = 1 if can_enter else 0
        inputs[1] = 1 if can_finish else 0
        inputs[2] = 1 if tower else 0
        inputs[3] = 1 if safe else 0
        inputs[4] = 1 if kill else 0
        inputs[5] = progress
    
    def get_weights(self):
        return self.ann.get_weights()
        
    def introduce_mutation(self, max_mutation_amount):
        self.ann.introduce_mutation(max_mutation_amount)
    
    def normalize(self, max_value):
        self.ann.normalize(max_value)
    
    def get_max_weight(self):
        return self.ann.get_max_weight()
    
    def mix_weights(self, other):
        child_network = self.ann.mix_weights(other.ann)
        return AiPlayer(child_network)

    def make_child(self, other, mutation_amount):
        child = self.mix_weights(other)
        child.introduce_mutation(mutation_amount)
        child.normalize(child.get_max_weight())
        return child
