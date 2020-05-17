import numpy as np
from player.abstract_player import *

class RandomPlayer(AbstractPlayer):
    def __init__(self):
        super().__init__()

    def select_piece_to_move(self, observation):
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner) = observation
        piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
        return piece_to_move
