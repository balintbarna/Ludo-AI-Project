import ludopy
import numpy as np
from player.random_player import RandomPlayer
from player.ai_player import AiPlayer

def play(players):
    # print("\n\nNew game\n----------------------\n")
    game = ludopy.Game()
    there_is_a_winner = False
    player_i = -1

    while not there_is_a_winner:
        observation, player_i = game.get_observation()
        (dice, movable_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner) = observation

        player = players[player_i]

        if len(movable_pieces) > 0:
            # print('player: #' + str(player_i) + '\t' + str(dice) + '\t' + str(movable_pieces.tolist()) + '\t\t' + str(player_pieces.tolist()) + '\t\t' + str(enemy_pieces.tolist()))
            piece_to_move = player.select_piece_to_move(observation)
        else:
            # print('-')
            piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = game.answer_observation(piece_to_move)

    # print("Saving history to numpy file")
    # game.save_hist(f"game_history.npy")
    # print("Saving game video")
    # game.save_hist_video(f"game_video.mp4")
    return player_i

def play_against_randoms(special_player):
    rand_player = RandomPlayer()
    players = [special_player, rand_player, rand_player, rand_player]
    special_wins = play(players) == 0
    return special_wins

def play_many_against_randoms(special_player, rounds = 10):
    wincount = 0
    for i in range(0, rounds):
        win = play_against_randoms(special_player)
        if win:
            wincount += 1
    return wincount
    