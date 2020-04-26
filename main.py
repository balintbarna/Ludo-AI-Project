import ludopy
import numpy as np
from player.random_player import RandomPlayer as Player

def main():
    game = ludopy.Game()
    there_is_a_winner = False
    players = []
    for i in range(0,4):
        players.append(Player())

    while not there_is_a_winner:
        observation, player_i = game.get_observation()
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner) = observation

        player = players[player_i]

        if len(move_pieces) > 0:
            piece_to_move = player.select_piece_to_move(observation)
        else:
            piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = game.answer_observation(piece_to_move)

    print("Saving history to numpy file")
    game.save_hist(f"game_history.npy")
    print("Saving game video")
    game.save_hist_video(f"game_video.mp4")

if __name__ == "__main__":
    main()
