import ludopy
import numpy as np
from player.random_player import RandomPlayer
from player.ai_player import AiPlayer

def main():
    print("\n\nNew game\n----------------------\n")
    game = ludopy.Game()
    there_is_a_winner = False
    players = create_players()

    while not there_is_a_winner:
        observation, player_i = game.get_observation()
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner) = observation

        player = players[player_i]

        if len(move_pieces) > 0:
            print('player: #' + str(player_i) + '\t' + str(observation))
            piece_to_move = player.select_piece_to_move(observation)
        else:
            print('-')
            piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = game.answer_observation(piece_to_move)

        if(there_is_a_winner):
            print('Winner is player #' + str(player_i))
            try:
                print(str(players[player_i].ann.get_weights()))
            except AttributeError:
                pass

    print("Saving history to numpy file")
    game.save_hist(f"game_history.npy")
    print("Saving game video")
    game.save_hist_video(f"game_video.mp4")

def create_players():
    players = []
    num_rand = 1
    for i in range(0,num_rand):
        players.append(RandomPlayer())
    for i in range(0,4-num_rand):
        players.append(AiPlayer())
    return players

if __name__ == "__main__":
    main()
