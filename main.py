import ludopy
import numpy as np
from player.random_player import RandomPlayer
from player.ai_player import AiPlayer
import matchmaking as mm

def main():
    players = create_players()
    winner = mm.play(players)

    print('Winner is player #' + str(winner))
    try:
        print(str(players[winner].ann.get_weights()))
    except AttributeError:
        pass

def create_players():
    players = []
    num_rand = 3
    rand_player = RandomPlayer()
    for i in range(0,num_rand):
        players.append(rand_player)
    for i in range(0,4-num_rand):
        players.append(AiPlayer.fromRandom())
    return players


if __name__ == "__main__":
    main()
