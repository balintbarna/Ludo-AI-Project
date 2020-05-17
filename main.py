import ludopy
import numpy as np
from player.random_player import RandomPlayer
from player.ai_player import AiPlayer
import matchmaking as mm

def main():
    measure_ai_with_hand_selected_weights()

def measure_ai_with_hand_selected_weights():
    rounds = 1000
    #            bias   enter   finish  tower   safe    kill    progress score
    weights = [[[0,     0.9,    1,      0.2,    0.2,    0.4,    0.3]]]
    ai = AiPlayer.fromWeights(weights)
    wincount = mm.play_many_against_randoms(ai, rounds)
    print("wins: " + str(wincount/rounds * 100) + "%")
    print(str(ai.ann.get_weights()))

def measure_ai_success_rate():
    ai = AiPlayer.fromRandom()
    wincount = mm.play_many_against_randoms(ai, 100)
    print(wincount)
    print(str(ai.ann.get_weights()))

def ai_against_randos():
    ai = AiPlayer.fromRandom()
    win = mm.play_against_randoms(ai)
    print(win)
    print(str(ai.ann.get_weights()))

def simple_game():
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
