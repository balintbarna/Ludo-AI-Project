import ludopy
import numpy as np
from player.random_player import RandomPlayer
from player.ai_player import AiPlayer
import matchmaking as mm
import artificial_intelligence.learning as lrn

generations = 10
children = 8
rounds = 100

def main():
    # measure_ai_with_hand_selected_weights()
    # create_ais_matchmake_with_randoms_and_measure_success()
    train_randomly_created_ais_over_generations()
    pass

def train_randomly_created_ais_over_generations():
    # first generation
    players = create_ais_matchmake_with_randoms_and_measure_success()
    for i in range(0, generations):
        print('----------- GENERATION ' + str(i+1) + ' -----------')
        players = players[:2] # keep only the two winners
        add_children(players, children)
        reset_win_counts(players)
        for index, player in enumerate(players):
            matchmake_with_randoms_and_measure_success(player)
        print("sorting list")
        players.sort(key=lambda x: x.wincount, reverse=True)
        for index, player in enumerate(players):
            print("wincount " + str(player.wincount))
    winner = players[0]
    print('best player wins: ' + str(winner.wincount))
    print('best player genome: ' + str(winner.ann.get_weights()))

def add_children(players, children_count):
    for _ in range(0, children_count):
        child = lrn.create_child(players[0].ann.get_weights(), players[1].ann.get_weights())
        lrn.introduce_mutation(child, 0.4)
        child_player = AiPlayer.fromWeights(child)
        players.append(child_player)

def reset_win_counts(players):
    for player in players:
        player.wincount = 0

def create_ais_matchmake_with_randoms_and_measure_success():
    ai_players = create_ai_players_with_random_weights(10)
    ai_players.append(hand_create_ai)
    for index, player in enumerate(ai_players):
        # print("player " + str(index))
        matchmake_with_randoms_and_measure_success(player)

    print("sorting list")
    ai_players.sort(key=lambda x: x.wincount, reverse=True)
    for index, player in enumerate(ai_players):
        # print("player " + str(index))
        print("wincount " + str(player.wincount))
    return ai_players

def matchmake_with_randoms_and_measure_success(player):
    players = mm.matchmake_with_randoms([player])
    mm.play_rounds(players, rounds)
    success_percentage = 100 * player.wincount / rounds
    # print("success rate " + str(success_percentage) + "%")

def measure_ai_with_hand_selected_weights():
    ai = hand_create_ai()
    wincount = mm.play_many_against_randoms(ai, rounds)
    print("wins: " + str(wincount/rounds * 100) + "%")
    print(str(ai.ann.get_weights()))

def hand_create_ai():
    #            bias   enter   finish  tower   safe    kill    progress score
    weights = [[[0,     0.9,    1,      0.2,    0.2,    0.4,    0.3]]]
    ai = AiPlayer.fromWeights(weights)
    return ai

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

def create_ai_players_with_random_weights(count):
    players_list = []
    for i in range(0, count):
        players_list.append(AiPlayer.fromRandom())
    return players_list


if __name__ == "__main__":
    main()
