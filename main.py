import ludopy
import numpy as np
from player.random_player import RandomPlayer
from player.ai_player import AiPlayer
from player.deep_ai_player import DeepAiPlayer
import matchmaking as mm
import artificial_intelligence.learning as lrn
import visualizer as viz

generations = 100
ai_boys = 25
rounds = 100

def main():
    print("Generations: " + str(generations))
    print("Max player count: " + str(ai_boys))
    print("Rounds: " +  str(rounds))
    # measure_ai_with_hand_selected_weights()
    # create_ais_matchmake_with_randoms_and_measure_success()
    # train_randomly_created_ais_over_generations()
    train_randomly_created_deep_ais_over_generations()
    pass

def train_randomly_created_deep_ais_over_generations():
    # first generation
    players = create_deep_ai_players_with_random_weights(ai_boys)
    winrates = np.empty(len(players))
    for i in range(0, generations):
        if i > 0:
            players = create_next_generation(players, i)
        print('----------- GENERATION ' + str(i) + ' -----------')
        for player in players:
            matchmake_with_randoms_and_measure_success(player)
        players.sort(key=lambda x: x.wincount, reverse=True)
        for i, player in enumerate(players):
            winrates[i] = player.wincount * 100 / rounds
        print("Win rates (%): " + str(winrates.tolist()))
        print("Overall win rate: " + str(np.average(winrates)) + "%")
        viz.add_generation_wincounts(winrates)
        viz.add_player_weights(players[0])

    winner = players[0]
    print('Best player of last generation win rate: ' + str(winner.wincount / rounds))
    print('Best player of last generation genome: ' + str(winner.ann.get_weights()))
    viz.show_plot()

def create_next_generation(players, generation):
    players = players[:2] # keep only the two winners
    add_ai_children(players, generation / generations)
    reset_win_counts(players)
    return players

def train_randomly_created_ais_over_generations():
    # first generation
    players = create_ais_matchmake_with_randoms_and_measure_success()
    wincounts = []
    for player in players:
        wincounts.append(player.wincount)
    print("wincounts " + str(wincounts))
    viz.add_generation_wincounts(wincounts)
    viz.add_player_weights(players[0])
    for i in range(0, generations):
        print('----------- GENERATION ' + str(i+1) + ' -----------')
        players = players[:2] # keep only the two winners
        add_ai_children(players, i / generations)
        reset_win_counts(players)
        for player in players:
            matchmake_with_randoms_and_measure_success(player)
        # print("sorting list")
        players.sort(key=lambda x: x.wincount, reverse=True)
        wincounts = []
        for player in players:
            wincounts.append(player.wincount)
        print("wincounts " + str(wincounts))
        viz.add_generation_wincounts(wincounts)
        viz.add_player_weights(players[0])

    winner = players[0]
    print('best player wins: ' + str(winner.wincount))
    print('best player genome: ' + str(winner.ann.get_weights()))
    viz.show_plot()

def add_ai_children(players, generation_fraction):
    mutation_amount = (1-generation_fraction) * 0.4
    # print("mutation amount: " + str(mutation_amount))
    dad = players[0]
    mom = players[1]
    while len(players) < ai_boys:
        child = dad.make_child(mom, mutation_amount)
        players.append(child)

def reset_win_counts(players):
    for player in players:
        player.wincount = 0

def create_ais_matchmake_with_randoms_and_measure_success():
    ai_players = create_ai_players_with_random_weights(ai_boys)
    # ai_players.append(hand_create_ai())
    for index, player in enumerate(ai_players):
        # print("player " + str(index))
        matchmake_with_randoms_and_measure_success(player)

    # print("sorting list")
    ai_players.sort(key=lambda x: x.wincount, reverse=True)
    # for index, player in enumerate(ai_players):
        # print("player " + str(index))
        # print("wincount " + str(player.wincount))
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

def create_deep_ai_players_with_random_weights(count):
    players_list = []
    for i in range(0, count):
        players_list.append(DeepAiPlayer.fromRandom())
    return players_list


if __name__ == "__main__":
    main()
