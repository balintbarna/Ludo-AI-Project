import numpy as np
import matplotlib.pyplot as plt

player_weights = [] # one element is all the weights of a player in 1 generation
weights_labels = ["bias", "can_enter", "can_finish", "can_tower", "safe_zone", "can_kill", "progress_score"]
wincounts_list = [] # one element is the list of wincounts of all players in 1 generation

def add_player_weights(player):
    player_weights.append(player.ann.get_weights()[0][0])

def add_generation_wincounts(wincounts):
    wincounts_list.append(wincounts)

def show_plot():
    arr = np.array(player_weights)
    arr = arr.T
    assert len(arr) == len(weights_labels)
    for i in range(0, len(arr)):
        y = arr[i]
        x = np.arange(0, len(y))
        plt.plot(x, y, label=weights_labels[i])
    plt.xlabel("generations")
    plt.ylabel("weight values")
    plt.legend()
    plt.show()
    plt.clf()
    
    arr = np.array(wincounts_list)
    arr = arr.T
    for i in range(0, len(arr)):
        y = arr[i]
        x = np.arange(0, len(y))
        plt.plot(x, y)
    plt.xlabel("generations")
    plt.ylabel("win counts from top player to bottom")
    plt.show()


if __name__ == "__main__":
    for i in range(0, 10):
        player_weights.append([0,1,2,3,4,5])
    show_plot()