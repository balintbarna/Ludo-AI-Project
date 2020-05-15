import matplotlib.pyplot as plt
import numpy as np
import math

def sigmoid(x):
    z = 1/(1 + np.exp(-x))
    return z

def relu(x):
    result = np.maximum(x, 0, x)
    return result

def leaky_relu(x):
    result = np.where(x > 0, x, x * 0.01)
    return result

if __name__ == "__main__":
    _inp = np.linspace(-10, 10, 100)
    print(_inp)
    _sig = sigmoid(_inp)
    print(_sig)
    plt.plot(_inp, _sig, label='sigmoid')
    _leaky_relu = leaky_relu(_inp)
    print(_leaky_relu)
    plt.plot(_inp, _leaky_relu, label='leaky')
    _relu = relu(_inp)
    print(_relu)
    plt.plot(_inp, _relu, label='relu')
    plt.xlabel("x")
    plt.legend()
    plt.show()
