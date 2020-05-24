import numpy as np

def create_child(dad, mom):
    if islist(dad) and islist(mom):
        assert_length(dad, mom)
        count = len(dad)
        child = np.empty_like(dad)
        for index in range(0, count):
            child[i] = create_child(dad[index], mom[index])
        assert_length(dad, child)
        return child
    else:
        child = pick_random(dad, mom)
        return child

def introduce_mutation(weights, max_mutation_amount = 1):
    for index, elem in enumerate(weights):
        if islist(elem):
            introduce_mutation(elem)
        else:
            mutation = np.random.rand() * 2 - 1
            mutation *= max_mutation_amount
            weights[index] += mutation

def pick_random(one, two):
    pick = np.random.randint(2)
    result = one if pick == 0 else two
    return result

def abs_max(arr):
    amax = np.max(arr)
    amin = np.min(arr)
    return np.where(-amin > amax, -amin, amax)

def islist(obj):
    return isinstance(obj, list) or isinstance(obj, np.ndarray)

def assert_length(one, two):
    assert len(one) == len(two)


if __name__ == "__main__":
    weights = [[[3, 2, 1],[1,2,3]],[[0,1,2], [-1, -2, -3]]]
    introduce_mutation(weights)
    print(weights)
    weights_dad = [[[3, 2, 1],[1,2,3]],[[0,1,2], [-1, -2, -3]]]
    weights_mom = [[[8,9,10],[4,5,6]],[[-10,-1,-2], [9,9,9]]]
    child = create_child(weights_dad, weights_mom)
    print(child)
