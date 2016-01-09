from numpy import mean, std, unique
from numpy.random import seed, permutation

def normalize(x):
    return (x - mean(x, 0)) / (std(x, 0) + 1e-9)

def one_hot(x):
    x0 = unique(x)
    return x0, (x == x0).astype(float)

def shuffle(*x):
    seed(0)
    perm = permutation(x[0].shape[0])
    return tuple(x0[perm] for x0 in x)
