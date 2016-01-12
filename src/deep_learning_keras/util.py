from numpy import mean, std, unique

def normalize(x):
    return (x - mean(x, 0)) / (std(x, 0) + 1e-9)

def one_hot(x):
    x0 = unique(x)
    return x0, (x == x0).astype(float)
