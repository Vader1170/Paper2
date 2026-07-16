import numpy as np

def trapezoidal_weights(x):
    """
    Voronoi (trapezoidal) weights w_i as defined in the paper.
    These are identical to the composite trapezoidal rule weights.
    """
    n = len(x) - 1
    delta = np.diff(x)
    w = np.zeros(n+1)
    w[0] = delta[0] / 2.0
    w[-1] = delta[-1] / 2.0
    for i in range(1, n):
        w[i] = (delta[i-1] + delta[i]) / 2.0
    return w

def uniform_weights(x):
    """Naive equal weights summing to interval length."""
    n = len(x) - 1
    return np.ones(n+1) * (x[-1] - x[0]) / (n+1)
