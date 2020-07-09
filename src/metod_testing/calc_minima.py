import numpy as np
from numpy import linalg as LA


def calc_minima(point, p, exp_const, store_x0, matrix_test, store_c):
    """ Finds the nearest local minima for x_n^(K_n) for the sum of Gaussians
     function

    Keyword arguments:
    point -- is a (d,) array
    p, exp_const, store_x0, matrix_test, store_c -- function arguments for the
     sum of Gaussians functions.
    """
    dist = np.zeros((p))
    for i in range(p):
        dist[i] = LA.norm(point - store_x0[i])
    return np.argmin(dist), np.min(dist)
