import math

import numpy as np
from numpy import linalg as LA


def calc_minima(point, p, exp_const, store_x0, matrix_test, store_c):
    dist = np.zeros((p))
    for i in range(p):
        dist[i] = LA.norm(point - store_x0[i])
    
    return np.argmin(dist), np.min(dist)
