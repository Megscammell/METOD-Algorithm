import math

import numpy as np
from numpy import linalg as LA


def calc_pos(point, p, store_x0, matrix_test):
    """Finding the position of the local minima in wich point belongs to.

    Keyword arguments:
    point -- is a (d,) array
    p -- number of minima
    store_x0 -- function parameters
    matrix_test -- function parameters
    """
    store_func_values = np.zeros((p))
    for i in range(p):
        store_func_values[i] = np.transpose(point - store_x0[i]) @ matrix_test[i] @ (point - store_x0[i])
    position_minimum = np.argmin(store_func_values)

    norm_with_minima = LA.norm(point - store_x0[position_minimum])
    return position_minimum, norm_with_minima