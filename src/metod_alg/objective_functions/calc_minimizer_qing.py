import numpy as np
from numpy import linalg as LA
from itertools import product


def calc_minimizer_qing(point, d):
    """
    Finds the nearest local minimizer for the Qing function.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the function.
            A point used to evaluate the function.
    d : integer
        Dimension.

    Returns
    -------
    i : integer
        Local minimizer index.
    """
    num = 2 ** d
    vertices = np.array(list(product([-1,1], repeat=d)))
    vals = np.arange(1, d + 1)
    vals = np.sqrt(vals)
    for i in range(num):
        if np.all(np.sign(point) == np.sign(vertices[i])):
            dist = LA.norm(point - (vertices[i] * vals))
            assert(dist < 0.1)
            return i
