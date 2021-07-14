import numpy as np
from numpy import linalg as LA


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
    vertices = 2 * ((np.arange(2 ** d)[:, None] & (1 << np.arange(d))) > 0) - 1
    vals = np.arange(1, d + 1)
    vals = np.sqrt(vals)
    for i in range(num):
        if np.all(np.sign(point) == np.sign(vertices[i])):
            dist = LA.norm(point - (vertices[i] * vals))
            assert(dist < 0.1)
            return i
