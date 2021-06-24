import numpy as np
from numpy import linalg as LA


def calc_minimizer_styb(point, d):
    """
    Finds the nearest local minimizer for point using the Styblinski Tang
    function.

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
    vertices = 2 * ((np.arange(2 ** d)[:,None] & (1 << np.arange(d))) > 0) - 1
    for i in range(num):
        if np.all(np.sign(point) == np.sign(vertices[i])):
            temp_local_min = np.zeros((d))
            for j in range(d):
                if np.sign(vertices[i][j]) == -1:
                    temp_local_min[j] = np.sign(vertices[i][j]) * 2.903534
                else:
                    temp_local_min[j] = np.sign(vertices[i][j]) * 2.746803
            dist = LA.norm(point - temp_local_min )
            assert(dist < 0.1)
            return i