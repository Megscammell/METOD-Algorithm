import numpy as np
from numpy import linalg as LA


def calc_minimizer_shekel(point, p, matrix_test, C, b):
    """
    Finds the nearest local minimizer for point using the Shekel
    function.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the function.
            A point used to evaluate the function.
    p : integer
        Number of local minima.
    matrix_test : 3-D array with shape (p, d, d).
    C : 2-D array with shape (d, p).
    B : 1-D array with shape (p, )

    Returns
    -------
    np.argmin(dist) : integer
                      Position of the local minimizer which produces the
                      smallest distance between point and all p local
                      minimizers.
    """
    dist = np.zeros((p))
    for i in range(p):
        dist[i] = LA.norm(point - C[:, i])
    assert(np.min(dist) < 0.1)
    return np.argmin(dist)
