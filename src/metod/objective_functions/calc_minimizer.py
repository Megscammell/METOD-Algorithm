import numpy as np
from numpy import linalg as LA


def calc_minimizer(point, p, sigma_sq, store_x0, matrix_test, store_c):
    """
    Finds the nearest local minimizer for point using the Sum of Gaussians
    function.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the function.
    p : integer
        Number of local minima.
    sigma_sq: float or integer
              Value of sigma squared.
    store_x0 : 2-D arrays with shape (p, d).
    matrix_test : 3-D arrays with shape (p, d, d).
    store_c : 3-D arrays with shape (p, ).

    Returns
    -------
    np.argmin(dist) : integer
                      Position of the local minimizer which produces the
                      smallest distance between point and all p local
                      minimizers.
    np.min(dist) : float
                   The smallest distance between point and all p local
                   minimizers
    """
    dist = np.zeros((p))
    for i in range(p):
        dist[i] = LA.norm(point - store_x0[i])
    return np.argmin(dist), np.min(dist)
