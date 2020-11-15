import numpy as np
from numpy import linalg as LA


def calc_pos(point, p, store_x0, matrix_test):
    """
    Finding the position of the local minimizer which point is closest
    to, using the minimum of several Quadratic forms function.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the function.
    p : integer
        Number of local minima.
    store_x0 : 2-D arrays with shape (p, d).
    matrix_test : 3-D arrays with shape (p, d, d).

    Returns
    -------
    position_minimum : integer
                       Position of the local minimizer which produces the
                       smallest distance between point and all p local
                       minimizers.
    norm_with_minimizer : float
                          The smallest distance between point and all p local
                          minimizers
    """
    store_func_values = np.zeros((p))
    for i in range(p):
        store_func_values[i] = 0.5 * (np.transpose(point - store_x0[i]) @
                                      matrix_test[i] @ (point - store_x0[i]))
    position_minimum = np.argmin(store_func_values)
    norm_with_minimizer = LA.norm(point - store_x0[position_minimum])
    return position_minimum, norm_with_minimizer
