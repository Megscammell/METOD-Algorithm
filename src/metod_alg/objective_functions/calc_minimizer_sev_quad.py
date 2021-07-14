import numpy as np
from numpy import linalg as LA


def calc_minimizer_sev_quad(point, p, store_x0, matrix_test):
    """
    Finding the position of the local minimizer which point is closest
    to using the minimum of several quadratic forms function.

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
    """
    d = point.shape[0]
    store_func_values = (np.transpose((point - store_x0).reshape(p, d, 1), (0, 2, 1)) @
                         matrix_test @ (point - store_x0).reshape(p, d, 1))
    position_minimum = np.argmin(store_func_values)
    norm_with_minimizer = LA.norm(point - store_x0[position_minimum])
    assert(norm_with_minimizer < 0.2)
    return position_minimum
