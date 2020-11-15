import numpy as np


def quad_gradient(point, p, store_x0, matrix_test):
    """Minimum of several Quadratic forms gradient.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the gradient.
    p : integer
        Number of local minima.
    store_x0 : 2-D array with shape (p, d).
    matrix_test : 3-D array with shape (p, d, d).

    Returns
    -------
    gradient : 1-D array with shape (d, )
               Gradient at point.
    """
    store_func_values = np.zeros((p))
    for i in range(p):
        store_func_values[i] = (np.transpose(point - store_x0[i]) @
                                matrix_test[i] @ (point-store_x0[i]))
    position_minimum = np.argmin(store_func_values)
    gradient = (matrix_test[position_minimum] @
                (point - store_x0[position_minimum]))
    return gradient
