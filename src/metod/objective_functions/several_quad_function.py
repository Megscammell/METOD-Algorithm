import numpy as np


def several_quad_function(point, p, store_x0, matrix_test):
    """
    Minimum of several Quadratic forms function.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the function.
    p : integer
        Number of local minima.
    store_x0 : 2-D array with shape (p, d).
    matrix_test : 3-D array with shape (p, d, d).

    Returns
    -------
    value_of_minimum : float
                       Function value.
    """
    store_func_values = np.zeros((p))
    for i in range(p):
        store_func_values[i] = 0.5 * (np.transpose(point - store_x0[i]) @
                                      matrix_test[i] @ (point-store_x0[i]))
    value_of_minimum = np.min(store_func_values)

    return value_of_minimum
