import numpy as np


def quad_function(point, p, store_x0, matrix_test):
    """Compute minimum of several quadratic forms function.

    Keyword arguments:
    point -- is a (d,) array and the function is evaluated at point.
    p, store_x0, matrix_test -- function parameters
    """
    store_func_values = np.zeros((p))
    for i in range(p):
        store_func_values[i] = 0.5 * (np.transpose(point - store_x0[i]) @
                                      matrix_test[i] @ (point-store_x0[i]))
    value_of_minimum = np.min(store_func_values)

    return value_of_minimum
