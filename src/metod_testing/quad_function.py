import numpy as np

def quad_function(point, p, store_x0, matrix_test):
    """Compute minimum of a number of quadratic functions at a given point.

    Keyword arguments:
    point -- is a (d,) array and the function is evaluated at point.
    p -- number of minima
    store_x0 -- function parameters
    matrix_test -- function parameters
    """
    store_func_values = np.zeros((p))
    for i in range(p):
        store_func_values[i] = np.transpose(point - store_x0[i]) @                                     matrix_test[i] @ (point-store_x0[i])
    value_of_minimum = np.min(store_func_values)

    return value_of_minimum


