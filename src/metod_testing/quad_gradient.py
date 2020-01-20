import numpy as np

def quad_gradient(point, p, store_x0, matrix_test):
    """Compute gradient of point.

    Keyword arguments:
    point -- is a (d,) array and the function is evaluated at point.
    p -- number of minima
    store_x0 -- function parameters
    matrix_test -- function parameters
    """
    store_func_values = np.zeros((p))
    for i in range(p):
        store_func_values[i] = np.transpose(point - store_x0[i]) @ matrix_test[i] @ (point-store_x0[i])
    position_minimum = np.argmin(store_func_values)
    gradient = 2 * (matrix_test[position_minimum] @ (point - store_x0                     [position_minimum])) 
    return gradient


