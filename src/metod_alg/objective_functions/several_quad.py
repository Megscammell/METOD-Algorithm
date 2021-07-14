import numpy as np


def several_quad_function(point, p, store_x0, matrix_test):
    """
    Minimum of several quadratic forms function.

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
    d = point.shape[0]
    store_func_values = 0.5 * (np.transpose((point - store_x0).reshape(p, d, 1), (0, 2, 1)) @
                               matrix_test @ (point - store_x0).reshape(p, d, 1))
    value_of_minimum = np.min(store_func_values)

    return value_of_minimum


def several_quad_gradient(point, p, store_x0, matrix_test):
    """Minimum of several quadratic forms gradient.

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
    d = point.shape[0]
    store_func_values = (np.transpose((point - store_x0).reshape(p, d, 1), (0, 2, 1)) @
                         matrix_test @ (point - store_x0).reshape(p, d, 1))
    position_minimum = np.argmin(store_func_values)
    gradient = (matrix_test[position_minimum] @
                (point - store_x0[position_minimum]))
    return gradient
