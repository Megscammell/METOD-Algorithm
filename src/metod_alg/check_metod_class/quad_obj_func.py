import numpy as np


def quad_function(point, p, store_x0, matrix_test):
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
    d = point.shape[0]
    store_func_values = (np.transpose((point - store_x0).reshape(p, d, 1), (0, 2, 1)) @
                         matrix_test @ (point - store_x0).reshape(p, d, 1))
    value_of_minimum = np.min(store_func_values)
    return value_of_minimum


def quad_gradient(point, p, store_x0, matrix_test):
    """
    Minimum of several Quadratic forms gradient.

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
    gradient = 2 * (matrix_test[position_minimum] @
                    (point - store_x0[position_minimum]))
    return gradient


def calc_minimizer_quad(point, p, store_x0, matrix_test):
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
    """
    d = point.shape[0]
    store_func_values = (np.transpose((point - store_x0).reshape(p, d, 1), (0, 2, 1)) @
                         matrix_test @ (point - store_x0).reshape(p, d, 1))
    position_minimum = np.argmin(store_func_values)
    norm_with_minimizer = np.linalg.norm(point - store_x0[position_minimum])
    assert(norm_with_minimizer < 0.2)
    return position_minimum
