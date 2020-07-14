import numpy as np


def sog_function(point, p, sigma_sq, store_x0, matrix_test, store_c):
    """Compute Sum of Gaussians function at a given point with given arguments.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the function.
    p : integer
        Number of local minima.
    sigma_sq: float or integer
              Value of sigma squared.
    store_x0 : 2-D array with shape (p, d).
    matrix_test : 3-D array with shape (p, d, d).
    store_c : 1-D array with shape (p, ).

    Returns
    -------
    float(-function_val) : float
                           Function value.
    """
    function_val = 0
    for i in range(p):
        function_val += store_c[i] * np.exp((- 1 / (2 * sigma_sq)) *
                                            (np.transpose(point -
                                             store_x0[i])) @ matrix_test[i] @
                                            (point - store_x0[i]))
    return float(-function_val)
