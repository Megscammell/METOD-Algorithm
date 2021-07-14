import numpy as np
from numpy import linalg as LA


def quantities(x, y, min_x, min_y, beta, p, store_x0, matrix_test):
    """
    For the minimum of several quadratic forms function, the expansion of c_1
    results in 3 quantities and the expansion of c_2 results in 2 quantities,
    where where c_1 = ||b||^2, c_2 = b^T (x - y) and
    b = beta * (g(y, *func_args) - g(x, *func_args)).

    Parameters
    ----------
    x : 1-D array of shape (d, )
        First point.
    y : 1-D array of shape (d, )
        Second point.
    min_x : integer
            Region of attraction index of x_tr.
    min_y : integer
            Region of attraction index of y_tr.
    beta : float or integer
           Small constant step size to compute the partner points.
    p : integer
        Number of local minima.
    store_x0 : 2-D arrays with shape (p, d), where d is the dimension.
    matrix_test : 3-D arrays with shape (p, d, d), where p and d are described
                  above.

    Returns
    -------
    quantities_array : 1-D array of shape (5)
                       Results of each of the 5 quantites.
    sum_quantities : float
                     Sum of all 5 quantities.
    """
    quantities_array = np.zeros((5))
    quantities_array[0] = (LA.norm(beta * (matrix_test[min_y] @ y -
                                   matrix_test[min_x] @ x))) ** 2
    quantities_array[1] = (LA.norm(beta * (matrix_test[min_x] @ store_x0
                           [min_x] - matrix_test[min_y] @ store_x0
                           [min_y]))) ** 2
    quantities_array[2] = (2 * (beta * (matrix_test[min_x] @
                                        store_x0[min_x] -
                                        matrix_test[min_y] @
                                        store_x0[min_y])).T @
                           (beta * (matrix_test[min_y] @ y -
                                    matrix_test[min_x] @ x)))
    quantities_array[3] = 2 * (beta * (matrix_test[min_x] @ store_x0[min_x]
                               - matrix_test[min_y] @ store_x0[min_y]).T @
                               (x - y))
    quantities_array[4] = 2 * ((beta * (matrix_test[min_y] @ y -
                                        matrix_test[min_x] @ x)).T @ (x - y))
    sum_quantities = np.sum(quantities_array)
    return quantities_array, sum_quantities
