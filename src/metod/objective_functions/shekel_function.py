import numpy as np


def shekel_function(point, p, matrix_test, C, b):
    """
    Compute the Shekel function at a given point with given arguments.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the function.
    p : integer
        Number of local minima.
    matrix_test : 3-D array with shape (p, d, d).
    C : 2-D array with shape (d, p).
    B : 1-D array with shape (p, )

    Returns
    -------
    float(-total_sum) : float
                        Function value.
    """
    total_sum = 0
    for i in range(p):
        total_sum += 1 / ((point - C[:,i]).T @  matrix_test[i] @
                          (point - C[:,i]) + b[i])
    return(-total_sum)
