import numpy as np


def shekel_gradient(point, p, matrix_test, C, b):
    """
    Compute Shekel gradient at a given point with given arguments.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the gradient.
    p : integer
        Number of local minima.
    matrix_test : 3-D array with shape (p, d, d).
    C : 2-D array with shape (d, p).
    b : 1-D array with shape (p, ).

    Returns
    -------
    grad : 1-D array with shape (d, )
           Gradient at point.
    """
    grad = 0
    for i in range(p):
        num = 2 * matrix_test[i] @ (point - C[:, i]) 
        denom = ((point - C[:, i]).T @  matrix_test[i] @ (point - C[:, i]) +
                 b[i]) ** (2)
        grad += (num / denom)
    return grad
