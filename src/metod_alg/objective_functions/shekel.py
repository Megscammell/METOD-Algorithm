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
    float(-total_sum * 0.5) : float
                              Function value.
    """
    total_sum = 0
    for i in range(p):
        total_sum += 1 / ((point - C[:, i]).T @  matrix_test[i] @
                          (point - C[:, i]) + b[i])
    return(-total_sum * 0.5)


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
        num = matrix_test[i] @ (point - C[:, i])
        denom = ((point - C[:, i]).T @  matrix_test[i] @ (point - C[:, i]) +
                 b[i]) ** (2)
        grad += (num / denom)
    return grad
