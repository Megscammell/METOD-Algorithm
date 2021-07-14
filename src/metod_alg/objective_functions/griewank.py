import numpy as np


def griewank_func(x, d):
    """
    Compute Griewank function at a given point with given arguments.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the function.
    d : integer
        Dimension

    Returns
    -------
    function value : float
    """
    s_a = 0
    s_m = 1
    for i in range(d):
        s_a += x[i] ** 2
        s_m *= np.cos(x[i] / np.sqrt(i + 1))
    return (1/4000) * s_a - s_m + 1


def griewank_grad(x, d):
    """
    Compute Griewank gradient at a given point with given arguments.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the function.
    d : integer
        Dimension

    Returns
    -------
    grad : 1-D array with shape (d,)
    """
    grad = np.zeros((d))
    for i in range(d):
        s_m = 1
        for j in range(d):
            if j != i:
                s_m *= np.cos(x[j] / np.sqrt(j + 1))
        grad[i] = ((x[i] / 2000) + (1/np.sqrt(i + 1)) *
                   np.sin(x[i] / np.sqrt(i + 1)) * s_m)
    return grad
