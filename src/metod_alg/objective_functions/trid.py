import numpy as np


def trid_func(x,d):
    """
    Compute Trid function at a given point with given arguments.

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
    s_1 =  np.sum((x - 1)**2)
    s_2 = np.sum(x[1:] * x[:-1])
    return s_1 - s_2


def trid_grad(x,d):
    """
    Compute Trid gradient at a given point with given arguments.

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
        if i == 0:
            grad[i] = 2 * (x[i] - 1) - x[i+1]
        elif i > 0 and i < (d - 1):
            grad[i] = 2 * (x[i] - 1) - x[i-1] - x[i+1]
        elif i == (d-1):
            grad[i] = 2 * (x[i] - 1) - x[i-1]
    return grad