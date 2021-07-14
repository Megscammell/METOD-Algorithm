import numpy as np


def zakharov_func(x, d):
    """
    Compute Zakharov function at a given point with given arguments.

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
    s_1 = np.sum(x ** 2)
    s_2 = (0.5 * np.arange(1, d + 1)).T @  x
    return s_1 + s_2 ** 2 + s_2 ** 4


def zakharov_grad(x, d):
    """
    Compute Zakharov gradient at a given point with given arguments.

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
    s_2 = (0.5 * np.arange(1, d + 1)).T @  x
    grad = (2 * x + np.arange(1, d + 1) *
            s_2 + 2 * np.arange(1, d + 1) * (s_2 ** 3))
    return grad
