import numpy as np


def styblinski_tang_function(point):
    """
    Styblinski-Tang function.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the function.

    Returns
    -------
    function value : float

    """
    return ((1 / 2) * (np.sum((point ** 4) - (16 * point ** 2) + 5 * point)))


def styblinski_tang_gradient(point):
    """
    Styblinski-Tang gradient.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the function.

    Returns
    -------
    gradient : 1-D array with shape (d, )

    """
    grad = (1 / 2) * (4 * point ** 3 - 32 * point + 5)
    return grad
