import numpy as np


def styblinski_tang_gradient(point, d):
    """
    Styblinski-Tang gradient.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the function.
    d : integer
        Dimension.

    Returns
    -------
    gradient : 1-D array with shape (d, )

    """
    grad = (1 / d) * (4 * point ** 3 - 32 * point + 5)
    return grad
