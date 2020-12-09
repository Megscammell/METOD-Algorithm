import numpy as np


def styblinski_tang_function(point, d):
    """
    Styblinski-Tang function.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the function.
    d : integer
        Dimension.

    Returns
    -------
    function value : float

    """
    return ((1 / d) * (np.sum((point ** 4) - (16 * point ** 2) + 5 * point)))
