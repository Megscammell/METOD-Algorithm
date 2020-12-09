import numpy as np


def qing_function(point, d):
    """
    Qing function.

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
    return np.sum((point ** 2 - np.arange(1, d + 1)) ** 2)
