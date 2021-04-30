import numpy as np
from numpy import linalg as LA


def calc_minimizer_hartmann6(point):
    """
    Finds the nearest local minimizer for point.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the function.

    Returns
    -------
    pos : integer or None
          Local minimizer index if dist < 0.1. Otherwise pos = None.
    """
    local_minimizers = np.array([[0.2017 , 0.15   , 0.47683, 0.27534, 0.31166, 0.6573 ],
                                 [0.40466, 0.88244, 0.84363, 0.57399, 0.1392 , 0.0385 ]])
    for i in range(2):
        dist = LA.norm(point - (local_minimizers[i]))
        if dist < 0.1:
            return i
    return None
