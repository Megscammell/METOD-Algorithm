import numpy as np


def single_quad_function(point, x0, A, rotation):
    """
    Quadratic function.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the gradient.
    x0 : 1-D array with shape (d, ).
    A : 2-D array with shape (d, d).
        Diagonal matrix.
    rotation : 2-D array with shape (d, d).
               Rotation matrix.

    Returns
    -------
    function value : float

    """

    return 0.5 * (point - x0).T @ rotation.T @ A @ rotation @ (point - x0)
