import numpy as np


def hartmann6_func_params():
    """
    Generate parameters used for Hartmann function with d=6.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the function.
    d : integer
        Dimension

    Returns
    -------
    a : 2-D array with shape (4, 6)
    c : 1-D array with shape (4,)
    p : 2-D array with shape (4, 6)
    """
    a = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    c = np.array([1, 1.2, 3, 3.2])
    p = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                  [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                  [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                  [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    return a, c, p


def hartmann6_func(x, d, a, c, p):
    """
    Compute Hartmann6 function at a given point with given arguments.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the function.
    d : integer
        Dimension
    a : 2-D array with shape (4, 6)
    c : 1-D array with shape (4,)
    p : 2-D array with shape (4, 6)

    Returns
    -------
    function value : float
    """
    func_val = 0
    for i in range(c.shape[0]):
        func_val += c[i] * np.exp(-np.sum(a[i] * (x - p[i]) ** 2))
    return -func_val


def hartmann6_grad(x, d, a, c, p):
    """
    Compute Hartmann6 gradient at a given point with given arguments.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the function.
    d : integer
        Dimension
    a : 2-D array with shape (4, 6)
    c : 1-D array with shape (4,)
    p : 2-D array with shape (4, 6)

    Returns
    -------
    grad : 1-D array with shape (d,)
    """
    grad = 0
    for i in range(c.shape[0]):
        grad += (2 * c[i] * a[i, :] * (x - p[i]) *
                 np.exp(-np.sum(a[i] * (x - p[i]) ** 2)))
    return grad
