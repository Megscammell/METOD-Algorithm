import numpy as np


def sog_gradient(point, p, sigma_sq, store_x0, matrix_test, store_c):
    """Compute Sum of Gaussians gradient at a given point with given arguments.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the gradient.
    p : integer
        Number of local minima.
    sigma_sq: float or integer
              Value of sigma squared.
    store_x0 : 2-D array with shape (p, d).
    matrix_test : 3-D array with shape (p, d, d).
    store_c : 1-D array with shape (p, ).

    Returns
    -------
    individual_gradient : 1-D array with shape (d, )
                          Gradient at point.
    """
    individual_gradient = 0
    for i in range(p):
        grad_val_1 = (store_c[i] / sigma_sq) * np.exp((- 1 / (2 * sigma_sq)) *
                                                      np.transpose(point -
                                                                   store_x0[i]
                                                                   ) @
                                                      matrix_test[i] @
                                                      (point - store_x0[i]))
        grad_val_2 = (matrix_test[i] @ (point - store_x0[i]))
        individual_gradient += grad_val_1 * grad_val_2

    return individual_gradient
