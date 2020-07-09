import numpy as np


def sog_gradient(point, p, sigma_sq, store_x0, matrix_test, store_c):
    """Compute the gradient of the sum of Gaussians function at point.

    Keyword arguments:
    point -- is a (d,) array and the function is evaluated at point.
    p, sigma_sq, store_x0, matrix_test, store_c -- parameters needed to
     compute the gradient
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
