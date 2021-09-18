import numpy as np


# def sog_function(x, p, exp_const, store_x0, matrix_test, store_c):
#     """
#     Compute Sum of Gaussians function at a given point with given arguments.

#     Parameters
#     ----------
#     point : 1-D array with shape (d, )
#             A point used to evaluate the function.
#     p : integer
#         Number of local minima.
#     sigma_sq: float or integer
#               Value of sigma squared.
#     store_x0 : 2-D array with shape (p, d).
#     matrix_test : 3-D array with shape (p, d, d).
#     store_c : 1-D array with shape (p, ).

#     Returns
#     -------
#     float(-function_val) : float
#                            Function value.
#     """
#     d = x.shape[0]
#     f_val = (store_c.reshape(p, 1, 1) @
#              (np.exp((-1 / (2 * exp_const)) *
#               np.transpose((x - store_x0).reshape(p, d, 1), (0, 2, 1)) @
#               matrix_test @ (x-store_x0).reshape(p, d, 1))))
#     sum_f_val = np.sum(f_val, axis=0)
#     return float(-sum_f_val)


# def sog_gradient(x, p, exp_const, store_x0, matrix_test, store_c):
#     """
#     Compute Sum of Gaussians gradient at a given point with given arguments.

#     Parameters
#     ----------
#     point : 1-D array with shape (d, )
#             A point used to evaluate the gradient.
#     p : integer
#         Number of local minima.
#     sigma_sq: float or integer
#               Value of sigma squared.
#     store_x0 : 2-D array with shape (p, d).
#     matrix_test : 3-D array with shape (p, d, d).
#     store_c : 1-D array with shape (p, ).

#     Returns
#     -------
#     total_gradient : 1-D array with shape (d, )
#                      Gradient at point.
#     """
#     d = x.shape[0]
#     grad_val_1 = ((store_c.reshape(p, 1, 1) * (1/exp_const))
#                   @ np.exp((-1 / (2 * exp_const)) *
#                   np.transpose((x - store_x0).reshape(p, d, 1), (0, 2, 1)) @
#                   matrix_test @ (x-store_x0).reshape(p, d, 1)))
#     grad_val_2 = (matrix_test @ (x-store_x0).reshape(p, d, 1))
#     gradient = grad_val_1 * grad_val_2
#     sum_g_val = np.sum(gradient, axis=0)
#     return sum_g_val.reshape(d,)


def sog_function(x, p, exp_const, store_x0, matrix_test, store_c):
    """
    Compute Sum of Gaussians function at a given point with given arguments.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the function.
    p : integer
        Number of local minima.
    sigma_sq: float or integer
              Value of sigma squared.
    store_x0 : 2-D array with shape (p, d).
    matrix_test : 3-D array with shape (p, d, d).
    store_c : 1-D array with shape (p, ).

    Returns
    -------
    float(-function_val) : float
                           Function value.
    """
    d = x.shape[0]
    f_val = (store_c.reshape(p, 1, 1) @
             (np.exp((-1 / (2 * exp_const)) *
              np.transpose((x - store_x0).reshape(p, d, 1), (0, 2, 1)) @
              matrix_test @ (x-store_x0).reshape(p, d, 1))))
    sum_f_val = np.sum(f_val, axis=0)
    return float(-sum_f_val * exp_const)


def sog_gradient(x, p, exp_const, store_x0, matrix_test, store_c):
    """
    Compute Sum of Gaussians gradient at a given point with given arguments.

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
    total_gradient : 1-D array with shape (d, )
                     Gradient at point.
    """
    d = x.shape[0]
    grad_val_1 = ((store_c.reshape(p, 1, 1))
                  @ np.exp((-1 / (2 * exp_const)) *
                  np.transpose((x - store_x0).reshape(p, d, 1), (0, 2, 1)) @
                  matrix_test @ (x-store_x0).reshape(p, d, 1)))
    grad_val_2 = (matrix_test @ (x-store_x0).reshape(p, d, 1))
    gradient = grad_val_1 * grad_val_2
    sum_g_val = np.sum(gradient, axis=0)
    return sum_g_val.reshape(d,)