import numpy as np

from metod import objective_functions as mt_obj


def function_parameters_shekel(lambda_1, lambda_2):
    """
    Create function arguments for the Sum of Gaussians function and gradient.

    Parameters
    ----------
    lambda_1 : integer
               Smallest eigenvalue of diagonal matrix.
    lambda_2 : integer
               Largest eigenvalue of diagonal matrix.

    Returns
    -------
    matrix_test : 3-D arrays with shape (p, d, d), Entries
                  are generated from
                  np.random.uniform(lambda_1 + 0.1,
                                    lambda_2 - 0.1).
    C : 2-D arrays with shape (4, 10).
    B : 1-D array with shape (10, ).
    """
    d = 4
    p = 10
    if type(lambda_1) is not int:
        raise ValueError('lambda_1 must be an integer.')
    if type(lambda_2) is not int:
        raise ValueError('lambda_2 must be an integer.')
    C = np.array([[4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                  [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
                  [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                  [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]])
    b = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    A = np.zeros((p, d, d))
    rotation = np.zeros((p, d, d))
    for j in range(p):
        rotation[j] = mt_obj.calculate_rotation_matrix(d, 3)
        diag_vals = np.zeros(d)
        diag_vals[:2] = np.array([lambda_1, lambda_2])
        diag_vals[2:] = np.random.uniform(lambda_1 + 0.1,
                                          lambda_2 - 0.1, (d - 2))
        A[j] = np.diag(diag_vals)
    matrix_test = np.transpose(rotation, (0, 2, 1)) @ A @ rotation
    return matrix_test, C, b
