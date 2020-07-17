import numpy as np

from metod import objective_functions as mt_obj


def function_parameters_sog(p, d, lambda_1, lambda_2):
    """Create function arguments for the Sum of Gaussians function and gradient.

    Parameters
    ----------
    p : integer
        Number of local minima.
    d : integer
        Size of dimension.
    lambda_1 : integer
               Smallest eigenvalue of diagonal matrix.
    lambda_2 : integer
               Largest eigenvalue of diagonal matrix.

    Returns
    -------
    store_x0 : 2-D arrays with shape (p, d). Entries are
               generated from np.random.uniform(0, 1).
    matrix_test : 3-D arrays with shape (p, d, d), Entries
                  are generated from
                  np.random.uniform(lambda_1 + 0.1,
                                    lambda_2 - 0.1).
    store_c : 1-D arrays with shape (p, ). Entries are
              generated from np.random.uniform(0.5, 1).

    """
    if type(p) is not int:
        raise ValueError('p must be an integer.')
    if type(d) is not int:
        raise ValueError('d must be an integer.')
    if type(lambda_1) is not int:
        raise ValueError('lambda_1 must be an integer.')
    if type(lambda_2) is not int:
        raise ValueError('lambda_2 must be an integer.')
    store_A = np.zeros((p, d, d))
    store_x0 = np.zeros((p, d))
    store_rotation = np.zeros((p, d, d))
    store_c = np.zeros((p))

    for i in range(p):
        diag_vals = np.zeros(d)
        diag_vals[:2] = np.array([lambda_1, lambda_2])
        diag_vals[2:] = np.random.uniform(lambda_1 + 0.1, lambda_2 - 0.1, (d -
                                                                           2))
        store_A[i] = np.diag(diag_vals)
        store_c[i] = np.random.uniform(0.5, 1)
        store_rotation[i] = mt_obj.calculate_rotation_matrix(d, 3)
        store_x0[i] = np.random.uniform(0, 1, (d))
    matrix_test = (np.transpose(store_rotation, (0, 2, 1)) @ store_A @
                   store_rotation)
    return store_x0, matrix_test, store_c
