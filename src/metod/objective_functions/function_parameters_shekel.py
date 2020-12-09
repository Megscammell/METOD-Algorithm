import numpy as np

from metod import objective_functions as mt_obj


def function_parameters_shekel(p, d, b_val, lambda_1, lambda_2):
    """
    Create function arguments for the Sum of Gaussians function and gradient.

    Parameters
    ----------
    p : integer
        Number of local minima.
    d : integer
        Size of dimension.
    b_val : integer or float
            Value of each element of array b.
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
    C : 2-D arrays with shape (d, p). Entries are
        generated from np.random.uniform(0, 1).
    B : 1-D array with shape (p, ). Each entry is b_val.
    """
    if type(p) is not int:
        raise ValueError('p must be an integer.')
    if type(d) is not int:
        raise ValueError('d must be an integer.')
    if type(lambda_1) is not int:
        raise ValueError('lambda_1 must be an integer.')
    if type(lambda_2) is not int:
        raise ValueError('lambda_2 must be an integer.')
    if (type(b_val) is not int) and (type(b_val) is not float):
        raise ValueError('b_val must be an integer.')
    C = np.zeros((d, p))
    b = np.zeros((p))
    A = np.zeros((p, d, d))
    rotation = np.zeros((p, d, d))

    for j in range(p):
        C[:, j] = np.random.uniform(0, 1, (d)) 
        rotation[j] = mt_obj.calculate_rotation_matrix(d, 3)
        b[j] = b_val
        diag_vals = np.zeros(d)
        diag_vals[:2] = np.array([lambda_1, lambda_2])
        diag_vals[2:] = np.random.uniform(lambda_1 + 0.1,
                                          lambda_2 - 0.1, (d - 2))
        A[j] = np.diag(diag_vals)
    matrix_test = np.transpose(rotation, (0, 2, 1)) @ A @ rotation
    return matrix_test, C, b
