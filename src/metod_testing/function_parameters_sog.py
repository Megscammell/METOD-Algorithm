import numpy as np

import metod_testing as mtv3


def function_parameters_sog(p, d, lambda_1, lambda_2):
    """Create set of function parameters. Note that this will not be used in main algorithm.

    Keyword arguments:
    p -- number of local minima
    d -- is the dimension
    lambda_1 -- smallest eigenvalue of diagonal matrix
    lambda_2 -- largest eigenvalue of diagonal matrix
    """
    store_A = np.zeros((p, d, d))
    store_x0 = np.zeros((p, d))
    store_rotation = np.zeros((p, d, d))
    store_c = np.zeros((p))       

    for i in range(p):
        diag_vals = np.zeros(d)
        diag_vals[:2] = np.array([lambda_1, lambda_2])
        diag_vals[2:] = np.random.uniform(lambda_1 + 1, lambda_2 - 1, (d - 2))
        store_A[i] = np.diag(diag_vals)       
        store_c[i] = np.random.uniform(0.5, 1)
        store_rotation[i] = mtv3.calculate_rotation_matrix(d, 3)
        store_x0[i] = np.random.uniform(0, 1, (d))
    matrix_test = np.transpose(store_rotation, (0, 2, 1)) @ store_A @ store_rotation
    return store_x0, matrix_test, store_c
