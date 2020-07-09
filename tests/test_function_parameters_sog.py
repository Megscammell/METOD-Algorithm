import numpy as np

import metod_testing as mtv3


def test_function_parameters_sog():
    """ Testing functionality of slices used in function_parameters_sog and
     comparing results by using for loop.
     Have not used for loop in function_parameters_sog as less efficient.
    """
    p = 4
    d = 5
    store_A = np.zeros((p, d, d))
    store_x0 = np.zeros((p, d))
    store_c = np.zeros((p, ))
    store_rotation = np.zeros((p, d, d))
    matrix_test = np.zeros((p, d, d))
    np.random.seed(90)
    for i in range(p):
        diag_vals = np.zeros(d, )
        a = 1
        diag_vals[0] = a
        b = 10
        diag_vals[1] = b
        for j in range(2, d):
            diag_vals[j] = np.random.uniform(2, 9)
        store_A[i] = np.diag(diag_vals)
        store_c[i] = np.random.uniform(0.5, 1)
        store_rotation[i] = mtv3.calculate_rotation_matrix(d, 3)
        store_x0[i] = np.random.uniform(0, 1, (d, ))
        matrix_test[i] = store_rotation[i].T @ store_A[i] @ store_rotation[i]
    np.random.seed(90)
    t_store_x0, t_matrix_test, t_store_c = (mtv3.function_parameters_sog
                                            (p, d, 1, 10))
    assert(np.all(t_store_x0 == store_x0))
    assert(np.all(t_matrix_test == matrix_test))
    assert(np.all(t_store_c == store_c))
