import numpy as np

import metod_testing as mtv3

def test_1():
    """Computational test where function value is 0.002025"""
    p = 2
    d = 2
    gamma = 0.05
    point = np.array([0.1, 0.2])
    store_x0 = np.zeros((p, d))
    store_x0[0] = np.array([0.05, 0.05])
    store_x0[1] = np.array([0.9, 0.7])
    matrix_test = np.zeros((p,d,d))
    matrix_test[0] = np.array([[1, 0],[0, 10]])
    matrix_test[1] = np.array([[1, 0],[0, 10]])
    func_args = p, store_x0, matrix_test
    f = mtv3.quad_function
    g = mtv3.quad_gradient
    func_val = mtv3.minimise_function(gamma, point, f, g, *func_args)

    #assert(np.round(func_val,6) == 0.002025)
    assert(np.round(func_val,6) == 0.029253)
