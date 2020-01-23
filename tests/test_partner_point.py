import numpy as np

import metod_testing as mtv3

def test_1():
    """Computational example to compute a single partner point
    """
    p = 2 
    d = 2
    beta = 0.05
    f = mtv3.quad_function
    g = mtv3.quad_gradient
    matrix_test = np.zeros((p, d, d))
    store_x0 = np.zeros((p, d))
    matrix_test[0] = np.array([[2, 4], [4, 1]])
    matrix_test[1] = np.array([[2, 4], [4, 1]])
    x = np.array([2, 1])
    store_x0[0] = np.array([3, 4])
    store_x0[1] = np.array([1, 0])
    func_args = p, store_x0, matrix_test
    
    partner_point = partner_point(x, beta, d, g, func_args)
    
    assert(partner_point == np.array([1.4, 0.5]))
