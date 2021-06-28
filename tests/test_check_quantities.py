import numpy as np

from metod_alg import metod_analysis as mt_ays
from metod_alg import objective_functions as mt_obj


def test_1():
    """
    Computational test, where x is closest to store_x0[1] and y is closest
    to store_x0[0].
    """
    g = mt_obj.several_quad_gradient
    p = 2
    d = 2
    beta = 0.005
    matrix_test = np.zeros((p, d, d))
    store_x0 = np.zeros((p, d))
    matrix_test[0] = np.array([[2, 4], [4, 1]])
    matrix_test[1] = np.array([[2, 4], [4, 1]])
    x = np.array([2, 1])
    y = np.array([4, 6])
    store_x0[0] = np.array([3, 4])
    store_x0[1] = np.array([1, 0])
    func_args = p, store_x0, matrix_test
    val = mt_ays.check_quantities(beta, x, y, g, func_args)
    assert(np.round(val, 6) == -0.129575)
