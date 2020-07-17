import numpy as np

from metod import objective_functions as mt_obj
from metod import metod_algorithm_functions as mt_alg


def test_1():
    """Computational test where function value is 0.029253"""
    p = 2
    d = 2
    gamma = 0.05
    point = np.array([0.1, 0.2])
    store_x0 = np.zeros((p, d))
    store_x0[0] = np.array([0.05, 0.05])
    store_x0[1] = np.array([0.9, 0.7])
    matrix_test = np.zeros((p, d, d))
    matrix_test[0] = np.array([[1, 0], [0, 10]])
    matrix_test[1] = np.array([[1, 0], [0, 10]])
    func_args = p, store_x0, matrix_test
    f = mt_obj.quad_function
    g = mt_obj.quad_gradient
    func_val = mt_alg.minimise_function(gamma, point, f, g, *func_args)
    assert(np.round(func_val, 6) == 0.029253)
