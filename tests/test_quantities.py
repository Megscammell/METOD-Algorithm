import numpy as np

import metod.metod_analysis as mt_ays


def test_1():
    """Numerical example."""
    matrix_test = np.zeros((2, 2, 2))
    store_x0 = np.zeros((2, 2))
    x = np.array([0.1, 0.9])
    y = np.array([0.9, 0.3])
    beta = 0.05
    matrix_test[0] = np.array([[1, 0], [0, 5]])
    matrix_test[1] = np.array([[1, 0], [0, 10]])
    store_x0[0] = np.array([0.2, 0.8])
    store_x0[1] = np.array([0.8, 0.1])
    func_args = 2, store_x0, matrix_test
    quantities_array, sum_quantities = mt_ays.quantities(x, y, 0, 1, beta, 2,
                                                         store_x0, matrix_test)
    calc = mt_ays.check_quantities(beta, x, y, func_args)
    assert(np.round(quantities_array[0], 6) == 0.007225)
    assert(np.round(quantities_array[1], 4) == 0.0234)
    assert(np.round(quantities_array[2], 4) == -0.0249)
    assert(np.round(quantities_array[3], 3) == 0.228)
    assert(np.round(quantities_array[4], 3) == -0.154)
    assert(np.round(sum_quantities, 5) == np.round(calc, 5))
