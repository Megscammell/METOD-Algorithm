import numpy as np

from metod_alg import objective_functions as mt_obj


def test_1():
    """Computational test for mt_obj.shekel_function() with d=2."""
    p = 3
    d = 2
    matrix_test = np.array([[[1, 0],
                            [0, 1]],
                            [[1, 0],
                            [0, 1]],
                            [[1, 0],
                            [0, 1]]])
    C = np.array([[10, 3, 0.5],
                  [11, 5, 1]])
    b = np.array([1, 1, 1])
    x = np.array([2, 4])
    args = p, matrix_test, C, b
    func_val = mt_obj.shekel_function(x, *args)
    assert(np.round(func_val, 4) == -0.4237)

