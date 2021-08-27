import numpy as np

from metod_alg import objective_functions as mt_obj


def test_1():
    """Computational test for mt_obj.shekel_gradient() with d=2."""
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
    grad = mt_obj.shekel_gradient(x, *args)
    assert(grad.shape == (d,))
    assert(np.all(np.round(grad, 3) == np.round(np.array([-0.203, -0.183]) * 0.5, 3)))
