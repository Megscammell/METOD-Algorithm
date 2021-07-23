import numpy as np

from metod_alg import objective_functions as mt_obj


def test_1():
    """Computational example for mt_obj.calc_minimizer_shekel()."""
    C = np.array([[1, 9, 5, 1],
                  [3, 7, 5, 1],
                  [4, 9, 7, 10],
                  [1, 8, 7, 1]])
    b = np.repeat(1, 4)
    matrix_test = np.array([[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]],
                            [[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]],
                            [[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]],
                            [[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]])

    x_points = np.array([[1.01, 3.05, 4.01, 1.02],
                         [9.05, 7.01, 9.02, 8.01],
                         [5.02, 5.05, 7.0, 7.01],
                         [1.01, 1.0, 10.0, 1.01]])
    func_args = 4, matrix_test, C, b

    minima_pos = np.zeros((4))
    for j in range(4):
        pos = mt_obj.calc_minimizer_shekel(x_points[j], *func_args)
        minima_pos[j] = pos
    assert(np.all(minima_pos == np.array([0, 1, 2, 3])))


def test_2():
    """Computational example for mt_obj.calc_minimizer_shekel()."""
    C = np.array([[1, 9, 5, 1],
                  [3, 7, 5, 1],
                  [4, 9, 7, 10],
                  [1, 8, 7, 1]])
    b = np.repeat(1, 4)
    matrix_test = np.array([[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]],
                            [[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]],
                            [[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]],
                            [[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]])

    x_points = np.array([[1.01, 1.0, 10.0, 1.01],
                         [9.05, 7.01, 9.02, 8.01],
                         [1.01, 3.05, 4.01, 1.02],
                         [1.01, 1.0, 10.0, 1.01]])
    func_args = 4, matrix_test, C, b

    minima_pos = np.zeros((4))
    for j in range(4):
        pos = mt_obj.calc_minimizer_shekel(x_points[j], *func_args)
        minima_pos[j] = pos
    assert(np.all(minima_pos == np.array([3, 1, 0, 3])))
    assert(np.unique(minima_pos).shape == (3, ))
