import numpy as np

from metod import objective_functions as mt_obj


def test_1():
    """Computational example"""
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

    x_points = np.array([[1.2, 3.1, 4.2, 0.9],
                         [9.1, 7.1, 9, 8.5],
                         [5, 5.4, 7.2, 6.8],
                         [1.1, 0.9, 10.1, 0.9]])
    func_args = 4, matrix_test, C, b

    minima_pos = np.zeros((4))
    for j in range(4):
        pos, dist = mt_obj.calc_minimizer_shekel(x_points[j], *func_args)
        minima_pos[j] = pos
    assert(np.all(minima_pos== np.array([0, 1, 2, 3])))


def test_2():
    """Computational example"""
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

    x_points = np.array([[7, 5, 7, 5],
                        [9.1, 7.1, 9, 8.5],
                         [5, 5.4, 7.2, 6.8],
                        [1.1, 0.9, 10.1, 0.9]])
    func_args = 4, matrix_test, C, b

    minima_pos = np.zeros((4))
    for j in range(4):
        pos, dist = mt_obj.calc_minimizer_shekel(x_points[j], *func_args)
        minima_pos[j] = pos
    assert(np.all(minima_pos == np.array([2, 1, 2, 3])))
    assert(np.unique(minima_pos).shape == (3, ))