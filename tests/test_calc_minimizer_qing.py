import numpy as np

from metod_alg import objective_functions as mt_obj


def test_1():
    """
    Numerical test for mt_obj.calc_minimizer_qing().
    """
    d = 3
    point = np.array([-1, -1.4142, 1.732])
    index = mt_obj.calc_minimizer_qing(point, d)
    assert(index == 4)


def test_2():
    """
    Numerical test for mt_obj.calc_minimizer_qing().
    """
    d = 3
    point = np.array([1, -1.4142, 1.7])
    index = mt_obj.calc_minimizer_qing(point, d)
    assert(index == 5)
