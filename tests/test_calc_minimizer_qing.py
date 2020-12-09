import numpy as np

from metod import objective_functions as mt_obj


def test_1():
    d = 3
    point = np.array([-1, -1.4142, 1.732])
    index, dist = mt_obj.calc_minimizer_qing(point, d)
    assert(index == 4)
    assert(np.round(dist, 3) == 0)


def test_2():
    d = 3
    point = np.array([1, -1.4142, 1.7])
    index, dist = mt_obj.calc_minimizer_qing(point, d)
    assert(index == 5)
    assert(np.round(dist, 3) == 0.032)
