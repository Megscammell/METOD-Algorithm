import numpy as np

from metod import objective_functions as mt_obj


def test_1():
    d = 3
    point = np.array([-2.903534, -2.903534, 2.903534])
    index, dist = mt_obj.calc_minimizer_styb(point, d)
    assert(index == 4)
    assert(dist == 0)


def test_2():
    d = 3
    point = np.array([2.903534, -2.903534, 2])
    index, dist = mt_obj.calc_minimizer_styb(point, d)
    assert(index == 5)
    assert(np.round(dist, 4) == 0.9035)
