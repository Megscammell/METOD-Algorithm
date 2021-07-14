import numpy as np

from metod_alg import objective_functions as mt_obj


def test_1():
    """
    Numerical test for mt_obj.calc_minimizer_styb().
    """
    point = np.array([-2.903534, 2.746803, -2.903534])
    index = mt_obj.calc_minimizer_styb(point)
    assert(index == 2)


def test_2():
    """
    Numerical test for mt_obj.calc_minimizer_styb().
    """
    point = np.array([2.746803, -2.903534, -2.85])
    index = mt_obj.calc_minimizer_styb(point)
    assert(index == 1)
