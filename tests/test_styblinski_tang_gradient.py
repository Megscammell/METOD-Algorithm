import numpy as np
from hypothesis import given, settings, strategies as st

from metod import objective_functions as mt_obj


def test_1():
    """Computational test for styblinski_tang_function with d = 3."""
    d = 3
    x = np.array([1, -1, 2])
    grad = mt_obj.styblinski_tang_gradient(x, d)
    assert(np.all(np.round(grad, 1) ==
           np.array([-7.7, 11, -9])))


def test_2():
    """Computational test for styblinski_tang with d = 3."""
    d = 3
    x = np.array([0.9, -0.5, 0.1])
    grad = mt_obj.styblinski_tang_gradient(x, d)
    assert(np.all(np.round(grad, 4) ==
           np.array([-6.9613, 6.8333, 0.6013])))
