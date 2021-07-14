import numpy as np
from hypothesis import given, settings, strategies as st

from metod_alg import objective_functions as mt_obj


def test_1():
    """Computational test for mt_obj.styblinski_tang_gradient() with d = 3."""
    d = 3
    x = np.array([1, -1, 2])
    grad = mt_obj.styblinski_tang_gradient(x)
    assert(np.all(np.round(grad, 1) ==
           np.array([-11.5, 16.5, -13.5])))


def test_2():
    """Computational test for mt_obj.styblinski_tang_gradient() with d = 3."""
    d = 3
    x = np.array([0.9, -0.5, 0.1])
    grad = mt_obj.styblinski_tang_gradient(x)
    assert(np.all(np.round(grad, 4) ==
           np.array([-10.4420, 10.2500, 0.9020])))
