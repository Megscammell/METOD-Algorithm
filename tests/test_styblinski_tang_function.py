import numpy as np
from hypothesis import given, settings, strategies as st

from metod_alg import objective_functions as mt_obj


def test_1():
    """Computational test for styblinski_tang_function with d = 3."""
    d = 3
    x = np.array([1, -1, 2])
    func_val = mt_obj.styblinski_tang_function(x)
    assert(np.round(func_val, 2) == -34.00)


def test_2():
    """Computational test for styblinski_tang with d = 3."""
    d = 3
    x = np.array([0.9, -0.5, 0.1])
    func_val = mt_obj.styblinski_tang_function(x)
    assert(np.round(func_val, 4) == -6.9506)
