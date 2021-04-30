import numpy as np
from hypothesis import given, settings, strategies as st

from metod_alg import objective_functions as mt_obj


def test_1():
    """Computational test for qing_function with d = 3."""
    d = 3
    x = np.array([1, -1, 2])
    func_val = mt_obj.qing_function(x, d)
    assert(func_val == 2)


def test_2():
    """Computational test for qing_function with d = 3."""
    d = 3
    x = np.array([0.9, -0.5, 0.1])
    func_val = mt_obj.qing_function(x, d)
    assert(np.round(func_val, 4) == 12.0387)
