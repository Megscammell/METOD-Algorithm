import numpy as np
from hypothesis import given, settings, strategies as st

from metod import objective_functions as mt_obj


def test_1():
    """Computational test for qing_gradient with d = 3."""
    d = 3
    x = np.array([1, -1, 2])
    grad = mt_obj.qing_gradient(x, d)
    assert(np.all(grad == np.array([0, 4, 8])))


def test_2():
    """Computational test for qing_gradient with d = 3."""
    d = 3
    x = np.array([0.9, -0.5, 0.1])
    grad = mt_obj.qing_gradient(x, d)
    assert(np.all(np.round(grad, 3) == np.array([-0.684, 3.500, -1.196])))
