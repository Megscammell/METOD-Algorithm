import numpy as np

from metod import objective_functions as mt_obj


def test_1():
    """Computational test where the function value is 3.7."""
    d = 2
    x0 = np.array([0.5, 0.6])
    A = np.array([[1, 0], [0, 10]])
    rotation = np.identity(d)
    x = np.array([1, 1])
    function_parameters = (x0, A, rotation)
    value = mt_obj.single_quad_function(x, *function_parameters)
    assert(value == 0.925)


def test_2():
    """Computational test where the function value is 3.7."""
    d = 2
    x0 = np.array([0.5, 0.6])
    A = np.array([[1, 0], [0, 10]])
    rotation = np.array([[ 0.4,  0.9], [-0.9,  0.4]])
    x = np.array([1, 1])
    function_parameters = (x0, A, rotation)
    value = mt_obj.single_quad_function(x, *function_parameters)
    assert(value == 0.5773)
        


