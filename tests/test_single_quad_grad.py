import numpy as np

from metod_alg import objective_functions as mt_obj


def test_1():
    """Computational test for mt_obj.single_quad_gradient()."""
    d = 2
    x0 = np.array([0.5, 0.6])
    A = np.array([[1, 0], [0, 10]])
    rotation = np.identity(d)
    x = np.array([1, 1])
    function_parameters = (x0, A, rotation)
    grad = mt_obj.single_quad_gradient(x, *function_parameters)
    assert(np.all(grad == np.array([0.5, 4])))


def test_2():
    """Computational test for mt_obj.single_quad_gradient()."""
    d = 2
    x0 = np.array([0.5, 0.6])
    A = np.array([[1, 0], [0, 10]])
    rotation = np.array([[ 0.4,  0.9], [-0.9,  0.4]])
    x = np.array([1, 1])
    function_parameters = (x0, A, rotation)
    grad = mt_obj.single_quad_gradient(x, *function_parameters)
    assert(np.all(np.round(grad, 3) == np.array([2.834, -0.656])))
        


