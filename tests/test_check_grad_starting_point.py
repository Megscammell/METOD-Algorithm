import numpy as np
import pytest

from metod_alg import objective_functions as mt_obj
from metod_alg import metod_algorithm_functions as mt_alg


def test_1():
    """
    Asserts error message when too many starting points have a very small
    gradient.
    """
    np.random.seed(90)
    g = mt_obj.sog_gradient
    d = 100
    P = 50
    lambda_1 = 1
    lambda_2 = 10
    sigma_sq = 1
    store_x0, matrix_combined, store_c = (mt_obj.function_parameters_sog
                                          (P, d, lambda_1, lambda_2))
    func_args = P, sigma_sq, store_x0, matrix_combined, store_c
    tolerance = 0.00001
    bounds_set_x = (0, 1)
    point_index = 0
    x = np.random.uniform(*bounds_set_x, (d,))
    set_x = 'random'
    sobol_points = None
    no_points = 0
    num_points = 1
    with pytest.raises(ValueError):
        mt_alg.check_grad_starting_point(x, point_index, no_points,
                                         bounds_set_x, sobol_points,
                                         d, g, func_args, set_x,
                                         tolerance, num_points)


def test_2():
    """
    Checks functionality of mt_alg.check_grad_starting_point()
    with Sobol points.
    """
    np.random.seed(90)
    g = mt_obj.several_quad_gradient
    d = 100
    p = 50
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_test = (mt_obj.function_parameters_several_quad
                             (p, d, lambda_1, lambda_2))
    func_args = p, store_x0, matrix_test
    tolerance = 0.00001
    bounds_set_x = (0, 1)
    point_index = 0
    set_x = 'sobol'
    num_points = 1000
    sobol_points = mt_alg.create_sobol_sequence_points(bounds_set_x[0],
                                                       bounds_set_x[1], d,
                                                       num_points)
    x = sobol_points[point_index]
    original_x = np.copy(x)
    no_points = 5
    point_index, x, grad = (mt_alg.check_grad_starting_point
                            (x, point_index, no_points, bounds_set_x,
                             sobol_points, d, g, func_args, set_x, tolerance,
                             num_points))
    assert(point_index == 0)
    assert(np.all(x == original_x))
    assert(no_points == 5)
    assert(np.all(g(x, *func_args) == grad))


def test_3():
    """
    Checks functionality of mt_alg.check_grad_starting_point()
    with random points.
    """
    np.random.seed(90)
    g = mt_obj.several_quad_gradient
    d = 100
    p = 50
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_test = (mt_obj.function_parameters_several_quad
                             (p, d, lambda_1, lambda_2))
    func_args = p, store_x0, matrix_test
    tolerance = 0.00001
    bounds_set_x = (0, 1)
    point_index = 0
    x = np.random.uniform(*bounds_set_x, (d, ))
    original_x = np.copy(x)
    set_x = 'random'
    sobol_points = None
    no_points = 5
    num_points = 1000
    point_index, x, grad = (mt_alg.check_grad_starting_point
                            (x, point_index, no_points, bounds_set_x,
                             sobol_points, d, g, func_args, set_x, tolerance,
                             num_points))
    assert(point_index == 0)
    assert(np.all(x == original_x))
    assert(no_points == 5)
    assert(np.all(g(x, *func_args) == grad))


def test_4():
    """
    Checks functionality of mt_alg.check_grad_starting_point()
    with random points.
    """
    np.random.seed(90)
    g = mt_obj.sog_gradient
    d = 100
    P = 50
    lambda_1 = 1
    lambda_2 = 10
    sigma_sq = 2
    store_x0, matrix_combined, store_c = (mt_obj.function_parameters_sog
                                          (P, d, lambda_1, lambda_2))
    func_args = P, sigma_sq, store_x0, matrix_combined, store_c
    tolerance = 0.00001
    bounds_set_x = (0, 1)
    point_index = 0
    x = np.random.uniform(*bounds_set_x, (d,))
    original_x = np.copy(x)
    set_x = 'random'
    sobol_points = None
    no_points = 1
    num_points = 1000
    with pytest.warns(RuntimeWarning):
        point_index, x, grad = (mt_alg.check_grad_starting_point
                                (x, point_index, no_points, bounds_set_x,
                                 sobol_points, d, g, func_args, set_x,
                                 tolerance, num_points))
    assert(point_index > 0)
    assert(np.all(x != original_x))
    assert(no_points == 1)
    assert(np.all(g(x, *func_args) == grad))
