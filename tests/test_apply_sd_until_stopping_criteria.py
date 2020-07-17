import numpy as np
import pytest
from numpy import linalg as LA
from hypothesis import given, settings, strategies as st

from metod import metod_algorithm_functions as mt_alg
from metod import objective_functions as mt_obj


@settings(max_examples=50, deadline=None)
@given(st.integers(5, 100), st.integers(2, 10))
def test_1(d, p):
    """Ensuring final iteration of steepest descent has norm of gradient
     smaller than tolerance.
    """
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_test = mt_obj.function_parameters_quad(p, d, lambda_1,
                                                            lambda_2)
    func_args = p, store_x0, matrix_test
    tolerance = 0.00001
    option = 'minimize'
    met = 'Nelder-Mead'
    initial_guess = 0.05
    f = mt_obj.quad_function
    g = mt_obj.quad_gradient
    projection = False
    bound_1 = 0
    bound_2 = 1
    usage = 'metod_algorithm'
    relax_sd_it = 1
    point = np.random.uniform(bound_1, bound_2, (d, ))
    sd_iterations, its = (mt_alg.apply_sd_until_stopping_criteria
                          (point, d, projection, tolerance, option, met,
                           initial_guess, func_args, f, g, bound_1, bound_2,
                           usage, relax_sd_it))
    assert(LA.norm(g(sd_iterations[its].reshape(d, ), *func_args)) < tolerance)
    assert(sd_iterations.shape[0] == its + 1)


def test_2():
    """Ensuring that point is overwritten by x_iteration"""
    point = np.array([1, 2, 3, 4, 5])
    c = 0
    while c < 5:
        x_iteration = np.arange(c, c + 5)
        point = x_iteration
        c += 1
    assert(np.all(point == np.array([4, 5, 6, 7, 8])))


def updating_array(d, arr):
    new_p = np.array([1, 2, 3, 4, 5])
    arr = np.vstack([arr, new_p.reshape((1, d))])
    return arr


def test_3():
    """Ensuring that array gets updated in function and new values are given
     to sd_iterations
    """
    d = 5
    arr = np.zeros((1, d))
    p = np.random.uniform(0, 1, (d))
    arr[0] = p.reshape(1, d)
    arr = updating_array(d, arr)
    assert(arr.shape[0] == 2)
    assert(np.all(arr[0] == p))
    assert(np.all(arr[1] == np.array([1, 2, 3, 4, 5])))


def test_4():
    """Checking functionality of np.vstack and ensuring it stores points as
     expected.
    """
    d = 10
    store_x = np.zeros((2, d))
    x = np.arange(1, 11).reshape(d, 1)
    store_x[0] = x.reshape(1, d)
    x = np.arange(11, 21).reshape(d, 1)
    store_x[1] = x.reshape(1, d)
    for j in range(2, 8):
        x = np.arange((j * 10) + 1, ((j * 10) + 11)).reshape(d, 1)
        store_x = np.vstack([store_x, x.reshape(1, d)])
    print(store_x)
    assert(np.all(store_x[2] == np.arange(21, 31)))
    assert(np.all(store_x[3] == np.arange(31, 41)))
    assert(np.all(store_x[4] == np.arange(41, 51)))
    assert(np.all(store_x[5] == np.arange(51, 61)))
    assert(np.all(store_x[6] == np.arange(61, 71)))
    assert(np.all(store_x[7] == np.arange(71, 81)))


def test_5():
    """Checks that error is raised if more than 200 iterations are
    computed.
    """
    np.random.seed(90)
    d = 100
    p = 50
    lambda_1 = 1
    lambda_2 = 50
    store_x0, matrix_test = mt_obj.function_parameters_quad(p, d, lambda_1,
                                                            lambda_2)
    func_args = p, store_x0, matrix_test
    tolerance = 0.000000001
    option = 'minimize'
    met = 'Nelder-Mead'
    initial_guess = 0.05
    f = mt_obj.quad_function
    g = mt_obj.quad_gradient
    projection = False
    bound_1 = 0
    bound_2 = 1
    usage = 'metod_algorithm'
    relax_sd_it = 1
    point = np.random.uniform(bound_1, bound_2, (d, ))
    with pytest.raises(ValueError):
        mt_alg.apply_sd_until_stopping_criteria(point, d, projection,
                                                tolerance, option, met,
                                                initial_guess, func_args, f,
                                                g, bound_1, bound_2, usage,
                                                relax_sd_it)


@settings(max_examples=50, deadline=None)
@given(st.integers(5, 100), st.integers(2, 10), st.integers(10, 30))
def test_6(d, p, iterations):
    """Ensuring final iteration of steepest descent has norm of gradient
    smaller than tolerance.
    """
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_test = mt_obj.function_parameters_quad(p, d, lambda_1,
                                                            lambda_2)
    func_args = p, store_x0, matrix_test
    option = 'minimize'
    met = 'Nelder-Mead'
    initial_guess = 0.05
    f = mt_obj.quad_function
    g = mt_obj.quad_gradient
    projection = True
    point = np.random.uniform(0, 1, (d, ))
    bound_1 = 0
    bound_2 = 1
    usage = 'metod_analysis'
    relax_sd_it = 1
    tolerance = 15
    sd_iterations, its = (mt_alg.apply_sd_until_stopping_criteria
                          (point, d, projection, tolerance, option, met,
                           initial_guess, func_args, f, g, bound_1, bound_2,
                           usage, relax_sd_it))
    assert(tolerance == its)
