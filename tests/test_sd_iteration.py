import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from metod import objective_functions as mt_obj
from metod import metod_algorithm_functions as mt_alg


def func_params(d=20, p=5, lambda_1=1, lambda_2=10):
    """Generate function parameters that will be used for tests."""
    option = 'minimize'
    met = 'Nelder-Mead'
    initial_guess = 0.05
    f = mt_obj.quad_function
    g = mt_obj.quad_gradient
    store_x0, matrix_test = mt_obj.function_parameters_quad(p, d, lambda_1,
                                                            lambda_2)
    func_args = p, store_x0, matrix_test
    projection = False
    bound_1 = 0
    bound_2 = 1
    relax_sd_it = 1
    point = np.random.uniform(0, 1, (d, ))
    return (point, projection, option, met, initial_guess,
            func_args, f, g, bound_1, bound_2, relax_sd_it)


def test_1():
    """Testing np.clip method with for loop to ensure points
    are projected correctly.
    """
    projection = True
    bound_1 = 0
    bound_2 = 1
    x = np.array([0.1, 0.5, 0.9, -0.9, 1.1, 0.9, -0.2, 1.1, 0.1, -0.5])
    old_x = np.copy(x)
    if projection is True:
        for j in range(10):
            if x[j] > 1:
                x[j] = 1
            if x[j] < 0:
                x[j] = 0
    assert(np.all(x == np.clip(old_x, bound_1, bound_2)))


@settings(max_examples=50, deadline=None)
@given(st.integers(5, 100), st.integers(2, 10))
def test_2(d, p):
    """Ensuring shape of new iteration is (d, ) when projection is
    False
    """
    (point, projection, option, met, initial_guess,
     func_args, f, g, bound_1, bound_2, relax_sd_it) = func_params(d, p)
    new_point = mt_alg.sd_iteration(point, projection, option, met,
                                    initial_guess, func_args, f, g, bound_1,
                                    bound_2, relax_sd_it)
    assert(new_point.shape == (d, ))


@settings(max_examples=50, deadline=None)
@given(st.integers(5, 100), st.integers(2, 10))
def test_3(d, p):
    """Ensuring shape of new iteration is (d, ) when projection is
    True.
    """
    (point, projection, option, met, initial_guess,
     func_args, f, g, bound_1, bound_2, relax_sd_it) = func_params(d, p)
    projection = False
    new_point = mt_alg.sd_iteration(point, projection, option, met,
                                    initial_guess, func_args, f, g, bound_1,
                                    bound_2, relax_sd_it)
    assert(new_point.shape == (d, ))


@settings(max_examples=50, deadline=None)
@given(st.integers(20, 100), st.integers(2, 10))
def test_4(d, p):
    """Ensuring shape of new iteration is (d, ) when minimize_scalar is
    selected with Golden method and projection is False.
    """
    (point, projection, option, met, initial_guess,
     func_args, f, g, bound_1, bound_2, relax_sd_it) = func_params(d, p)
    option = 'minimize_scalar'
    met = 'Golden'
    new_point = mt_alg.sd_iteration(point, projection, option, met,
                                    initial_guess, func_args, f, g, bound_1,
                                    bound_2, relax_sd_it)
    assert(new_point.shape == (d, ))


@settings(max_examples=50, deadline=None)
@given(st.integers(20, 100), st.integers(2, 10))
def test_5(d, p):
    """Ensuring shape of new iteration is (d, ) when minimize_scalar is
    selected with Golden method and projection is True.
    """
    (point, projection, option, met, initial_guess,
     func_args, f, g, bound_1, bound_2, relax_sd_it) = func_params(d, p)
    option = 'minimize_scalar'
    met = 'Golden'
    projection = True
    new_point = mt_alg.sd_iteration(point, projection, option, met,
                                    initial_guess, func_args, f, g, bound_1,
                                    bound_2, relax_sd_it)
    assert(new_point.shape == (d, ))


def test_6():
    """Ensuring shape of new iteration is (d, ) when minimize_scalar is
    selected with Bounded method and projection is False.
    """
    d = 50
    p = 5
    (point, projection, option, met, initial_guess,
     func_args, f, g, bound_1, bound_2, relax_sd_it) = func_params(d, p)
    option = 'minimize_scalar'
    met = 'Bounded'
    projection = False
    new_point = mt_alg.sd_iteration(point, projection, option, met,
                                    initial_guess, func_args, f, g, bound_1,
                                    bound_2, relax_sd_it)
    assert(new_point.shape == (d, ))


def test_7():
    """Ensuring  error is raised when met is not specified correctly
    for minimize option.
    """
    (point, projection, option, met, initial_guess, func_args, f, g, bound_1,
     bound_2, relax_sd_it) = func_params()
    met = 'Nelder-Mead_v2'
    with pytest.raises(ValueError):
        mt_alg.sd_iteration(point, projection, option, met, initial_guess,
                            func_args, f, g, bound_1, bound_2, relax_sd_it)


def test_8():
    """Ensuring  error is raised if step size is less than zero for
     minimize_scalar option.
    """
    np.random.seed(3)
    d = 2
    p = 3
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_test = mt_obj.function_parameters_quad(p, d, lambda_1,
                                                            lambda_2)
    func_args = p, store_x0, matrix_test
    option = 'minimize_scalar'
    met = 'Golden'
    initial_guess = 0.05
    f = mt_obj.quad_function
    g = mt_obj.quad_gradient
    projection = False
    bound_1 = 0
    bound_2 = 1
    relax_sd_it = 1
    point = np.random.uniform(0, 1, (d, ))
    with pytest.raises(ValueError):
        mt_alg.sd_iteration(point, projection, option, met,
                            initial_guess, func_args, f, g,
                            bound_1, bound_2, relax_sd_it)


def test_9():
    """Ensuring error is raised if step size is less than zero for
     minimize option.
    """
    np.random.seed(3)
    d = 2
    p = 10
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_test = (mt_obj.function_parameters_quad
                             (p, d, lambda_1, lambda_2))
    func_args = p, store_x0, matrix_test
    option = 'minimize'
    met = 'Powell'
    initial_guess = 0.05
    f = mt_obj.quad_function
    g = mt_obj.quad_gradient
    projection = False
    bound_1 = 0
    bound_2 = 1
    relax_sd_it = 1
    point = np.random.uniform(0, 1, (d, ))

    with pytest.raises(ValueError):
        mt_alg.sd_iteration(point, projection, option, met,
                            initial_guess, func_args, f, g, bound_1, bound_2,
                            relax_sd_it)


def test_10():
    """Ensuring error is raised if method is not specified correctly
      for minimize_scalar option.
    """
    (point, projection, option, met, initial_guess, func_args, f, g, bound_1,
     bound_2, relax_sd_it) = func_params()
    option = 'minimize_scalar'
    met = 'Golden_v2'
    with pytest.raises(ValueError):
        mt_alg.sd_iteration(point, projection, option, met, initial_guess,
                            func_args, f, g, bound_1, bound_2, relax_sd_it)


def test_11():
    """Ensuring error is raised if option is not specified correctly"""
    (point, projection, option, met, initial_guess, func_args, f, g, bound_1,
     bound_2, relax_sd_it) = func_params()
    option = 'minimize_v1'
    with pytest.raises(ValueError):
        mt_alg.sd_iteration(point, projection, option, met, initial_guess,
                            func_args, f, g, bound_1, bound_2, relax_sd_it)
