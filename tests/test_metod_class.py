import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st

from metod_alg import check_metod_class as prev_mt_alg
from metod_alg import objective_functions as mt_obj
from metod_alg import metod_algorithm_functions as mt_alg


def func_params(d=20, p=2, lambda_1=1, lambda_2=10):
    """Generates parameters to use for tests."""
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    store_x0, matrix_test = (mt_obj.function_parameters_several_quad
                             (p, d, lambda_1, lambda_2))
    func_args = p, store_x0, matrix_test
    return f, g, func_args


def test_1():
    """Asserts error message when num_points is not integer."""
    d = 20
    f, g, func_args = func_params()
    num_points_t = 0.01
    with pytest.raises(ValueError):
        prev_mt_alg.metod_class(f, g, func_args, d, num_points=num_points_t)


def test_2():
    """Asserts error message when d is not integer."""
    d = 0.01
    p = 10
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    func_args = (p, np.random.uniform(0, 1, (p, )),
                 np.random.uniform(0, 1, (p, 10, 10)))
    with pytest.raises(ValueError):
        prev_mt_alg.metod_class(f, g, func_args, d)


def test_3():
    """Asserts error message when beta is not integer or float."""
    d = 20
    f, g, func_args = func_params()
    beta_t = True
    with pytest.raises(ValueError):
        prev_mt_alg.metod_class(f, g, func_args, d, beta=beta_t)


def test_4():
    """Asserts error message when tolerance is not float."""
    d = 20
    f, g, func_args = func_params()
    tolerance_t = True
    with pytest.raises(ValueError):
        prev_mt_alg.metod_class(f, g, func_args, d, tolerance=tolerance_t)


def test_5():
    """Asserts error message when projection is not boolean."""
    d = 20
    f, g, func_args = func_params()
    projection_t = 0.01
    with pytest.raises(ValueError):
        prev_mt_alg.metod_class(f, g, func_args, d, projection=projection_t)


def test_6():
    """Asserts error message when const is not integer or float."""
    d = 20
    f, g, func_args = func_params()
    const_t = 'test'
    with pytest.raises(ValueError):
        prev_mt_alg.metod_class(f, g, func_args, d, const=const_t)


def test_7():
    """Asserts error message when m is not integer."""
    d = 20
    f, g, func_args = func_params()
    m_t = 0.9
    with pytest.raises(ValueError):
        prev_mt_alg.metod_class(f, g, func_args, d, m=m_t)


def test_8():
    """Asserts error message when option is not a string."""
    d = 20
    f, g, func_args = func_params()
    option_t = True
    with pytest.raises(ValueError):
        prev_mt_alg.metod_class(f, g, func_args, d, option=option_t)


def test_9():
    """Asserts error message when met is not a string."""
    d = 20
    f, g, func_args = func_params()
    met_t = 0.1
    with pytest.raises(ValueError):
        prev_mt_alg.metod_class(f, g, func_args, d, met=met_t)


def test_10():
    """Asserts error message when initial_guess is not a integer or float."""
    d = 20
    f, g, func_args = func_params()
    initial_guess_t = '213'
    with pytest.raises(ValueError):
        prev_mt_alg.metod_class(f, g, func_args, d,
                                initial_guess=initial_guess_t)


def test_11():
    """Asserts error message when d < 2."""
    d = 1
    p = 10
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    func_args = (p, np.random.uniform(0, 1, (p, )),
                 np.random.uniform(0, 1, (p, 10, 10)))
    with pytest.raises(ValueError):
        prev_mt_alg.metod_class(f, g, func_args, d)


def test_12():
    """Asserts error message when m < 1."""
    d = 20
    f, g, func_args = func_params()
    m_t = 0
    with pytest.raises(ValueError):
        prev_mt_alg.metod_class(f, g, func_args, d, m=m_t)


def test_13():
    """
    Asserts error message when bounds_set_x does not contain an integer or
    float.
    """
    d = 20
    f, g, func_args = func_params()
    bounds_set_x_t = (True, 1)
    with pytest.raises(ValueError):
        prev_mt_alg.metod_class(f, g, func_args, d,
                                bounds_set_x=bounds_set_x_t)


def test_14():
    """
    Asserts error message when bounds_set_x does not contain an integer or
    float.
    """
    d = 20
    f, g, func_args = func_params()
    bounds_set_x_t = (0, 'False')
    with pytest.raises(ValueError):
        prev_mt_alg.metod_class(f, g, func_args, d,
                                bounds_set_x=bounds_set_x_t)


def test_15():
    """Asserts warning message when beta >= 1."""
    d = 20
    f, g, func_args = func_params()
    beta_t = 1
    with pytest.warns(RuntimeWarning):
        prev_mt_alg.metod_class(f, g, func_args, d, beta=beta_t)


def test_16():
    """Asserts warning message when tolerance > 0.1."""
    d = 20
    f, g, func_args = func_params()
    tolerance_t = 0.2
    with pytest.warns(RuntimeWarning):
        prev_mt_alg.metod_class(f, g, func_args, d, tolerance=tolerance_t)


def test_17():
    """
    Asserts error message when number of iterations is less than m.
    """
    np.random.seed(90)
    d = 2
    p = 2
    lambda_1 = 1
    lambda_2 = 3
    tolerance_t = 0.1
    m_t = 6
    f, g, func_args = func_params(d, p, lambda_1, lambda_2)
    with pytest.raises(ValueError):
        prev_mt_alg.metod_class(f, g, func_args, d,
                                tolerance=tolerance_t, m=m_t)


def test_18():
    """Asserts error message when len(bounds_set_x) > 2."""
    d = 20
    f, g, func_args = func_params()
    bounds_set_x_t = (0, 1, 2)
    with pytest.raises(ValueError):
        prev_mt_alg.metod_class(f, g, func_args, d,
                                bounds_set_x=bounds_set_x_t)


def test_19():
    """
    Asserts error message when relax_sd_it is not
    integer or float.
    """
    d = 20
    f, g, func_args = func_params()
    relax_sd_it_t = 'Test'
    with pytest.raises(ValueError):
        prev_mt_alg.metod_class(f, g, func_args, d,
                                relax_sd_it=relax_sd_it_t)


def test_20():
    """Asserts error message when relax_sd_it is less than zero."""
    d = 20
    f, g, func_args = func_params()
    relax_sd_it_t = -0.1
    with pytest.raises(ValueError):
        prev_mt_alg.metod_class(f, g, func_args, d,
                                relax_sd_it=relax_sd_it_t)


def test_21():
    """Asserts error message when set_x is not a valid choice."""
    d = 20
    set_x_t = 'random_unif'
    f, g, func_args = func_params()
    with pytest.raises(ValueError):
        prev_mt_alg.metod_class(f, g, func_args, d,
                                set_x=set_x_t)


def test_22():
    """Asserts error message when set_x is not a string."""
    num_points = 1000
    d = 20
    set_x_t = np.random.uniform(0, 1, (num_points, d))
    f, g, func_args = func_params()
    with pytest.raises(ValueError):
        prev_mt_alg.metod_class(f, g, func_args, d,
                                set_x=set_x_t)


@settings(max_examples=10, deadline=None)
@given(st.integers(2, 20), st.integers(0, 3), st.integers(2, 100))
def test_23(p, m, d):
    """
    Test m is being applied correctly in metod_class.py when computing
    distances.
    """
    np.random.seed(p)
    x = np.random.uniform(0, 1, (d, ))
    tolerance = 0.00001
    projection = False
    option = 'minimize_scalar'
    met = 'Brent'
    initial_guess = 0.005
    beta = 0.095
    matrix_test = np.zeros((p, d, d))
    store_x0 = np.random.uniform(0, 1, (p, d))
    diag_vals = np.zeros(d)
    diag_vals[:2] = np.array([1, 10])
    diag_vals[2:] = np.random.uniform(2, 9, (d - 2))
    matrix_test[0] = np.diag(diag_vals)
    diag_vals = np.zeros(d)
    diag_vals[:2] = np.array([1, 10])
    diag_vals[2:] = np.random.uniform(2, 9, (d - 2))
    matrix_test[1] = np.diag(diag_vals)
    func_args = p, store_x0, matrix_test
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    usage = 'metod_algorithm'
    relax_sd_it = 1
    bound_1 = 0
    bound_2 = 1
    (iterations_of_sd,
     its,
     store_grad) = (mt_alg.apply_sd_until_stopping_criteria
                    (x, d, projection, tolerance, option, met,
                     initial_guess, func_args, f, g, bound_1,
                     bound_2, usage, relax_sd_it, None))
    """METOD algorithm checks the below"""
    assume(its > m)
    sd_iterations_partner_points = (mt_alg.partner_point_each_sd
                                    (iterations_of_sd, beta,
                                     store_grad))
    test_x = np.random.uniform(0, 1, (d, ))
    original_shape = iterations_of_sd.shape[0]
    """Checking correct warm up applied when checking distances"""
    set_dist = mt_alg.distances(iterations_of_sd, test_x, m, d, 'All')
    assert(set_dist.shape == (original_shape - m,))
    assert(set_dist.shape == (its + 1 - m,))
    assert(sd_iterations_partner_points.shape[0] == iterations_of_sd.shape[0])


@settings(max_examples=10, deadline=None)
@given(st.integers(2, 20), st.integers(5, 100), st.integers(50, 1000))
def test_24(p, d, num_points_t):
    """
    Check ouputs of algorithm with minimum of several Quadratic forms
    function and gradient.
    """
    np.random.seed(p)
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_test = (mt_obj.function_parameters_several_quad
                             (p, d, lambda_1, lambda_2))
    func_args = p, store_x0, matrix_test
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    (discovered_minimizers,
     number_minimizers,
     func_vals_of_minimizers,
     excessive_descents,
     starting_points,
     no_grad_evals,
     classification_point,
     count_gr_2) = prev_mt_alg.metod_class(f, g, func_args, d,
                                           num_points=num_points_t)
    """Check outputs are as expected"""
    assert(len(discovered_minimizers) == number_minimizers)
    assert(number_minimizers == len(func_vals_of_minimizers))
    assert(np.unique(classification_point).shape[0] == number_minimizers)
    """Ensure that each region of attraction discovered is unique"""
    mt_obj.check_unique_minimizers(discovered_minimizers, number_minimizers,
                                   mt_obj.calc_minimizer_sev_quad, func_args)
    assert(no_grad_evals[0] > 4)
    assert(count_gr_2 >= 0)
    assert(np.where(no_grad_evals > 4)[0].shape[0] == excessive_descents
           + number_minimizers)
    """Ensure that starting points used are of correct form"""
    assert(np.array(starting_points).shape == (num_points_t, d))
    assert(excessive_descents == 0)
    for j in range(num_points_t):
        for i in range(j+1, num_points_t):
            assert(np.any(np.round(starting_points[j], 5) !=
                   np.round(starting_points[i], 5)))


@settings(max_examples=10, deadline=None)
@given(st.integers(2, 20), st.integers(5, 100), st.integers(50, 1000))
def test_25(p, d, num_points_t):
    """
    Check ouputs of algorithm with minimum of several Quadratic forms
    function and gradient with set_x = 'random'.
    """
    np.random.seed(p)
    lambda_1 = 1
    lambda_2 = 10
    set_x_t = 'random'
    store_x0, matrix_test = (mt_obj.function_parameters_several_quad
                             (p, d, lambda_1, lambda_2))
    func_args = p, store_x0, matrix_test
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    (discovered_minimizers,
     number_minimizers,
     func_vals_of_minimizers,
     excessive_descents,
     starting_points,
     no_grad_evals,
     classification_point,
     count_gr_2) = prev_mt_alg.metod_class(f, g, func_args, d,
                                           num_points=num_points_t,
                                           set_x=set_x_t)
    """Check outputs are as expected"""
    assert(len(discovered_minimizers) == number_minimizers)
    assert(number_minimizers == len(func_vals_of_minimizers))
    assert(np.unique(classification_point).shape[0] == number_minimizers)
    assert(no_grad_evals[0] > 4)
    assert(count_gr_2 >= 0)
    assert(np.where(no_grad_evals > 4)[0].shape[0] == excessive_descents
           + number_minimizers)
    """Ensure that each region of attraction discovered is unique"""
    mt_obj.check_unique_minimizers(discovered_minimizers, number_minimizers,
                                   mt_obj.calc_minimizer_sev_quad, func_args)

    """Ensure that starting points used are of correct form"""
    assert(np.array(starting_points).shape == (num_points_t, d))
    assert(excessive_descents == 0)
    for j in range(num_points_t):
        for i in range(j+1, num_points_t):
            assert(np.any(np.round(starting_points[j], 5) !=
                   np.round(starting_points[i], 5)))


def test_26():
    """
    Checks ouputs of algorithm with Sum of Gaussians function and
    gradient
    """
    np.random.seed(15)
    d = 20
    p = 10
    sigma_sq = 0.8
    lambda_1 = 1
    lambda_2 = 10
    matrix_test = np.zeros((p, d, d))
    store_x0, matrix_test, store_c = (mt_obj.function_parameters_sog
                                      (p, d, lambda_1, lambda_2))
    func_args = p, sigma_sq, store_x0, matrix_test, store_c
    f = mt_obj.sog_function
    g = mt_obj.sog_gradient
    (discovered_minimizers,
     number_minimizers,
     func_vals_of_minimizers,
     excessive_descents,
     starting_points,
     no_grad_evals,
     classification_point,
     count_gr_2) = prev_mt_alg.metod_class(f, g, func_args, d)
    """Check outputs are as expected"""
    assert(len(discovered_minimizers) == number_minimizers)
    assert(number_minimizers == len(func_vals_of_minimizers))
    assert(np.unique(classification_point).shape[0] == number_minimizers)
    assert(no_grad_evals[0] > 4)
    assert(count_gr_2 >= 0)
    assert(np.where(no_grad_evals > 4)[0].shape[0] == excessive_descents
           + number_minimizers)
    """Ensure that each region of attraction discovered is unique"""
    mt_obj.check_unique_minimizers(discovered_minimizers, number_minimizers,
                                   mt_obj.calc_minimizer_sog, func_args)

    """Ensure that starting points used are of correct form"""
    assert(np.array(starting_points).shape == (1000, d))
    assert(excessive_descents >= 0)
    for j in range(len(starting_points)):
        for i in range(j+1, len(starting_points)):
            assert(np.any(np.round(starting_points[j], 5) !=
                   np.round(starting_points[i], 5)))
