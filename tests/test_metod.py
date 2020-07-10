import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st

import metod_testing as mtv3


def func_params(d=20, p=2, lambda_1=1, lambda_2=10):
    """Generates parameters to use for tests 1 - 20"""
    f = mtv3.quad_function
    g = mtv3.quad_gradient
    store_x0, matrix_test = mtv3.function_parameters_quad(p, d, lambda_1,
                                                          lambda_2)
    func_args = p, store_x0, matrix_test
    return f, g, func_args


def test_1():
    """Asserts error message when num_points is not integer"""
    d = 20
    f, g, func_args = func_params()
    num_points_t = 0.01
    with pytest.raises(ValueError):
        mtv3.metod(f, g, func_args, d, num_points=num_points_t)


def test_2():
    """Asserts error message when d is not integer"""
    d = 0.01
    p = 10
    f = mtv3.quad_function
    g = mtv3.quad_gradient
    func_args = (p, np.random.uniform(0, 1, (p, )),
                 np.random.uniform(0, 1, (p, 10, 10)))
    with pytest.raises(ValueError):
        mtv3.metod(f, g, func_args, d)


def test_3():
    """Asserts error message when beta is not integer or float"""
    d = 20
    f, g, func_args = func_params()
    beta_t = True
    with pytest.raises(ValueError):
        mtv3.metod(f, g, func_args, d, beta=beta_t)


def test_4():
    """Asserts error message when tolerance is not float"""
    d = 20
    f, g, func_args = func_params()
    tolerance_t = True
    with pytest.raises(ValueError):
        mtv3.metod(f, g, func_args, d, tolerance=tolerance_t)


def test_5():
    """Asserts error message when projection is not boolean"""
    d = 20
    f, g, func_args = func_params()
    projection_t = 0.01
    with pytest.raises(ValueError):
        mtv3.metod(f, g, func_args, d, projection=projection_t)


def test_6():
    """Asserts error message when const is not integer or float"""
    d = 20
    f, g, func_args = func_params()
    const_t = 'test'
    with pytest.raises(ValueError):
        mtv3.metod(f, g, func_args, d, const=const_t)


def test_7():
    """Asserts error message when m is not integer"""
    d = 20
    f, g, func_args = func_params()
    m_t = 0.9
    with pytest.raises(ValueError):
        mtv3.metod(f, g, func_args, d, m=m_t)


def test_8():
    """Asserts error message when option is not a string"""
    d = 20
    f, g, func_args = func_params()
    option_t = True
    with pytest.raises(ValueError):
        mtv3.metod(f, g, func_args, d, option=option_t)


def test_9():
    """Asserts error message when met is not a string"""
    d = 20
    f, g, func_args = func_params()
    met_t = 0.1
    with pytest.raises(ValueError):
        mtv3.metod(f, g, func_args, d, met=met_t)


def test_10():
    """Asserts error message when initial_guess is not a integer or float"""
    d = 20
    f, g, func_args = func_params()
    initial_guess_t = '213'
    with pytest.raises(ValueError):
        mtv3.metod(f, g, func_args, d,
                   initial_guess=initial_guess_t)


def test_11():
    """Asserts error message when d < 2"""
    d = 1
    p = 10
    f = mtv3.quad_function
    g = mtv3.quad_gradient
    func_args = (p, np.random.uniform(0, 1, (p, )),
                 np.random.uniform(0, 1, (p, 10, 10)))
    with pytest.raises(ValueError):
        mtv3.metod(f, g, func_args, d)


def test_12():
    """Asserts error message when m < 1"""
    d = 20
    f, g, func_args = func_params()
    m_t = 0
    with pytest.raises(ValueError):
        mtv3.metod(f, g, func_args, d, m=m_t)


def test_13():
    """Asserts error message when dimension of set_x not the same as d"""
    d = 20
    f, g, func_args = func_params()
    set_x_test = np.random.uniform(0, 1, (50, 10))
    with pytest.raises(ValueError):
        mtv3.metod(f, g, func_args, d, set_x=set_x_test)


def test_14():
    """
    Asserts error message when bounds_set_x do not contain integers or
    floats
    """
    d = 20
    f, g, func_args = func_params()
    bounds_set_x_t = (True, 1)
    with pytest.raises(ValueError):
        mtv3.metod(f, g, func_args, d,
                   bounds_set_x=bounds_set_x_t)


def test_15():
    """
    Asserts error message when bounds_set_x do not contain integers or
    floats
    """
    d = 20
    f, g, func_args = func_params()
    bounds_set_x_t = (0, 'False')
    with pytest.raises(ValueError):
        mtv3.metod(f, g, func_args, d,
                   bounds_set_x=bounds_set_x_t)


def test_16():
    """Asserts warning message when beta >= 1"""
    d = 20
    f, g, func_args = func_params()
    beta_t = 1
    with pytest.warns(RuntimeWarning):
        mtv3.metod(f, g, func_args, d, beta=beta_t)


def test_17():
    """Asserts warning message when tolerance > 0.1"""
    d = 20
    f, g, func_args = func_params()
    tolerance_t = 0.2
    with pytest.warns(RuntimeWarning):
        mtv3.metod(f, g, func_args, d, tolerance=tolerance_t)


def test_18():
    """
    Asserts error message when set_x does not contain points of same d
    """
    d = 20
    f, g, func_args = func_params()
    set_x_test = []
    set_x_test.append(np.random.uniform(0, 1, (20, )))
    set_x_test.append(np.random.uniform(0, 1, (20, )))
    set_x_test.append(np.random.uniform(0, 1, (18, )))
    with pytest.raises(ValueError):
        mtv3.metod(f, g, func_args, d, set_x=set_x_test)


def test_19():
    """
    Asserts error message when number of iterations is less than m
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
        mtv3.metod(f, g, func_args, d, tolerance=tolerance_t, m=m_t)


def test_20():
    """Asserts error message when len(bounds_set_x) > 2"""
    d = 20
    f, g, func_args = func_params()
    bounds_set_x_t = (0, 1, 2)
    with pytest.raises(ValueError):
        mtv3.metod(f, g, func_args, d,
                   bounds_set_x=bounds_set_x_t)


@settings(max_examples=10, deadline=None)
@given(st.integers(2, 20), st.integers(0, 3), st.integers(2, 100))
def test_21(p, m, d):
    """ Test m is being applied correctly in metod.py when computing
     distances """
    np.random.seed(p)
    x = np.random.uniform(0, 1, (d, ))
    tolerance = 0.00001
    projection = False
    initial_guess = 0.05
    option = 'minimize'
    met = 'Nelder-Mead'
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
    f = mtv3.quad_function
    g = mtv3.quad_gradient
    iterations_of_sd, its = (mtv3.apply_sd_until_stopping_criteria
                             (x, d, projection, tolerance, option, met,
                              initial_guess, func_args, f, g, bound_1=0,
                              bound_2=1))
    """METOD algorithm checks the below"""
    assume(its > m)
    sd_iterations_partner_points = (mtv3.partner_point_each_sd
                                    (iterations_of_sd, d, beta, its, g,
                                     func_args))
    test_x = np.random.uniform(0, 1, (d, ))
    original_shape = iterations_of_sd.shape[0]
    """Checking correct warm up applied when checking distances"""
    set_dist = mtv3.distances(iterations_of_sd, test_x, m, d)
    assert(set_dist.shape == (original_shape - m,))
    assert(set_dist.shape == (its + 1 - m,))
    assert(sd_iterations_partner_points.shape[0] == iterations_of_sd.shape[0])


@settings(max_examples=10, deadline=None)
@given(st.integers(2, 20), st.integers(5, 100), st.integers(50, 1000))
def test_22(p, d, num_points_t):
    """Check ouputs of algorithm with minimum of several Quadratic forms
     function and gradient """
    np.random.seed(p)
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_test = mtv3.function_parameters_quad(p, d, lambda_1,
                                                          lambda_2)
    func_args = p, store_x0, matrix_test
    f = mtv3.quad_function
    g = mtv3.quad_gradient
    (discovered_minimas, number_minimas, func_vals_of_minimas,
     number_excessive_descents) = mtv3.metod(f, g, func_args, d,
                                             num_points=num_points_t)
    """Check outputs are as expected"""
    assert(len(discovered_minimas) == number_minimas)
    assert(number_minimas == len(func_vals_of_minimas))
    norms_with_minima = np.zeros((number_minimas))
    pos_list = np.zeros((number_minimas))
    for j in range(number_minimas):
        pos, norm_minima = mtv3.calc_pos(discovered_minimas[j].reshape(d, ),
                                         *func_args)
        pos_list[j] = pos
        norms_with_minima[j] = norm_minima
    """Ensures discovered minima is very close to actual minima"""
    assert(np.max(norms_with_minima) < 0.0001)
    """Ensure that each region of attraction discovered is unique"""
    assert(np.unique(pos_list).shape[0] == number_minimas)


def test_23():
    """Checks ouputs of algorithm with Sum of Gaussians function and
     gradient"""
    np.random.seed(11)
    d = 100
    p = 10
    sigma_sq = 2
    lambda_1 = 1
    lambda_2 = 10
    matrix_test = np.zeros((p, d, d))
    store_x0, matrix_test, store_c = (mtv3.function_parameters_sog
                                      (p, d, lambda_1, lambda_2))
    args = p, sigma_sq, store_x0, matrix_test, store_c
    f = mtv3.sog_function
    g = mtv3.sog_gradient
    (discovered_minimas, number_minimas, func_vals_of_minimas,
     number_excessive_descents) = mtv3.metod(f, g, args, d)
    """Check outputs are as expected"""
    assert(len(discovered_minimas) == number_minimas)
    assert(number_minimas == len(func_vals_of_minimas))
    norms_with_minima = np.zeros((number_minimas))
    pos_list = np.zeros((number_minimas))
    for j in range(number_minimas):
        pos, min_dist = mtv3.calc_minima(discovered_minimas[j], *args)
        pos_list[j] = pos
        norms_with_minima[j] = min_dist
    """Ensures discovered minima is very close to actual minima"""
    assert(np.max(norms_with_minima) < 0.0001)
    """Ensure that each region of attraction discovered is unique"""
    assert(np.unique(pos_list).shape[0] == number_minimas)


@settings(max_examples=10, deadline=None)
@given(st.integers(2, 20), st.integers(1, 5), st.integers(2, 100))
def test_24(p, m, d):
    """Check that continued iterations from x_2 to a minimizer,
    (iterations_of_sd_part), joined with the initial warm up points
    (warm_up_sd), has the same points and shape compared to when
    initial point x has steepest descent iterations applied.
    """
    beta = 0.099
    tolerance = 0.00001
    initial_guess = 0.05
    projection = False
    lambda_1 = 1
    lambda_2 = 10
    option = 'minimize'
    met = 'Nelder-Mead'
    f = mtv3.quad_function
    g = mtv3.quad_gradient
    """Create objective function parameters"""
    store_x0, matrix_test = mtv3.function_parameters_quad(p, d, lambda_1,
                                                          lambda_2)
    func_args = p, store_x0, matrix_test
    """Generate random starting point"""
    bound_1 = 0
    bound_2 = 1
    x = np.random.uniform(bound_1, bound_2, (d, ))
    warm_up_sd, warm_up_sd_partner_points = (mtv3.apply_sd_until_warm_up
                                             (x, d, m, beta, projection,
                                              option, met, initial_guess,
                                              func_args, f, g, bound_1,
                                              bound_2))
    x_2 = warm_up_sd[m].reshape(d, )
    iterations_of_sd_part, its = (mtv3.apply_sd_until_stopping_criteria
                                  (x_2, d, projection, tolerance, option, met,
                                   initial_guess, func_args, f, g, bound_1,
                                   bound_2))
    iterations_of_sd = np.vstack([warm_up_sd, iterations_of_sd_part[1:, ]]
                                 )
    sd_iterations_partner_points = (mtv3.partner_point_each_sd
                                    (iterations_of_sd, d, beta, its + m, g,
                                     func_args))
    iterations_of_sd_test, its_test = (mtv3.apply_sd_until_stopping_criteria
                                       (x, d, projection, tolerance, option,
                                        met, initial_guess, func_args, f, g,
                                        bound_1, bound_2))
    sd_iterations_partner_points_test = (mtv3.partner_point_each_sd
                                         (iterations_of_sd_test, d, beta,
                                          its_test, g, func_args))

    assert(np.all(np.round(iterations_of_sd_test, 4) == np.round
           (iterations_of_sd, 4)))

    assert(np.all(np.round(sd_iterations_partner_points_test, 4) == np.round
           (sd_iterations_partner_points, 4)))

    assert(iterations_of_sd_test.shape[0] == iterations_of_sd.shape[0])

    assert(sd_iterations_partner_points_test.shape[0] ==
           sd_iterations_partner_points.shape[0])

    assert(its_test == its + m)
