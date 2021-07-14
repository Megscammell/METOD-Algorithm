import numpy as np
from hypothesis import given, settings, strategies as st

from metod_alg import metod_analysis as mt_ays
from metod_alg import objective_functions as mt_obj
from metod_alg import metod_algorithm_functions as mt_alg


@settings(max_examples=50, deadline=None)
@given(st.integers(20, 100), st.floats(0.0001, 0.1))
def test_1(d, beta):
    """
    Test that outputs from mt_ays.evaluate_quantities_with_points_quad()
    are the same as mt_ays.check_quantities() for different values of d and beta.
    """
    p = 2
    lambda_1 = 1
    lambda_2 = 10
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    projection = False
    tolerance = 2
    option = 'minimize'
    met = 'Nelder-Mead'
    initial_guess = 0.05
    bound_1 = 0
    bound_2 = 1
    usage = 'metod_analysis'
    relax_sd_it = 1
    store_x0, matrix_test = (mt_obj.function_parameters_several_quad(p, d, 
                             lambda_1, lambda_2))
    func_args = p, store_x0, matrix_test

    x = np.random.uniform(0, 1, (d, ))
    y = np.random.uniform(0, 1, (d, ))
    while (mt_ays.calc_minimizer_sev_quad_no_dist_check(x, *func_args) ==
           mt_ays.calc_minimizer_sev_quad_no_dist_check(y, *func_args)):
        x = np.random.uniform(0, 1, (d, ))
        y = np.random.uniform(0, 1, (d, ))

    (x_tr,
     its_x,
     store_grad_x) = (mt_alg.apply_sd_until_stopping_criteria
                      (x, d, projection, tolerance, option, met, initial_guess,
                       func_args, f, g, bound_1, bound_2, usage, relax_sd_it,
                       None))
    assert(its_x == tolerance)
    assert(store_grad_x.shape == (tolerance + 1, d))
    (y_tr,
     its_y,
     store_grad_y) = (mt_alg.apply_sd_until_stopping_criteria
                      (y, d, projection, tolerance, option, met, initial_guess,
                       func_args, f, g, bound_1, bound_2, usage, relax_sd_it,
                       None))
    assert(its_y == tolerance)
    assert(store_grad_y.shape == (tolerance + 1, d))
    min_x = int(mt_ays.calc_minimizer_sev_quad_no_dist_check(x, *func_args))
    min_y = int(mt_ays.calc_minimizer_sev_quad_no_dist_check(y, *func_args))
    quantities_array, sum_quantities = (mt_ays.evaluate_quantities_with_points_quad
                                        (beta, x_tr, y_tr, min_x, min_y, d,
                                         g, func_args))

    assert(np.round(sum_quantities[0], 5) == np.round(mt_ays.check_quantities
                                                      (beta, x_tr[1, :], y_tr
                                                       [1, :], g, func_args), 5))
    assert(np.round(sum_quantities[1], 5) == np.round(mt_ays.check_quantities
                                                      (beta, x_tr[1, :], y_tr
                                                       [2, :], g, func_args), 5))
    assert(np.round(sum_quantities[2], 5) == np.round(mt_ays.check_quantities
                                                      (beta, x_tr[2, :], y_tr
                                                       [1, :], g, func_args), 5))
    assert(np.round(sum_quantities[3], 5) == np.round(mt_ays.check_quantities
                                                      (beta, x_tr[2, :], y_tr
                                                       [2, :], g, func_args), 5))



@settings(max_examples=5, deadline=None)
@given( st.floats(0.0001, 0.1))
def test_2(beta):
    """
    Test outputs from mt_ays.evaluate_quantities_with_points() have
    expected form.
    """
    p = 10
    d = 20
    sigma_sq = 0.7
    lambda_1 = 1
    lambda_2 = 10
    f = mt_obj.sog_function
    g = mt_obj.sog_gradient
    check_func = mt_obj.calc_minimizer_sog
    usage = 'metod_algorithm'
    tolerance = 0.000001
    projection = False
    option = 'minimize'
    met = 'Nelder-Mead'
    initial_guess = 0.005
    bound_1 = 0
    bound_2 = 1
    relax_sd_it = 1
    store_x0, matrix_test, store_c = (mt_obj.function_parameters_sog
                                      (p, d, lambda_1, lambda_2))
    func_args = (p, sigma_sq, store_x0, matrix_test, store_c)

    x = np.random.uniform(0, 1, (d, ))
    (x_tr,
     its_x,
     store_grad_x) = (mt_alg.apply_sd_until_stopping_criteria
                      (x, d, projection, tolerance, option, met, initial_guess,
                       func_args, f, g, bound_1, bound_2, usage, relax_sd_it,
                       None))
    
    y = np.random.uniform(0, 1, (d, ))
    (y_tr,
     its_y,
     store_grad_y) = (mt_alg.apply_sd_until_stopping_criteria
                      (y, d, projection, tolerance, option, met, initial_guess,
                       func_args, f, g, bound_1, bound_2, usage, relax_sd_it,
                       None))
    while (check_func(x_tr[-1], *func_args) ==
           check_func(y_tr[-1], *func_args)):
        x = np.random.uniform(0, 1, (d, ))
        (x_tr,
        its_x,
        store_grad_x) = (mt_alg.apply_sd_until_stopping_criteria
                        (x, d, projection, tolerance, option, met, initial_guess,
                        func_args, f, g, bound_1, bound_2, usage, relax_sd_it,
                        None))
        y = np.random.uniform(0, 1, (d, ))
        (y_tr,
        its_y,
        store_grad_y) = (mt_alg.apply_sd_until_stopping_criteria
                        (y, d, projection, tolerance, option, met, initial_guess,
                        func_args, f, g, bound_1, bound_2, usage, relax_sd_it,
                        None))

    min_x = int(check_func(x_tr[-1], *func_args))
    min_y = int(check_func(y_tr[-1], *func_args))
    assert(min_x != min_y)
    quantities_array, sum_quantities = (mt_ays.evaluate_quantities_with_points
                                        (beta, x_tr, y_tr, d,
                                         g, func_args))

