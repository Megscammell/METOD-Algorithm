import numpy as np
from hypothesis import given, settings, strategies as st

from metod_alg import metod_analysis as mt_ays
from metod_alg import objective_functions as mt_obj


@settings(max_examples=50, deadline=None)
@given(st.integers(20, 100), st.integers(5, 20), st.integers(11, 20))
def test_1(d, num_points, tolerance):
    """
    Ensuring outputs from compute_trajectories.py have expected
    properties
    """
    lambda_1 = 1
    lambda_2 = 10
    p = 2
    store_x0, matrix_test = (mt_obj.function_parameters_several_quad
                             (p, d, lambda_1, lambda_2))
    func_args = p, store_x0, matrix_test
    func_args_check_func = func_args
    option = 'minimize'
    met = 'Nelder-Mead'
    initial_guess = 0.05
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    check_func = mt_ays.calc_minimizer_sev_quad_no_dist_check
    projection = False
    bounds_1 = 0
    bounds_2 = 1
    usage = 'metod_analysis'
    relax_sd_it = 1
    (store_x_values_list,
     store_minimizer,
     counter_non_matchings,
     counter_matchings,
     store_grad_all) = (mt_ays.compute_trajectories
                        (num_points, d, projection, tolerance, option,
                         met, initial_guess, func_args, f, g, bounds_1,
                         bounds_2, usage, relax_sd_it, check_func,
                         func_args_check_func))
    mt_ays.check_sp_fp(store_x_values_list, num_points, func_args)
    assert(type(counter_non_matchings) is int or type(counter_non_matchings)
           is float)
    assert(type(counter_matchings) is int or type(counter_matchings)
           is float)
    assert(store_minimizer.shape == (num_points, ))
    assert(len(store_x_values_list) == num_points)
    for j in range(num_points):
        x_tr = store_x_values_list[j]
        grad = store_grad_all[j]
        assert(x_tr.shape == (tolerance + 1, d))
        assert(grad.shape == (tolerance + 1, d))
        for k in range(tolerance + 1):
            assert(np.all(grad[k] == g(x_tr[k], *func_args)))


@settings(max_examples=10, deadline=None)
@given(st.integers(20, 100), st.integers(5, 20), st.integers(11, 20))
def test_2(d, num_points, tolerance):
    """
    Ensuring outputs from compute_trajectories.py have expected
    properties
    """
    d = 10
    f = mt_obj.zakharov_func
    g = mt_obj.zakharov_grad
    check_func = None
    func_args = (d,)
    func_args_check_func = func_args
    bounds_1 = -5
    bounds_2 = 10

    option = 'minimize'
    met = 'Nelder-Mead'
    initial_guess = 0.05
    projection = False
    usage = 'metod_algorithm'
    tolerance = 0.00001
    relax_sd_it = 1
    (store_x_values_list,
     store_minimizer,
     counter_non_matchings,
     counter_matchings,
     store_grad_all) = (mt_ays.compute_trajectories
                        (num_points, d, projection, tolerance, option,
                         met, initial_guess, func_args, f, g, bounds_1,
                         bounds_2, usage, relax_sd_it, check_func,
                         func_args_check_func))
    assert(type(counter_non_matchings) is int or type(counter_non_matchings)
           is float)
    assert(type(counter_matchings) is int or type(counter_matchings)
           is float)
    assert(store_minimizer.shape == (num_points, ))
    assert(len(store_x_values_list) == num_points)
    for j in range(num_points):
        x_tr = store_x_values_list[j]
        grad = store_grad_all[j]
        for k in range(len(x_tr)):
            assert(np.all(grad[k] == g(x_tr[k], *func_args)))
