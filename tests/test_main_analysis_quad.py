import numpy as np
from hypothesis import given, settings, strategies as st

from metod_alg import metod_analysis as mt_ays
from metod_alg import objective_functions as mt_obj
from metod_alg import metod_algorithm_functions as mt_alg


def test_1():
    """Checks that there are separate outputs for different values of beta."""
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    d = 100
    num_points = 30
    p = 2
    lambda_1 = 1
    lambda_2 = 10
    num = 1
    projection = False
    option = 'minimize'
    met = 'Nelder-Mead'
    initial_guess = 0.05
    relax_sd_it = 1
    bounds_1 = 0
    bounds_2 = 1
    tolerance = 15
    test_beta = [0.01, 0.1]
    usage = 'metod_analysis'
    num_functions = 3
    total_count_nsm_b_01 = np.zeros((tolerance - num, tolerance - num))
    total_total_nsm_b_01 = np.zeros((tolerance - num, tolerance - num))
    total_count_nsm_b_1 = np.zeros((tolerance - num, tolerance - num))
    total_total_nsm_b_1 = np.zeros((tolerance - num, tolerance - num))
    for k in range(num_functions):
        np.random.seed(k + 1)
        store_x0, matrix_test = (mt_obj.function_parameters_several_quad
                                 (p, d, lambda_1, lambda_2))
        func_args = p, store_x0, matrix_test
        (store_x_values,
         store_minimizer,
         counter_non_matchings,
         counter_matchings) = (mt_ays.compute_trajectories
                               (num_points, d, projection, tolerance, option,
                                met, initial_guess, func_args, f, g, bounds_1,
                                bounds_2, usage, relax_sd_it))

        beta = 0.01
        store_z_values = []
        for j in range(num_points):
            points_x = store_x_values[j]
            points_z = mt_alg.partner_point_each_sd(points_x, d, beta,
                                                    tolerance, g, func_args)
            store_z_values.append(points_z)
        (count_sm_b_01, comparisons_sm_b_01, total_sm_b_01, count_nsm_b_01,
         comparisons_nsm_b_01, total_nsm_b_01, calc_b_nsm_match_calc_nsm_b_01,
         calc_b_pos_nsm_b_01) = (mt_ays.all_comparisons_matches_both
                                 (d, store_x_values, store_z_values,
                                  num_points, store_minimizer, num, beta,
                                  counter_non_matchings, tolerance, func_args))
        total_count_nsm_b_01 += count_nsm_b_01
        total_total_nsm_b_01 += total_nsm_b_01

        beta = 0.1
        store_z_values = []
        for j in range(num_points):
            points_x = store_x_values[j]
            points_z = mt_alg.partner_point_each_sd(points_x, d, beta,
                                                    tolerance, g, func_args)
            store_z_values.append(points_z)

        (count_sm_b_1, comparisons_sm_b_1,
         total_sm_b_1, count_nsm_b_1,
         comparisons_nsm_b_1, total_nsm_b_1,
         calc_b_nsm_match_calc_nsm_b_1,
         calc_b_pos_nsm_b_1) = (mt_ays.all_comparisons_matches_both
                                (d, store_x_values, store_z_values,
                                 num_points, store_minimizer, num, beta,
                                 counter_non_matchings, tolerance, func_args))
        total_count_nsm_b_1 += count_nsm_b_1
        total_total_nsm_b_1 += total_nsm_b_1
    (fails_nsm_total, checks_nsm_total,
     fails_sm_total, checks_sm_total,
     max_b_calc_func_val_nsm) = (mt_ays.main_analysis_quad
                                 (d, f, g, test_beta, num_functions,
                                  num_points, p, lambda_1, lambda_2,
                                  projection, tolerance, option, met,
                                  initial_guess, bounds_1, bounds_2, usage,
                                  relax_sd_it, num))
    assert(np.all(fails_nsm_total[0] == total_count_nsm_b_01))
    assert(np.all(checks_nsm_total[0] == total_total_nsm_b_01))
    assert(np.all(fails_nsm_total[1] == total_count_nsm_b_1))
    assert(np.all(checks_nsm_total[1] == total_total_nsm_b_1))


def test_2():
    """
    Ensuring outputs of main_analysis_quad.py have expected properties.
    """
    d = 10
    tolerance = 12
    lambda_1 = 1
    lambda_2 = 10
    p = 2
    test_beta = [0.1]
    option = 'minimize'
    met = 'Nelder-Mead'
    initial_guess = 0.05
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    projection = False
    bounds_1 = 0
    bounds_2 = 1
    num_functions = 1
    num_points = 2
    usage = 'metod_analysis'
    relax_sd_it = 1
    num = 1
    (fails_nsm_total, checks_nsm_total,
     fails_sm_total, checks_sm_total,
     max_b_calc_func_val_nsm) = (mt_ays.main_analysis_quad
                                 (d, f, g, test_beta, num_functions,
                                  num_points, p, lambda_1, lambda_2,
                                  projection, tolerance, option, met,
                                  initial_guess, bounds_1, bounds_2, usage,
                                  relax_sd_it, num))
    assert(fails_nsm_total.shape == (len(test_beta), tolerance - num,
                                     tolerance - num))
    assert(fails_sm_total.shape == (len(test_beta), tolerance - num,
                                    tolerance - num))
    assert(checks_nsm_total.shape == (len(test_beta), tolerance - num,
                                      tolerance - num))
    assert(checks_sm_total.shape == (len(test_beta), tolerance - num,
                                     tolerance - num))
    assert(max_b_calc_func_val_nsm.shape == (len(test_beta), num_functions))
    assert(np.all(max_b_calc_func_val_nsm == np.zeros((len(test_beta),
                                                       num_functions))))


@settings(max_examples=10, deadline=None)
@given(st.integers(20, 100), st.integers(5, 20), st.integers(11, 20),
       st.integers(0, 10))
def test_3(d, num_points, tolerance, num):
    """
    Ensuring outputs of main_analysis_quad.py have expected properties.
    """
    lambda_1 = 1
    lambda_2 = 10
    p = 2
    test_beta = [0.001, 0.01, 0.1]
    option = 'minimize'
    met = 'Nelder-Mead'
    initial_guess = 0.05
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    projection = False
    bounds_1 = 0
    bounds_2 = 1
    num_functions = 10
    usage = 'metod_analysis'
    relax_sd_it = 1
    (fails_nsm_total, checks_nsm_total,
     fails_sm_total, checks_sm_total,
     max_b_calc_func_val_nsm) = (mt_ays.main_analysis_quad
                                 (d, f, g, test_beta, num_functions,
                                  num_points, p, lambda_1, lambda_2,
                                  projection, tolerance, option, met,
                                  initial_guess, bounds_1, bounds_2, usage,
                                  relax_sd_it, num))
    assert(fails_nsm_total.shape == (len(test_beta), tolerance - num,
                                     tolerance - num))
    assert(fails_sm_total.shape == (len(test_beta), tolerance - num,
                                    tolerance - num))
    assert(checks_nsm_total.shape == (len(test_beta), tolerance - num,
                                      tolerance - num))
    assert(checks_sm_total.shape == (len(test_beta), tolerance - num,
                                     tolerance - num))
    assert(max_b_calc_func_val_nsm.shape == (len(test_beta), num_functions))
