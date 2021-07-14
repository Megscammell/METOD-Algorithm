import numpy as np
from hypothesis import given, settings, strategies as st

from metod_alg import metod_analysis as mt_ays
from metod_alg import objective_functions as mt_obj
from metod_alg import metod_algorithm_functions as mt_alg


def test_1():
    """
    Checks that there are separate outputs for different values of beta
    for mt_ays.main_analysis_shekel().
    """
    f = mt_obj.shekel_function
    g = mt_obj.shekel_gradient
    check_func = mt_obj.calc_minimizer_shekel
    d = 4
    p = 10
    lambda_1 = 1
    lambda_2 = 10
    num_points = 30
    num = 1
    projection = False
    option = 'minimize_scalar'
    met = 'Brent'
    initial_guess = 0.005
    relax_sd_it = 1
    bounds_1 = 0
    bounds_2 = 10
    tolerance = 0.0001
    number_its_compare = 3
    test_beta = [0.01, 0.1]
    usage = 'metod_algorithm'
    num_functions = 3

    total_count_nsm_b_01 = np.zeros((number_its_compare - num, number_its_compare - num))
    total_total_nsm_b_01 = np.zeros((number_its_compare - num, number_its_compare - num))
    total_count_nsm_b_1 = np.zeros((number_its_compare - num, number_its_compare - num))
    total_total_nsm_b_1 = np.zeros((number_its_compare - num, number_its_compare - num))
    for k in range(num_functions):
        np.random.seed(k + 1)
        matrix_test, C, b = (mt_obj.function_parameters_shekel
                             (lambda_1, lambda_2, p))
        func_args = (p, matrix_test, C, b)
        func_args_check_func = func_args
        (store_x_values,
         store_minimizer,
         counter_non_matchings,
         counter_matchings,
         store_grad_all) = (mt_ays.compute_trajectories
                            (num_points, d, projection, tolerance, option,
                             met, initial_guess, func_args, f, g, bounds_1,
                             bounds_2, usage, relax_sd_it, check_func,
                             func_args_check_func))

        beta = 0.01
        store_z_values = []
        for j in range(num_points):
            points_x = store_x_values[j]
            grad_x = store_grad_all[j]
            points_z = mt_alg.partner_point_each_sd(points_x, beta,
                                                    grad_x)
            store_z_values.append(points_z)
        (count_sm_b_01,
         comparisons_sm_b_01,
         total_sm_b_01,
         count_nsm_b_01,
         comparisons_nsm_b_01,
         total_nsm_b_01,
         calc_b_nsm_match_calc_nsm_b_01,
         calc_b_pos_nsm_b_01) = (mt_ays.all_comparisons_matches_both
                                 (d, store_x_values, store_z_values,
                                  num_points, store_minimizer, num, beta,
                                  counter_non_matchings, number_its_compare,
                                  g, func_args))
        total_count_nsm_b_01 += count_nsm_b_01
        total_total_nsm_b_01 += total_nsm_b_01

        beta = 0.1
        store_z_values = []
        for j in range(num_points):
            points_x = store_x_values[j]
            grad_x = store_grad_all[j]
            points_z = mt_alg.partner_point_each_sd(points_x, beta,
                                                    grad_x)
            store_z_values.append(points_z)

        (count_sm_b_1, comparisons_sm_b_1,
         total_sm_b_1, count_nsm_b_1,
         comparisons_nsm_b_1, total_nsm_b_1,
         calc_b_nsm_match_calc_nsm_b_1,
         calc_b_pos_nsm_b_1) = (mt_ays.all_comparisons_matches_both
                                (d, store_x_values, store_z_values,
                                 num_points, store_minimizer, num, beta,
                                 counter_non_matchings, number_its_compare,
                                 g, func_args))
        total_count_nsm_b_1 += count_nsm_b_1
        total_total_nsm_b_1 += total_nsm_b_1
    (fails_nsm_total, checks_nsm_total,
     fails_sm_total, checks_sm_total,
     max_b_calc_func_val_nsm,
     store_all_its,
     store_all_norm_grad) = (mt_ays.main_analysis_shekel
                            (d, test_beta, num_functions, num_points, p,
                            lambda_1, lambda_2, projection, tolerance, option, met,
                            initial_guess, bounds_1, bounds_2, usage, relax_sd_it,
                            num, number_its_compare))
    assert(np.all(fails_nsm_total[0] == total_count_nsm_b_01))
    assert(np.all(checks_nsm_total[0] == total_total_nsm_b_01))
    assert(np.all(fails_nsm_total[1] == total_count_nsm_b_1))
    assert(np.all(checks_nsm_total[1] == total_total_nsm_b_1))
    assert(store_all_its.shape == (num_functions, num_points))
    assert(store_all_norm_grad.shape == (num_functions, num_points))


def test_2():
    """
    Ensuring outputs of mt_ays.main_analysis_shekel() have expected properties.
    """
    test_beta = [0.001, 0.01, 0.1]
    num_functions = 100
    num_points = 100
    f = mt_obj.shekel_function
    g = mt_obj.shekel_gradient
    check_func = mt_obj.function_parameters_shekel
    d = 4
    p = 10
    lambda_1 = 1
    lambda_2 = 10
    projection = False
    tolerance = 0.0001
    option = 'minimize_scalar'
    met = 'Brent'
    initial_guess = 0.005
    usage = 'metod_algorithm'
    bounds_1 = 0
    bounds_2 = 10
    relax_sd_it = 1
    number_its_compare = 4
    num = 1
    num_functions = 3
    (fails_nsm_total, checks_nsm_total,
     fails_sm_total, checks_sm_total,
     max_b_calc_func_val_nsm,
     store_all_its,
     store_all_norm_grad) = (mt_ays.main_analysis_shekel
                            (d, test_beta, num_functions, num_points, p,
                            lambda_1, lambda_2, projection, tolerance, option, met,
                            initial_guess, bounds_1, bounds_2, usage, relax_sd_it,
                            num, number_its_compare))
    assert(fails_nsm_total.shape == (len(test_beta), number_its_compare - num,
                                     number_its_compare - num))
    assert(fails_sm_total.shape == (len(test_beta), number_its_compare - num,
                                    number_its_compare - num))
    assert(checks_nsm_total.shape == (len(test_beta), number_its_compare - num,
                                      number_its_compare - num))
    assert(checks_sm_total.shape == (len(test_beta), number_its_compare - num,
                                     number_its_compare - num))
    assert(max_b_calc_func_val_nsm.shape == (len(test_beta), num_functions))
    assert(store_all_its.shape == (num_functions, num_points))
    assert(store_all_norm_grad.shape == (num_functions, num_points))


def test_3():
    """
    Ensuring outputs of mt_ays.main_analysis_quad() have expected properties.
    """
    d = 4
    p = 10
    lambda_1 = 1
    lambda_2 = 10
    f = mt_obj.shekel_function
    g = mt_obj.shekel_gradient
    check_func = mt_obj.calc_minimizer_shekel
    matrix_test, C, b = (mt_obj.function_parameters_shekel
                             (lambda_1, lambda_2, p))
    func_args = (p, matrix_test, C, b)
    projection = False
    option = 'minimize_scalar'
    met = 'Brent'
    initial_guess = 0.005
    relax_sd_it = 1
    bounds_1 = 0
    bounds_2 = 10
    tolerance = 0.00001
    num_points = 20
    usage = 'metod_algorithm'
    for i in range((num_points)):
        x = np.random.uniform(bounds_1, bounds_2, (d, ))
        points_x, its, grad = (mt_alg.apply_sd_until_stopping_criteria
                              (x, d, projection, tolerance, option,
                               met, initial_guess, func_args, f, g,
                               bounds_1, bounds_2, usage, relax_sd_it,
                               None))
        assert(len(points_x)-1 == its)