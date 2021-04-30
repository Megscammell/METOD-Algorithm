import numpy as np


import metod_alg as mt
from metod_alg import objective_functions as mt_obj
from metod_alg import metod_algorithm_functions as mt_alg


def test_1():
    """
    Check ouputs of multistart with minimum of several Quadratic forms
    function and gradient.
    """
    np.random.seed(20)
    lambda_1 = 1
    lambda_2 = 10
    p = 10
    d = 20
    num_points = 100
    store_x0, matrix_test = (mt_obj.function_parameters_several_quad
                             (p, d, lambda_1, lambda_2))
    func_args = p, store_x0, matrix_test
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    tolerance = 0.00001
    projection = False
    starting_points = np.random.uniform(0, 1, (num_points, d))
    const = 0.1
    met = 'None'
    option = 'forward_backward_tracking'
    initial_guess = 0.005
    bounds_set_x = (0, 1)
    relax_sd_it = 1
    (unique_minimizers_mult,
     unique_number_of_minimizers_mult,
     store_func_vals_mult,
     time_taken_des,
     store_minimizer_des,
     no_its) = mt.multistart(f, g, func_args, d, starting_points,
                                          num_points, tolerance, projection,
                                          const, option, met, initial_guess,
                                          bounds_set_x, relax_sd_it)
    """Check outputs are as expected"""
    assert(len(unique_minimizers_mult) == unique_number_of_minimizers_mult)
    assert(unique_number_of_minimizers_mult == len(store_func_vals_mult))

    """Ensure that each region of attraction discovered is unique"""
    mt_obj.check_unique_minimizers(store_minimizer_des,
                                   unique_number_of_minimizers_mult,
                                   mt_obj.calc_minimizer_sev_quad, func_args)
    assert(time_taken_des >= 0)
    assert(np.all(no_its > 0))


def test_2():
    """
    Checks ouputs of multistart with Sum of Gaussians function and
    gradient.
    """
    np.random.seed(11)
    d = 20
    p = 10
    num_points = 100
    sigma_sq = 0.8
    lambda_1 = 1
    lambda_2 = 10
    matrix_test = np.zeros((p, d, d))
    store_x0, matrix_test, store_c = (mt_obj.function_parameters_sog
                                      (p, d, lambda_1, lambda_2))
    func_args = p, sigma_sq, store_x0, matrix_test, store_c
    f = mt_obj.sog_function
    g = mt_obj.sog_gradient
    tolerance = 0.00001
    projection = False
    starting_points = np.random.uniform(0, 1, (num_points, d))
    const = 0.1
    met = 'None'
    option = 'forward_backward_tracking'
    initial_guess = 0.005
    bounds_set_x = (0, 1)
    relax_sd_it = 1


    (unique_minimizers_mult,
     unique_number_of_minimizers_mult,
     store_func_vals_mult,
     time_taken_des,
     store_minimizer_des,
     no_its) = mt.multistart(f, g, func_args, d, starting_points,
                                          num_points, tolerance, projection,
                                          const, option, met, initial_guess,
                                          bounds_set_x, relax_sd_it)
    """Check outputs are as expected"""
    assert(len(unique_minimizers_mult) == unique_number_of_minimizers_mult)
    assert(unique_number_of_minimizers_mult == len(store_func_vals_mult))

    """Ensure that each region of attraction discovered is unique"""
    mt_obj.check_unique_minimizers(store_minimizer_des, 
                                    unique_number_of_minimizers_mult,
                                    mt_obj.calc_minimizer_sog, func_args)
    assert(time_taken_des >= 0)
    assert(np.all(no_its > 0))