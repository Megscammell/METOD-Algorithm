import numpy as np
from hypothesis import given, settings, strategies as st

import metod.metod_analysis as mt_ays
import metod.objective_functions as mt_obj
import metod.metod_algorithm as mt_alg


@settings(max_examples=50, deadline=None)
@given(st.integers(20, 100), st.floats(0.0001, 0.1))
def test_1(d, beta):
    """Test that outputs from evaluate_quantities_with_points.py are the same
    as check_quantities.py for different values of d and beta.
    """
    p = 2
    lambda_1 = 1
    lambda_2 = 10
    f = mt_obj.quad_function
    g = mt_obj.quad_gradient
    projection = False
    tolerance = 2
    option = 'minimize'
    met = 'Nelder-Mead'
    initial_guess = 0.05
    bound_1 = 0
    bound_2 = 1
    usage = 'metod_analysis'
    relax_sd_it = 1
    store_x0, matrix_test = mt_obj.function_parameters_quad(p, d, lambda_1,
                                                            lambda_2)
    func_args = p, store_x0, matrix_test

    x = np.random.uniform(0, 1, (d, ))
    y = np.random.uniform(0, 1, (d, ))
    while (mt_obj.calc_pos(x, *func_args)[0] ==
           mt_obj.calc_pos(y, *func_args)[0]):
        x = np.random.uniform(0, 1, (d, ))
        y = np.random.uniform(0, 1, (d, ))

    x_tr, its_x = (mt_alg.apply_sd_until_stopping_criteria
                   (x, d, projection, tolerance, option, met, initial_guess,
                    func_args, f, g, bound_1, bound_2, usage, relax_sd_it))
    assert(its_x == tolerance)
    y_tr, its_y = (mt_alg.apply_sd_until_stopping_criteria
                   (y, d, projection, tolerance, option, met, initial_guess,
                    func_args, f, g, bound_1, bound_2, usage, relax_sd_it))
    assert(its_y == tolerance)
    min_x = int(mt_obj.calc_pos(x, *func_args)[0])
    min_y = int(mt_obj.calc_pos(y, *func_args)[0])
    quantities_array, sum_quantities = (mt_ays.evaluate_quantities_with_points
                                        (beta, x_tr, y_tr, min_x, min_y, d,
                                         func_args))

    assert(np.round(sum_quantities[0], 5) == np.round(mt_ays.check_quantities
                                                      (beta, x_tr[1, :], y_tr
                                                       [1, :], func_args), 5))
    assert(np.round(sum_quantities[1], 5) == np.round(mt_ays.check_quantities
                                                      (beta, x_tr[1, :], y_tr
                                                       [2, :], func_args), 5))
    assert(np.round(sum_quantities[2], 5) == np.round(mt_ays.check_quantities
                                                      (beta, x_tr[2, :], y_tr
                                                       [1, :], func_args), 5))
    assert(np.round(sum_quantities[3], 5) == np.round(mt_ays.check_quantities
                                                      (beta, x_tr[2, :], y_tr
                                                       [2, :], func_args), 5))
