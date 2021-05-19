import numpy as np
from hypothesis import given, settings, strategies as st

from metod_alg import metod_analysis as mt_ays
from metod_alg import objective_functions as mt_obj


@settings(max_examples=50, deadline=None)
@given(st.integers(20, 100), st.integers(5, 20), st.integers(11, 20),
       st.floats(0.0001, 0.1))
def test_1(d, num_points, tolerance, beta):
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
    option = 'minimize'
    met = 'Nelder-Mead'
    initial_guess = 0.05
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
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
                        bounds_2, usage, relax_sd_it))
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
