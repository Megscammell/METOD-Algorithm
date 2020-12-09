import numpy as np
from hypothesis import given, settings, strategies as st

from metod import metod_analysis as mt_ays
from metod import metod_algorithm_functions as mt_alg
from metod import objective_functions as mt_obj


def test_1():
    """
    Testing functionality of all_comparisons_matches_both.py, where point
    indices that do not belong to the same region of attraction are stored."""
    num_points = 5
    store_minimizer = np.array([0, 1, 0, 1, 1])
    combos_nsm = np.array(np.zeros((6, 2)))
    combos_sm = np.array(np.zeros((4, 2)))
    count_comparisons_sm = 0
    count_comparisons_nsm = 0
    for j in range((num_points)):
        for k in range(j + 1, num_points):
            if int(store_minimizer[j]) == int(store_minimizer[k]):
                combos_sm[count_comparisons_sm] = np.array([j, k])
                count_comparisons_sm += 1
            else:
                combos_nsm[count_comparisons_nsm] = np.array([j, k])
                count_comparisons_nsm += 1
    assert(count_comparisons_sm == 4)
    assert(count_comparisons_nsm == 6)
    assert(np.all(combos_nsm == np.array([[0, 1],
                                         [0, 3],
                                         [0, 4],
                                         [1, 2],
                                         [2, 3],
                                         [2, 4]])))
    assert(np.all(combos_sm == np.array([[0, 2],
                                        [1, 3],
                                        [1, 4],
                                        [3, 4]])))


@settings(max_examples=50, deadline=None)
@given(st.integers(20, 100), st.integers(5, 20), st.integers(11, 20),
       st.integers(0, 10), st.floats(0.0001, 0.1))
def test_2(d, num_points, tolerance, num, beta):
    """
    Ensuring outputs of all_comparisons_both.py have expected
    properties.
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
     counter_matchings) = (mt_ays.compute_trajectories
                           (num_points, d, projection, tolerance, option,
                            met, initial_guess, func_args, f, g, bounds_1,
                            bounds_2, usage, relax_sd_it))
    store_z_values_list = []
    for i in range(num_points):
        points_x = store_x_values_list[i]
        points_z = mt_alg.partner_point_each_sd(points_x, d, beta,
                                                tolerance, g, func_args)
        store_z_values_list.append(points_z)
    (all_comparison_matrix_sm, count_comparisons_sm,
     total_number_of_checks_sm,
     all_comparison_matrix_nsm,
     count_comparisons_nsm,
     total_number_of_checks_nsm,
     calculate_sum_quantities_nsm,
     indices_nsm) = (mt_ays.all_comparisons_matches_both
                     (d, store_x_values_list, store_z_values_list, num_points,
                      store_minimizer, num, beta, counter_non_matchings,
                      tolerance, func_args))
    assert(all_comparison_matrix_sm.shape ==
           (tolerance - num, tolerance - num))
    assert(total_number_of_checks_sm.shape ==
           (tolerance - num, tolerance - num))
    assert(all_comparison_matrix_nsm.shape ==
           (tolerance - num, tolerance - num))
    assert(total_number_of_checks_nsm.shape ==
           (tolerance - num, tolerance - num))
    assert(count_comparisons_sm == counter_matchings)
    assert(count_comparisons_nsm == counter_non_matchings)
    assert(indices_nsm.shape == (counter_non_matchings, 2))
    assert(calculate_sum_quantities_nsm.shape == (counter_non_matchings, ))
