import numpy as np

from metod_alg import metod_analysis as mt_ays


def all_comparisons_matches_both(d, store_x_values_list, store_z_values_list,
                                 num_points, store_minimizer, num, beta,
                                 counter_non_matchings, number_its_compare,
                                 g, func_args):
    """
    Apply steepest descent iterations to each starting point, chosen
    uniformly at random from [bounds_1, bounds_2]^d. The number of
    starting points to generate is dependent on the value of num_points.
    Produce arrays that show the number of times the METOD algorithm
    inequality [1, Eq. 9] fails for a given iteration number.

    Parameters
    ----------
    d : integer
        Size of dimension.
    store_x_values_list : list
                          Contains all trajectories from all starting points
                          and store_x_values_list is of length num_points.
    store_z_values_list : list
                          Contains all corresponding partner points of
                          store_x_values_list.
    num_points : integer
                 Total number of points to generate uniformly at random from
                 [bounds_1, bounds_2]^d.
    store_minimizer : 1-D array
                      The region of attraction index of each trajectory.
    num: integer
         Iteration number to start comparing inequalities. E.g for
         trajectories x_i^(k_i) and x_j^(k_j), we have k_i =
         (num,...,K) and k_j = (num,...,K).
    beta : float or integer
           Small constant step size to compute the partner points.
    counter_non_matchings : integer
                            Total number of trajectories which belong to
                            different regions of attraction.
    number_its_compare : integer
                         Number of iterations to compare. For
                         example k_j, k_i = (num,...,number_its_compare).
    g : gradient of objective function.

       ``g(x, *func_args) -> 1-D array with shape (d, )``

        where ``x`` is a 1-D array with shape(d, ) and func_args is a
        tuple of arguments needed to compute the gradient.
    func_args : tuple
                Arguments passed to f and g.


    Returns
    -------
    all_comparison_matrix_sm : 2-D array with shape (number_its_compare - num,
                               number_its_compare - num)
                               Array which counts the total number of times
                               the METOD algroithm condition [1, Eq. 9]
                               fails for points, x_j^(k_j), x_j^(k_j + 1),
                               x_i^(k_i) and x_i^(k_i + 1), where x_j and x_i
                               belong to the same region of attraction.
                               E.g if the  condition fails, a 1 will be added
                               to the all_comparison_matrix_sm array at
                               position (k_j - num, k_i - num).
    count_comparisons_sm : 2-D array with shape (number_its_compare - num,
                           number_its_compare - num)
                           Array which counts the total number of times
                           the METOD algroithm condition [1, Eq. 9] is
                           evaluated for points, x_i^(k_j), x_i^(k_j + 1),
                           x_j^(k_i) and x_j^(k_i + 1), where x_j and x_i
                           belong to the same region of attraction.
                           E.g a 1 will be added to the count_comparisons_sm
                           array at position (k_j - num, k_i - num).
    total_number_of_checks_sm : integer
                                Counts the total number of trajectories
                                x_i^(k_j) and x_j^(k_i) which belong to
                                the same region of attraction.
    all_comparison_matrix_nsm : 2-D array with shape (number_its_compare - num,
                                number_its_compare - num)
                                Same as all_comparison_matrix_sm, with
                                the exception that x_j and x_i belong to
                                different regions of attraction.
    count_comparisons_nsm : 2-D array with shape (number_its_compare - num,
                            number_its_compare - num)
                            Same as count_comparisons_sm, with
                            the exception that x_j and x_i belong to
                            different regions of attraction.
    total_number_of_checks_nsm : integer
                                 Counts the total number of trajectories
                                 x_j^(k_j) and x_i^(k_i) which belong to
                                 different regions of attraction.
    calculate_sum_quantities_nsm : 1-D array with shape
                                   (counter_non_matchings, )
                                   Calculates b ** 2 + 2 * b.T @ (x_j - x_i),
                                   where x_j and x_i belong to different
                                   regions of attraction and
                                   b = beta * (g(x_i, *func_args) -
                                   g(x_j, *func_args)).
    indices_nsm : 2-D array with shape (counter_non_matchings, 2)
                  Contains the indicies (j, i) of x_j and x_i, which
                  belong to different regions of attraction.

    References
    ----------
    1) Zilinskas, A., Gillard, J., Scammell, M., Zhigljavsky, A.: Multistart
       with early termination of descents. Journal of Global Optimization pp.
       1–16 (2019)
    """
    all_comparison_matrix_sm = np.zeros((number_its_compare - num,
                                         number_its_compare - num))
    total_number_of_checks_sm = np.zeros((number_its_compare - num,
                                          number_its_compare - num))
    count_comparisons_sm = 0

    all_comparison_matrix_nsm = np.zeros((number_its_compare - num,
                                          number_its_compare - num))
    total_number_of_checks_nsm = np.zeros((number_its_compare - num,
                                           number_its_compare - num))
    count_comparisons_nsm = 0
    calculate_sum_quantities_nsm = np.zeros((counter_non_matchings))
    indices_nsm = np.zeros((counter_non_matchings, 2))
    index_nsm = 0

    for j in range((num_points)):
        assert(len(store_x_values_list[j]) >= number_its_compare + 1)
        x_tr_1 = store_x_values_list[j]
        z_tr_1 = store_z_values_list[j]
        for i in range(j + 1, num_points):
            assert(len(store_x_values_list[i]) >= number_its_compare + 1)
            x_tr_2 = store_x_values_list[i][:number_its_compare + 1]
            z_tr_2 = store_z_values_list[i][:number_its_compare + 1]
            comparisons_check, total_checks = (mt_ays.individual_comparisons
                                               (d, x_tr_1, z_tr_1, x_tr_2,
                                                z_tr_2, number_its_compare,
                                                num))

            if int(store_minimizer[j]) == int(store_minimizer[i]):
                count_comparisons_sm += 1
                all_comparison_matrix_sm += comparisons_check
                total_number_of_checks_sm += total_checks
            else:
                count_comparisons_nsm += 1
                all_comparison_matrix_nsm += comparisons_check
                total_number_of_checks_nsm += total_checks
                calculate_sum_quantities_nsm[index_nsm] = (mt_ays.
                                                           check_quantities
                                                           (beta,
                                                            x_tr_1[0].reshape
                                                            (d, ),
                                                            x_tr_2[0].reshape
                                                            (d, ), g,
                                                            func_args))
                indices_nsm[index_nsm, 0] = j
                indices_nsm[index_nsm, 1] = i
                index_nsm += 1
    return (all_comparison_matrix_sm, count_comparisons_sm,
            total_number_of_checks_sm, all_comparison_matrix_nsm,
            count_comparisons_nsm, total_number_of_checks_nsm,
            calculate_sum_quantities_nsm, indices_nsm)
