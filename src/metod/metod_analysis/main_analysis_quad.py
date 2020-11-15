import numpy as np
import tqdm

from metod import metod_algorithm_functions as mt_alg
from metod import objective_functions as mt_obj
from metod import metod_analysis as mt_ays


def main_analysis_quad(d, f, g, test_beta, num_functions, num_points, p,
                       lambda_1, lambda_2, projection, tolerance, option, met,
                       initial_guess, bounds_1, bounds_2, usage, relax_sd_it,
                       num):
    """
    Calculates the total number of times the METOD algorithm condition
    fails for trajectories that belong to the same region of attraction and
    different regions of attraction, for a large number of different function
    parameters and different values of beta.

    Parameters
    ----------
    d : integer
        Size of dimension.
    f : objective function.

        ``f(x, *func_args) -> float``

        where ``x`` is a 1-D array with shape(d, ) and func_args is a
        tuple of arguments needed to compute the function value.
    g : gradient of objective function.

       ``g(x, *func_args) -> 1-D array with shape (d, )``

        where ``x`` is a 1-D array with shape(d, ) and func_args is a
        tuple of arguments needed to compute the gradient.
    test_beta : list
                Contains a list of small constant step sizes to compute the
                partner points.
    num_functions : integer
                    Number of different function parameters.
    num_points : integer
                 Total number of points generated uniformly at random from
                 [0,1]^d.
    p : integer
        Number of local minima.
    lambda_1 : integer
               Smallest eigenvalue of diagonal matrix.
    lambda_2 : integer
               Largest eigenvalue of diagonal matrix.
    projection : boolean
                 If projection is True, points are projected back to
                 bounds_set_x. If projection is False, points are
                 kept the same.
    tolerance : integer or float
                Stopping condition for steepest descent iterations.
                Steepest descent iterations are applied until the total number
                of iterations is greater than some tolerance (usage =
                metod_analysis).
    option : string
             Choose from 'minimize' or 'minimize_scalar'. For more
             information about each option see
             https://docs.scipy.org/doc/scipy/reference/optimize.html.
    met : string
         Choose method for option. For more information see
         - https://docs.scipy.org/doc/scipy/reference/generated/
         scipy.optimize.minimize.html#scipy.optimize.minimize
         - https://docs.scipy.org/doc/scipy/reference/generated/
         scipy.optimize.minimize_scalar.html#scipy.optimize.minimize_scalar
    initial_guess : float or integer
                    Initial guess passed to scipy.optimize.minimize. This
                    is recommended to be small.
    bounds_1 : integer
               Lower bound used for projection.
    bounds_2 : integer
               Upper bound used for projection.
    usage : string
            Used to decide stopping criterion for steepest descent
            iterations. Should be either usage='metod_algorithm' or
            usage='metod_analysis'.
    relax_sd_it : float or integer
                  Multiply the step size by a small constant in [0, 2], to
                  obtain a new step size for steepest descent iterations. This
                  process is known as relaxed steepest descent [1].
    num: integer
         Iteration number to start comparing inequalities. E.g for
         trajectories x_i^(k_i) and x_j^(k_j), we have k_i =
         (num,...,K_i) and k_j = (num,...,K_i).


    Returns
    -------
    fails_nsm_total : 3-D array with shape
                      (len(test_beta), iterations - num, iterations - num)
                       The array all_comparison_matrix_nsm will be added to
                       fails_nsm_total for each set of function parameters.
                       each value of beta.
    checks_nsm_total : 3-D array with shape
                      (len(test_beta), iterations - num, iterations - num)
                       The array count_comparisons_nsm will be added to
                       fails_nsm_total for each set of function parameters.
    fails_sm_total : 3-D array with shape
                     (len(test_beta), iterations - num, iterations - num)
                      The array all_comparison_matrix_sm will be added to
                      fails_sm_total for each set of function parameters.
    checks_sm_total : 3-D array with shape
                      (len(test_beta), iterations - num, iterations - num)
                       The array count_comparisons_sm will be added to
                       fails_sm_total for each set of function parameters.
    calculate_sum_quantities_nsm_each_func : 2-D array with shape
                                             (len(test_beta),
                                             num_functions)
                                             Stores the maximum value of b **
                                             2 + 2 * b.T @ (x_j - x_i),
                                             for each function and different
                                             values of beta from test_beta,
                                             where x_j and x_i belong to
                                             different regions of attraction.

    References
    ----------
    1) Raydan, M., Svaiter, B.F.: Relaxed steepest descent and
       cauchy-barzilai- borwein method. Computational Optimization and
       Applications 21(2), 155–167 (2002)

    """

    calculate_sum_quantities_nsm_each_func = np.zeros((len(test_beta),
                                                       num_functions))
    fails_nsm_total = np.zeros((len(test_beta), tolerance - num, tolerance -
                                num))
    fails_sm_total = np.zeros((len(test_beta), tolerance - num, tolerance -
                               num))
    checks_nsm_total = np.zeros((len(test_beta), tolerance - num, tolerance -
                                 num))
    checks_sm_total = np.zeros((len(test_beta), tolerance - num, tolerance -
                                num))

    for j in tqdm.tqdm(range(num_functions)):
        np.random.seed(j + 1)
        total = (num_points * (num_points - 1)) / 2
        store_x0, matrix_test = mt_obj.function_parameters_quad(p, d, lambda_1,
                                                                lambda_2)
        func_args = p, store_x0, matrix_test
        (store_x_values_list,
         store_minimizer,
         counter_non_matchings,
         counter_matchings) = (mt_ays.compute_trajectories
                               (num_points, d, projection, tolerance, option,
                                met, initial_guess, func_args, f, g, bounds_1,
                                bounds_2, usage, relax_sd_it))

        index = 0
        for beta in test_beta:
            store_z_values_list = []
            for i in range(num_points):
                points_x = store_x_values_list[i]
                points_z = mt_alg.partner_point_each_sd(points_x, d, beta,
                                                        tolerance, g,
                                                        func_args)
                store_z_values_list.append(points_z)
            (count_sm, comparisons_sm,
             total_sm, count_nsm,
             comparisons_nsm, total_nsm,
             calculate_sum_quantities_nsm,
             indices_nsm) = (mt_ays.all_comparisons_matches_both
                             (d, store_x_values_list, store_z_values_list,
                              num_points, store_minimizer, num, beta,
                              counter_non_matchings, tolerance, func_args))
            assert(comparisons_nsm == counter_non_matchings)
            fails_nsm_total[index] += count_nsm
            checks_nsm_total[index] += total_nsm
            if counter_non_matchings > 0:
                calculate_sum_quantities_nsm_each_func[index, j] = np.max(calculate_sum_quantities_nsm)

            assert(comparisons_sm == counter_matchings)
            fails_sm_total[index] += count_sm
            checks_sm_total[index] += total_sm
            assert(comparisons_nsm + comparisons_sm == total)
            index += 1
    return (fails_nsm_total, checks_nsm_total, fails_sm_total,
            checks_sm_total, calculate_sum_quantities_nsm_each_func)
