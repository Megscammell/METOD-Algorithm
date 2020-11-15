import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

from metod import metod_analysis as mt_ays
from metod import objective_functions as mt_obj
from metod import metod_algorithm_functions as mt_alg
sns.set()


def plot_figure(beta, pos_largest_calculation, store_b, projection):
    """Produces bar charts of the c_1 and c_2 components.

    Parameters
    ----------
    beta : float or integer (optional)
           Small constant step size to compute the partner points.
    pos_largest_calculation : integer
                              Function parameter number which produces points
                              that have largest b ** 2 + 2 * b.T @ (x - y).
                              Look at
                              calculate_quantities_beta=%s_d=%s_prop_%s_relax_c
                              =%s_'.csv and find the row number of the largest
                              value for a particular value of beta.
    store_b_quantities : 2-D array with shape (4, 2)
                         Array containing values of c_1 and c_2 (columns) for
                         each iteration (rows).
    projection : boolean
                 If projection is True, this projects points back to
                 bounds_set_x. If projection is False, points are
                 kept the same.


    """
    plt.figure(figsize=(18, 5))
    X = [r'$c_1$',
         r'$c_2$']

    Y = store_b[0, :]
    Z = store_b[1, :]
    R = store_b[2, :]
    K = store_b[3, :]
    X_num = np.arange(len(X))
    plt.bar(X_num - 0.3, Y, 0.2, color=sns.xkcd_rgb["medium blue"])
    plt.bar(X_num - 0.1, Z, 0.2, color=sns.xkcd_rgb["medium green"])
    plt.bar(X_num + 0.1, R, 0.2, color=sns.xkcd_rgb["medium purple"])
    plt.bar(X_num + 0.3, K, 0.2, color=sns.xkcd_rgb["pale red"])
    plt.xticks(X_num, X, fontsize=20)
    plt.yticks(fontsize=20)
    blue_patch = mpatches.Patch(color=sns.xkcd_rgb["medium blue"],
                                label=r'$k_1=1,k_2=1$')
    green_patch = mpatches.Patch(color=sns.xkcd_rgb["medium green"],
                                 label=r'$k_1=1,k_2=2$')
    purple_patch = mpatches.Patch(color=sns.xkcd_rgb["medium purple"],
                                  label=r'$k_1=2,k_2=1$')
    red_patch = mpatches.Patch(color=sns.xkcd_rgb["pale red"],
                               label=r'$k_1=2,k_2=2$')
    plt.legend(handles=[blue_patch, green_patch, purple_patch, red_patch],
               bbox_to_anchor=[1.21, 1.035], loc='upper right',
               prop={'size': 20})
    plt.savefig('beta=%s_quadratic_%s_proj=%s.pdf' %
                (beta, pos_largest_calculation, projection),
                bbox_inches="tight")


def metod_analysis_compute_quantities(beta, d, pos_largest_calculation):
    """Calculates the quantities for the two points which produce largest b **
    2 + 2 * b.T @ (x - y), where b = where b = beta * (g(y, *func_args) -
    g(x, *func_args)).

    Parameters
    ----------
    beta : float or integer
           Small constant step size to compute the partner points.
    d : integer
        Size of dimension.
    pos_largest_calculation : integer
                              Function parameter number which produces points
                              that have largest b ** 2 + 2 * b.T @ (x - y).
                              Look at
                              calculate_quantities_beta=%s_d=%s_prop_%s_relax_c
                              =%s_'.csv and find the row number of the largest
                              value for a particular value of beta.

    """
    f = mt_obj.quad_function
    g = mt_obj.quad_gradient
    num_points = 100
    p = 2
    lambda_1 = 1
    lambda_2 = 10
    projection = False
    tolerance = 15
    option = 'minimize'
    met = 'L-BFGS-B'
    initial_guess = 0.05
    bound_1 = 0
    bound_2 = 1
    usage = 'metod_analysis'
    relax_sd_it = 1
    num = 1
    np.random.seed(pos_largest_calculation + 1)
    store_x0, matrix_test = mt_obj.function_parameters_quad(p, d, lambda_1,
                                                            lambda_2)
    func_args = p, store_x0, matrix_test
    (store_x_values, store_minima,
     counter_non_match, counter_match) = (mt_ays.compute_trajectories
                                          (num_points, d, projection,
                                           tolerance, option, met,
                                           initial_guess, func_args,
                                           f, g, bound_1, bound_2, usage,
                                           relax_sd_it))
    store_z_values = []
    for i in range(num_points):
        points_x = store_x_values[i]
        points_z = mt_alg.partner_point_each_sd(points_x, d, beta,
                                                tolerance, g, func_args)
        store_z_values.append(points_z)

    (all_comparison_matrix_sm, count_comparisons_sm,
     total_number_of_checks_sm, all_comparison_matrix_nsm,
     count_comparisons_nsm, total_number_of_checks_nsm,
     calculate_sum_quantities_nsm,
     indices_nsm) = (mt_ays.all_comparisons_matches_both
                     (d, store_x_values, store_z_values,
                      num_points, store_minima, num, beta,
                      counter_non_match, tolerance,
                      func_args))

    pos_1 = indices_nsm[np.argmax(calculate_sum_quantities_nsm), 0]
    pos_2 = indices_nsm[np.argmax(calculate_sum_quantities_nsm), 1]
    x_tr = store_x_values[int(pos_1)]
    y_tr = store_x_values[int(pos_2)]
    min_x = store_minima[int(pos_1)]
    min_y = store_minima[int(pos_2)]

    # data = np.genfromtxt('calculate_quantities_beta=%s_d=%s_prop_False'
    #                      '_relax_c=1_num=1_L-BFGS-B.csv' % (beta, d),
    #                       delimiter=",")
    # assert(mt_ays.check_quantities(beta, x_tr[0], y_tr[0],
    #                                func_args) == data[1])

    store_b, sum_b = (mt_ays.evaluate_quantities_with_points
                      (beta, x_tr, y_tr, int(min_x), int(min_y), d, func_args))

    new_store_b = np.zeros((4, 2))
    for i in range(4):
        new_store_b[i, 0] = store_b[i, 0] + store_b[i, 1] + store_b[i, 2]
        new_store_b[i, 1] = store_b[i, 3] + store_b[i, 4]
    for i in range(4):
        assert(np.round(np.sum(new_store_b[i]), 6) ==
               np.round(sum_b[i], 6))
    sns.set_style("white")
    plot_figure(beta, pos_largest_calculation, new_store_b, projection)
    np.savetxt('quantities_beta_%s_nsm_d_%s_%s_proj_%s.csv' %
               (beta, d, pos_largest_calculation, projection),
               new_store_b, delimiter=",")
    np.savetxt('sum_beta_%s_nsm_d_%s_%s_proj_%s.csv' %
               (beta, d, pos_largest_calculation, projection),
               sum_b, delimiter=",")


if __name__ == "__main__":
    beta = float(sys.argv[1])
    d = int(sys.argv[2])
    pos_largest_calculation = int(sys.argv[3])
    metod_analysis_compute_quantities(beta, d, pos_largest_calculation)
