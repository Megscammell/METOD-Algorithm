import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import pandas as pd

from metod_alg import metod_analysis as mt_ays
from metod_alg import objective_functions as mt_obj
from metod_alg import metod_algorithm_functions as mt_alg
sns.set()


def plot_figure(beta, func_name, pos_largest_calculation, store_b, projection):
    """
    Produces bar charts of the c_1 and c_2 components.

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
    plt.figure(figsize=(7, 5))
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
                                label='$k_1=1$, \n$k_2=1$')
    green_patch = mpatches.Patch(color=sns.xkcd_rgb["medium green"],
                                 label='$k_1=1$, \n$k_2=2$')
    purple_patch = mpatches.Patch(color=sns.xkcd_rgb["medium purple"],
                                  label='$k_1=2$, \n$k_2=1$')
    red_patch = mpatches.Patch(color=sns.xkcd_rgb["pale red"],
                               label='$k_1=2$, \n$k_2=2$')
    plt.legend(handles=[blue_patch, green_patch, purple_patch, red_patch],
            bbox_to_anchor=[0.99, 1.035], loc='upper left',
            prop={'size': 20})
    plt.savefig('beta=%s_%s_%s_proj=%s.png' %
                (beta, func_name, pos_largest_calculation, projection),
                bbox_inches="tight")


def metod_analysis_compute_quantities(func_name):
    """
    Calculates the quantities c_1 and c_2 for two points, where c_1 + c_2
    = b ** 2 + 2 * b.T @ (x - y), with b = beta * (g(y, *func_args) -
    g(x, *func_args)).

    Parameters
    ----------
    func_name : string
                Choose between 'quad' and 'sog'.

    """
    pos_largest_calculation = 50
    beta_list = [0.001, 0.01, 0.1]
    if func_name == 'quad':
        p = 2
        d = 100
        f = mt_obj.several_quad_function
        g = mt_obj.several_quad_gradient
        check_func = mt_ays.calc_minimizer_sev_quad_no_dist_check
        usage = 'metod_analysis'
        tolerance = 15
        number_its_compare = 4
        bound_1 = 0
        bound_2 = 1
    elif func_name == 'sog':
        p = 10
        d = 20
        sigma_sq = 0.7
        f = mt_obj.sog_function
        g = mt_obj.sog_gradient
        check_func = mt_obj.calc_minimizer_sog
        usage = 'metod_algorithm'
        tolerance = 0.000001
        number_its_compare = 4
        bound_1 = 0
        bound_2 = 1
    elif func_name == 'shekel':
        p = 10
        d = 4
        f = mt_obj.shekel_function
        g = mt_obj.shekel_gradient
        check_func = mt_obj.calc_minimizer_shekel
        usage = 'metod_algorithm'
        tolerance = 0.0001
        number_its_compare = 4
        bound_1 = 0
        bound_2 = 10
    num_points = 100
    lambda_1 = 1
    lambda_2 = 10
    projection = False
    option = 'minimize_scalar'
    met = 'Brent'
    initial_guess = 0.005
    relax_sd_it = 1
    num = 1
    np.random.seed(pos_largest_calculation + 1)
    if func_name == 'quad':
        store_x0, matrix_test = (mt_obj.function_parameters_several_quad
                                (p, d, lambda_1, lambda_2))
        func_args = p, store_x0, matrix_test
        func_args_check_func = func_args
    elif func_name == 'sog':
        store_x0, matrix_test, store_c = (mt_obj.function_parameters_sog
                                          (p, d, lambda_1, lambda_2))
        func_args = (p, sigma_sq, store_x0, matrix_test, store_c)
        func_args_check_func = func_args
    elif func_name == 'shekel':
        matrix_test, C, b = (mt_obj.function_parameters_shekel
                             (lambda_1, lambda_2, p))
        func_args = (p, matrix_test, C, b)
        func_args_check_func = func_args
    (store_x_values, store_minima,
     counter_non_match, counter_match,
     store_grad_all) = (mt_ays.compute_trajectories
                        (num_points, d, projection,
                        tolerance, option, met,
                        initial_guess, func_args,
                        f, g, bound_1, bound_2, usage,
                        relax_sd_it, check_func,
                        func_args_check_func))

    c1_c2_prop = np.zeros((len(beta_list), 4))
    c1_plus_c2 = np.zeros((len(beta_list), 4))
    index = 0
    for beta in beta_list:
        store_z_values = []
        for i in range(num_points):
            points_x = store_x_values[i]
            grad_x = store_grad_all[i]
            points_z = mt_alg.partner_point_each_sd(points_x, beta,
                                                    grad_x)
            store_z_values.append(points_z)

        (all_comparison_matrix_sm,
        count_comparisons_sm,
        total_number_of_checks_sm,
        all_comparison_matrix_nsm,
        count_comparisons_nsm,
        total_number_of_checks_nsm,
        calculate_sum_quantities_nsm,
        indices_nsm) = (mt_ays.all_comparisons_matches_both
                        (d, store_x_values, store_z_values,
                        num_points, store_minima, num, beta,
                        counter_non_match, number_its_compare,
                        g, func_args))

        pos_1 = indices_nsm[np.argmax(calculate_sum_quantities_nsm), 0]
        pos_2 = indices_nsm[np.argmax(calculate_sum_quantities_nsm), 1]
        x_tr = store_x_values[int(pos_1)]
        y_tr = store_x_values[int(pos_2)]
        min_x = store_minima[int(pos_1)]
        min_y = store_minima[int(pos_2)]
        assert(min_x != min_y)
        if func_name == 'quad':
            store_b, sum_b = (mt_ays.evaluate_quantities_with_points_quad
                              (beta, x_tr, y_tr, int(min_x), int(min_y),
                               d, g, func_args))
            new_store_b = np.zeros((4, 2))
            for i in range(4):
                new_store_b[i, 0] = np.sum(store_b[i, 0] + store_b[i, 1] +
                                        store_b[i, 2])
                new_store_b[i, 1] = np.sum(store_b[i, 3] + store_b[i, 4])
                c1_c2_prop[index, i] = ((store_b[i, 0] + store_b[i, 1] +
                                        store_b[i, 2]) /
                                        (store_b[i, 3] + store_b[i, 4]))
                assert(np.round(np.sum(store_b[i]), 6) ==
                        np.round(sum_b[i], 6))
                c1_plus_c2[index, i] = sum_b[i]

            sns.set_style("white")
            plot_figure(beta, func_name, pos_largest_calculation, new_store_b, projection)

        elif func_name == 'sog' or func_name == 'shekel':
            store_b, sum_b = (mt_ays.evaluate_quantities_with_points
                              (beta, x_tr, y_tr,
                               d, g, func_args))
            for i in range(4):
                assert(np.round(np.sum(store_b[i]), 6) ==
                        np.round(sum_b[i], 6))
                c1_c2_prop[index, i] = store_b[i, 0] / store_b[i, 1]
                c1_plus_c2[index, i] = sum_b[i]
            sns.set_style("white")
            plot_figure(beta, func_name, pos_largest_calculation, store_b, projection)
        index += 1

    np.savetxt('%s_c1_c2_prop_nsm_d_%s_%s_proj_%s.csv' %
               (func_name, d, pos_largest_calculation, projection),
               np.round(abs(c1_c2_prop), 4), delimiter=",")
    np.savetxt('%s_c1_plus_c2_nsm_d_%s_%s_proj_%s.csv' %
               (func_name, d, pos_largest_calculation, projection),
               np.round(c1_plus_c2, 4), delimiter=",")
    
    full_table = np.concatenate((np.round(c1_plus_c2, 4),
                                 np.round(abs(c1_c2_prop), 4)))
    full_table_pd_nsm = pd.DataFrame(data = full_table,
                                 index=[0.001, 0.01, 0.1, 0.001, 0.01, 0.1],
                                 columns = [1,2,3,4])
    full_table_pd_nsm.to_csv('%s_quant_nsm_d=%s_%s_relax_c=%s_num=%s_%s.csv' %
                            (func_name, d, projection, relax_sd_it, num, met))
    with open('%s_quant_nsm_d=%s_%s_relax_c=%s_num=%s_%s.tex' 
            % (func_name, d, projection, relax_sd_it, num, met), 'w') as tf:
        tf.write(full_table_pd_nsm.to_latex())

if __name__ == "__main__":
    func_name = str(sys.argv[1])
    metod_analysis_compute_quantities(func_name)
