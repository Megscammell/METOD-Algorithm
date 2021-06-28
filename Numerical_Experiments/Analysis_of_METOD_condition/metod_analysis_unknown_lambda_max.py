import numpy as np
import sys
from time import process_time
from time import perf_counter

from metod_alg import metod_analysis as mt_ays
from metod_alg import objective_functions as mt_obj


def metod_analysis_unknwon_lambda_max(f, g, func_args, d, bound_1, bound_2,
                                      number_its_compare, num, test_beta,
                                      func_name, num_point):
    """Calculates the total number of times the METOD algorithm condition
    fails for trajectories that belong to the same region of attraction and
    different regions of attraction. Saves all results for different values of
    beta.

    Parameters
    ----------
    d : integer
        Size of dimension.

    """
    num_functions = 100
    projection = False
    tolerance = 0.00001
    option = 'minimize_scalar'
    met = 'Brent'
    initial_guess = 0.005
    usage = 'metod_algorithm'
    relax_sd_it = 1
    (all_comparison_matrix_nsm_total,
     total_number_of_checks_nsm_total,
     all_comparison_matrix_sm_total,
     total_number_of_checks_sm_total,
     calculate_sum_quantities_nsm_each_func,
     store_all_its) = (mt_ays.main_analysis_other
                                                (d, f, g, check_func, func_args,
                                                 test_beta,
                                                 num_functions,
                                                 num_points,
                                                 projection, tolerance,
                                                 option, met,
                                                 initial_guess, bound_1,
                                                 bound_2, usage,
                                                 relax_sd_it, num, number_its_compare))
    np.savetxt('%s_store_all_its_nsm_d=%s_%s_relax_c=%s_num=%s_%s.csv' %
                   (func_name, d, projection, relax_sd_it, num, met),
                   store_all_its, delimiter=",")
    index = 0
    for beta in test_beta:
        np.savetxt('%s_beta=%s_nsm_d=%s_%s_relax_c=%s_num=%s_%s.csv' %
                   (func_name, beta, d, projection, relax_sd_it, num, met),
                   all_comparison_matrix_nsm_total[index], delimiter=",")
        np.savetxt('%s_beta=%s_tot_nsm_d=%s_%s_relax_c=%s_num=%s_%s.csv' %
                   (func_name, beta, d, projection, relax_sd_it, num, met),
                   total_number_of_checks_nsm_total[index], delimiter=",")
        np.savetxt('%s_beta=%s_sm_d=%s_%s_relax_c=%s_num=%s_%s.csv' %
                   (func_name, beta, d, projection, relax_sd_it, num, met),
                   all_comparison_matrix_sm_total[index], delimiter=",")
        np.savetxt('%s_beta=%s_tot_sm_d=%s_%s_relax_c=%s_num=%s_%s.csv' %
                   (func_name, beta, d, projection, relax_sd_it, num, met),
                   total_number_of_checks_sm_total[index], delimiter=",")

        if np.all(total_number_of_checks_nsm_total[index] > 0):
            prop_nsm = (all_comparison_matrix_nsm_total[index] /
                        total_number_of_checks_nsm_total[index])
            np.savetxt('%s_beta=%s_nsm_d=%s_prop_%s_relax_c=%s_num=%s_%s.csv' %
                        (func_name, beta, d, projection, relax_sd_it, num, met),
                        prop_nsm, delimiter=",")

        if np.all(total_number_of_checks_sm_total[index] > 0):
            prop_sm = (all_comparison_matrix_sm_total[index] /
                        total_number_of_checks_sm_total[index])
            np.savetxt('%s_beta=%s_sm_d=%s_prop_%s_relax_c=%s_num=%s_%s.csv' %
                    (func_name, beta, d, projection, relax_sd_it, num, met),
                    prop_sm, delimiter=",")
        index += 1


if __name__ == "__main__":
    func_name = str(sys.argv[1])
    if func_name == 'styb':
        d = 5
        num_points = 100
        f = mt_obj.styblinski_tang_function
        g = mt_obj.styblinski_tang_gradient
        check_func = mt_obj.calc_minimizer_styb
        func_args = ()
        bound_1 = -5
        bound_2 = 5
        number_its_compare = 2
        num = 0
        test_beta = [0.001, 0.01, 0.025, 0.05]
    
    elif func_name == 'qing':
        d = 5
        num_points = 100
        f = mt_obj.qing_function
        g = mt_obj.qing_gradient
        check_func = mt_obj.calc_minimizer_qing
        func_args = (d,)
        bound_1 = -3
        bound_2 = 3
        number_its_compare = 3
        num = 1
        test_beta = [0.001, 0.01, 0.025, 0.05, 0.1]

    elif func_name == 'zak':
        d = 10
        num_points = 100
        f = mt_obj.zakharov_func
        g = mt_obj.zakharov_grad
        check_func = None
        func_args = (d,)
        bound_1 = -5
        bound_2 = 10
        number_its_compare = 2
        num = 0
        test_beta = [0.00000001, 0.000001, 0.0001]


    metod_analysis_unknwon_lambda_max(f, g, func_args, d, bound_1, bound_2,
                                      number_its_compare, num, test_beta,
                                      func_name, num_points)
