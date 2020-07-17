import numpy as np
import sys

import metod.metod_analysis as mt_ays
import metod.objective_functions as mt_obj


def metod_analysis(d):
    """Calculates the total number of times the METOD algorithm condition
    fails for trajectories that belong to the same region of attraction and
    different regions of attraction. Saves all results for different values of
    beta.

    Parameters
    ----------
    d : integer
        Size of dimension.

    """
    f = mt_obj.quad_function
    g = mt_obj.quad_gradient
    test_beta = [0.001, 0.01, 0.1]
    num_functions = 100
    num_points = 100
    p = 2
    lambda_1 = 1
    lambda_2 = 10
    projection = False
    tolerance = 15
    option = 'minimize'
    met = 'Nelder-Mead'
    initial_guess = 0.05
    usage = 'metod_analysis'
    bound_1 = 0
    bound_2 = 1
    relax_sd_it = 1
    num = 1
    (all_comparison_matrix_nsm_total, total_number_of_checks_nsm_total,
     all_comparison_matrix_sm_total, total_number_of_checks_sm_total,
     calculate_sum_quantities_nsm_each_func) = (mt_ays.main_analysis_quad
                                                (d, f, g, test_beta,
                                                 num_functions,
                                                 num_points, p,
                                                 lambda_1, lambda_2,
                                                 projection, tolerance,
                                                 option, met,
                                                 initial_guess, bound_1,
                                                 bound_2, usage,
                                                 relax_sd_it, num))
    index = 0
    for beta in test_beta:
        max_b = np.zeros(2)
        np.savetxt('beta=%s_nsm_d=%s_%s_relax_c=%s_num=%s.csv' %
                   (beta, d, projection, relax_sd_it, num),
                   all_comparison_matrix_nsm_total[index], delimiter=",")
        np.savetxt('beta=%s_tot_nsm_d=%s_%s_relax_c=%s_num=%s.csv' %
                   (beta, d, projection, relax_sd_it, num),
                   total_number_of_checks_nsm_total[index], delimiter=",")
        np.savetxt('beta=%s_sm_d=%s_%s_relax_c=%s_num=%s.csv' %
                   (beta, d, projection, relax_sd_it, num),
                   all_comparison_matrix_sm_total[index], delimiter=",")
        np.savetxt('beta=%s_tot_sm_d=%s_%s_relax_c=%s_num=%s.csv' %
                   (beta, d, projection, relax_sd_it, num),
                   total_number_of_checks_sm_total[index], delimiter=",")

        prop_nsm = (all_comparison_matrix_nsm_total[index, :11, :11] /
                    total_number_of_checks_nsm_total[index, :11, :11])
        prop_sm = (all_comparison_matrix_sm_total[index, :11, :11] /
                   total_number_of_checks_sm_total[index, :11, :11])
        np.savetxt('beta=%s_nsm_d=%s_prop_%s_relax_c=%s_num=%s.csv' %
                   (beta, d, projection, relax_sd_it, num),
                   prop_nsm, delimiter=",")
        np.savetxt('b_%s_sm_d_%s_prop_%s_relax_c=%s_num=%s.csv' %
                   (beta, d, projection, relax_sd_it, num),
                   prop_sm, delimiter=",")
        max_b[0] = np.argmax(calculate_sum_quantities_nsm_each_func[index])
        max_b[1] = np.max(calculate_sum_quantities_nsm_each_func[index])
        np.savetxt('calculate_quantities_beta=%s_d=%s_prop_%s_relax_c=%s_'
                   'num=%s.csv' % (beta, d, projection, relax_sd_it, num),
                   max_b, delimiter=",")
        index += 1


if __name__ == "__main__":
    d = int(sys.argv[1])
    metod_analysis(d)
