import numpy as np
import sys

from metod_alg import metod_analysis as mt_ays


def metod_analysis_quad(d):
    """
    Calculates the total number of times the METOD algorithm condition
    fails for trajectories that belong to the same region of attraction and
    different regions of attraction. Saves all results for different values of
    beta.

    Parameters
    ----------
    d : integer
        Size of dimension.

    """
    test_beta = [0.01, 0.1, 0.2]
    num_functions = 100
    num_points = 100
    p = 2
    lambda_1 = 1
    lambda_2 = 10
    projection = False
    tolerance = 0.01
    option = 'minimize_scalar'
    met = 'Brent'
    initial_guess = 0.005
    usage = 'metod_algorithm'
    bound_1 = 0
    bound_2 = 1
    relax_sd_it = 1
    num = 0
    number_its_compare = 3
    (all_comparison_matrix_nsm_total,
     total_number_of_checks_nsm_total,
     all_comparison_matrix_sm_total,
     total_number_of_checks_sm_total,
     calculate_sum_quantities_nsm_each_func,
     store_all_its,
     store_all_norm_grad) = (mt_ays.main_analysis_quad
                             (d, test_beta,
                              num_functions,
                              num_points, p,
                              lambda_1, lambda_2,
                              projection, tolerance,
                              option, met,
                              initial_guess, bound_1,
                              bound_2, usage,
                              relax_sd_it, num, number_its_compare))

    np.savetxt('quad_store_all_its_nsm_d=%s_%s_relax_c=%s_num=%s_%s.csv' %
               (d, projection, relax_sd_it, num, met),
               store_all_its, delimiter=",")
    np.savetxt('quad_store_all_grad_norms_nsm_d=%s_%s_relax'
               '_c=%s_num=%s_%s.csv' %
               (d, projection, relax_sd_it, num, met),
               store_all_norm_grad, delimiter=",")
    index = 0
    for beta in test_beta:
        max_b = np.zeros(2)
        np.savetxt('quad_beta=%s_nsm_d=%s_%s_relax_c=%s_num=%s_%s.csv' %
                   (beta, d, projection, relax_sd_it, num, met),
                   all_comparison_matrix_nsm_total[index], delimiter=",")
        np.savetxt('quad_beta=%s_tot_nsm_d=%s_%s_relax_c=%s_num=%s_%s.csv' %
                   (beta, d, projection, relax_sd_it, num, met),
                   total_number_of_checks_nsm_total[index], delimiter=",")
        np.savetxt('quad_beta=%s_sm_d=%s_%s_relax_c=%s_num=%s_%s.csv' %
                   (beta, d, projection, relax_sd_it, num, met),
                   all_comparison_matrix_sm_total[index], delimiter=",")
        np.savetxt('quad_beta=%s_tot_sm_d=%s_%s_relax_c=%s_num=%s_%s.csv' %
                   (beta, d, projection, relax_sd_it, num, met),
                   total_number_of_checks_sm_total[index], delimiter=",")

        prop_nsm = (all_comparison_matrix_nsm_total[index, :11, :11] /
                    total_number_of_checks_nsm_total[index, :11, :11])
        prop_sm = (all_comparison_matrix_sm_total[index, :11, :11] /
                   total_number_of_checks_sm_total[index, :11, :11])
        np.savetxt('quad_beta=%s_nsm_d=%s_prop_%s_relax_c=%s_num=%s_%s.csv' %
                   (beta, d, projection, relax_sd_it, num, met),
                   prop_nsm, delimiter=",")
        np.savetxt('quad_beta=%s_sm_d=%s_prop_%s_relax_c=%s_num=%s_%s.csv' %
                   (beta, d, projection, relax_sd_it, num, met),
                   prop_sm, delimiter=",")
        max_b[0] = np.argmax(calculate_sum_quantities_nsm_each_func[index])
        max_b[1] = np.max(calculate_sum_quantities_nsm_each_func[index])
        np.savetxt('quad_calculate_quantities_beta=%s_d=%s_prop_%s_relax_c=%s_'
                   'num=%s_%s.csv' % (beta, d, projection, relax_sd_it, num,
                                      met), max_b, delimiter=",")
        index += 1


if __name__ == "__main__":
    d = int(sys.argv[1])
    metod_analysis_quad(d)
