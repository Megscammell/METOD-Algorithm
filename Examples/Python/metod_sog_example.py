import numpy as np
import pandas as pd
import sys

import metod_alg as mt
from metod_alg import objective_functions as mt_obj


def metod_sog(d, seed, P, sigma_sq, lambda_1, lambda_2):
    """
    Example of using the METOD algorithm with the Sum of Gaussians objective
    function.
    """
    np.random.seed(seed)
    f = mt_obj.sog_function
    g = mt_obj.sog_gradient
    store_x0, matrix_combined, store_c = (mt_obj.function_parameters_sog
                                          (P, d, lambda_1, lambda_2))
    args = P, sigma_sq, store_x0, matrix_combined, store_c
    (discovered_minimizers, number_minimizers,
     func_vals_of_minimizers, excessive_no_descents,
     starting_points, no_grad_evals) = mt.metod(f, g, args, d)
    np.savetxt('discovered_minimizers_d_%s_p_%s_sog.csv' % (d, P),
               discovered_minimizers, delimiter=",")
    np.savetxt('func_vals_discovered_minimizers_d_%s_p_%s_sog.csv' % (d, P),
               func_vals_of_minimizers, delimiter=",")
    summary_table = (pd.DataFrame({
                    "Total number of unique minimizers": [number_minimizers],
                    "Excessive descents": [excessive_no_descents]}))
    summary_table.to_csv('summary_table_d_%s_p_%s_sog.csv' % (d, P))


if __name__ == "__main__":
    d = int(sys.argv[1])
    seed = int(sys.argv[2])
    P = int(sys.argv[3])
    sigma_sq = float(sys.argv[4])
    lambda_1 = int(sys.argv[5])
    lambda_2 = int(sys.argv[6])
    metod_sog(d, seed, P, sigma_sq, lambda_1, lambda_2)
