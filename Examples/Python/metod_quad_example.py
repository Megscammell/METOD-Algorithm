import numpy as np
import pandas as pd
import sys

import metod as mt
from metod import objective_functions as mt_obj


def metod_quad(d, seed, P, lambda_1, lambda_2):
    """
    Example of using the METOD algorithm with the minimum of several quadratic
    forms objective function.
    """
    np.random.seed(seed)
    f = mt_obj.quad_function
    g = mt_obj.quad_gradient
    store_x0, matrix_combined = mt_obj.function_parameters_quad(P, d, lambda_1,
                                                                lambda_2)
    args = P, store_x0, matrix_combined
    (discovered_minimizers, number_minimizers,
     func_vals_of_minimizers, excessive_no_descents) = mt.metod(f, g, args, d)
    np.savetxt('discovered_minimizers_d_%s_p_%s_quad.csv' % (d, P),
               discovered_minimizers, delimiter=",")
    np.savetxt('func_vals_discovered_minimizers_d_%s_p_%s_quad.csv' % (d, P),
               func_vals_of_minimizers, delimiter=",")
    summary_table = pd.DataFrame({
                                "Total number of unique minimizers": [number_minimizers],
                                "Excessive descents": [excessive_no_descents]})
    summary_table.to_csv('summary_table_d_%s_p_%s_quad.csv' % (d, P))


if __name__ == "__main__":
    d = int(sys.argv[1])
    seed = int(sys.argv[2])
    P = int(sys.argv[3])
    lambda_1 = int(sys.argv[4])
    lambda_2 = int(sys.argv[5])
    metod_quad(d, seed, P, lambda_1, lambda_1)
