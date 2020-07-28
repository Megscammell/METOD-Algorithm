import numpy as np
import pandas as pd

import metod as mt
from metod import objective_functions as mt_obj


"""
1) Define a function, gradient and dimension. Here the minimum of several
 quadratic forms function and gradient is used, and the dimension is 20.
"""
f = mt_obj.quad_function
g = mt_obj.quad_gradient
d = 50

"""
2) Will need to amend/add/delete the below if function and gradient is not
 the minimum of several quadratic forms. Otherwise, can update values for the
 below parameters
"""
P = 5
lambda_1 = 1
lambda_2 = 10
store_x0, matrix_combined = mt_obj.function_parameters_quad(P, d, lambda_1,
                                                            lambda_2)
args = P, store_x0, matrix_combined

"""
3) Run METOD algorithm. If any of the optional input parameters need to be
 updated this will need to be passed to mt.metod(f, g, args, d).
"""
discovered_minima, number_minima, func_vals_of_minima, excessive_no_descents = mt.metod(f, g, args, d)
np.savetxt('discovered_minimas_d_%s_p_%s.csv' % (d, P), discovered_minima,
           delimiter=",")

"""
4) Save outputs from metod.py
"""
np.savetxt('func_vals_discovered_minimas_d_%s_p_%s.csv' % (d, P),
           func_vals_of_minima, delimiter=",")

summary_table = pd.DataFrame({
                             "Total number of unique minima": [number_minima],
                             "Extra descents": [excessive_no_descents]})
summary_table.to_csv('summary_table_d_%s_p_%s.csv' % (d, P))
