import numpy as np
import pandas as pd

import metod as mt
from metod import objective_functions as mt_obj


"""
1) Define a function, gradient and dimension. Here the Sum of Gaussians function and gradient is used, and the dimension is 20.
"""
f = mt_obj.sog_function
g = mt_obj.sog_gradient
d = 100

"""
2) Update values for the below parameters and update seed number if required.
Please note that sigma_sq needs to be carefully chosen, otherwise unexpected
results may occur. Can select sigma_sq by trial and error.
"""

np.random.seed(90)
P = 5
lambda_1 = 1
lambda_2 = 10
sigma_sq = 4
store_x0, matrix_combined, store_c = mt_obj.function_parameters_sog(P, d, 
                                                                  lambda_1, lambda_2)
args = P, sigma_sq, store_x0, matrix_combined, store_c

"""
3) Run METOD algorithm. If any of the optional input parameters need to be
 updated this will need to be passed to mt.metod(f, g, args, d). Update seed number if required.
"""

np.random.seed(91)
(discovered_minima, number_minima,
 func_vals_of_minima, excessive_no_descents) = mt.metod(f, g, args, d, 
                                                        met='Nelder-Mead')
np.savetxt('discovered_minimas_d_%s_p_%s.csv' % (d, P), discovered_minima,
           delimiter=",")

"""
4) Save outputs from metod.py
"""

np.savetxt('func_vals_discovered_minimas_d_%s_p_%s_sog.csv' % (d, P),
           func_vals_of_minima, delimiter=",")

summary_table = pd.DataFrame({
                             "Total number of unique minima": [number_minima],
                             "Extra descents": [excessive_no_descents]})
summary_table.to_csv('summary_table_d_%s_p_%s_sog.csv' % (d, P))
