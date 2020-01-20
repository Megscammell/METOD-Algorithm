import sys  
sys.path.insert(0, 'metod_testing/src')
import metod_testing as mtv3
import pandas as pd


### Need to update the following parameters ###
f = mtv3.quad_function
g = mtv3.quad_gradient
d = 20
P = 5
lambda_1 = 1
lambda_2 = 10
store_x0, matrix_combined = mtv3.function_parameters_quad(P, d, lambda_1, lambda_2)
args = P, store_x0, matrix_combined
###############################################
discovered_minima, number_minima, func_vals_of_minima, excessive_no_descents  = mtv3.metod(f, g, args, d)
np.savetxt('discovered_minimas_d_%s_p_%s.csv' % (d, P), discovered_minima, delimiter=",")

np.savetxt('func_vals_discovered_minimas_d_%s_p_%s.csv' % (d, P), func_vals_of_minima, delimiter=",")

summary_table = pd.DataFrame({
"Total number of unique minima": [number_minima],
"Extra descents": [excessive_no_descents]})
summary_table.to_csv('summary_table_d_%s_p_%s.csv' % (d, P))

