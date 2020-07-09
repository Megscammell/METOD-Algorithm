import dask
import numpy as np
import tqdm
import time
import sys
import pandas as pd

import metod_testing as mtv3


@dask.delayed
def metod_numerical_exp_quad(f, g, func_args, d, num_points=1000, beta=0.01,
                             tolerance=0.00001, projection=False, const=0.1,
                             m=3, option='minimize', met='Nelder-Mead',
                             initial_guess=0.05):
    t0 = time.time()
    (unique_minimas, unique_number_of_minima_alg,
     func_vals_of_minimas, extra_descents, store_its,
     des_x_points, des_z_points, starting_points) = (mtv3.metod_indepth(f, g,
                                                     func_args, d, num_points,
                                                     beta, tolerance,
                                                     projection, const, m,
                                                     option, met,
                                                     initial_guess)
                                                     )
    t1 = time.time()
    time_taken_alg = t1-t0
    for minima in unique_minimas:
        position_minimum, norm_with_minima = mtv3.calc_pos(minima, *func_args)
        assert(norm_with_minima < 0.1)
    return unique_number_of_minima_alg,  extra_descents, time_taken_alg


if __name__ == "__main__":
    d = int(sys.argv[1])
    p = int(sys.argv[2])
    lambda_1 = int(sys.argv[3])
    lambda_2 = int(sys.argv[4])
    f = mtv3.quad_function
    g = mtv3.quad_gradient
    m_t = int(sys.argv[5])
    beta_t = float(sys.argv[6])
    met_t = str(sys.argv[7])
    option_t = str(sys.argv[8])
    num_func = 100
    num_workers = 1
    number_minimas_per_func_metod = np.zeros((num_func))
    number_extra_descents_per_func_metod = np.zeros((num_func))
    number_extra_descents_per_func = np.zeros((num_func))
    time_metod = np.zeros((num_func))
    for func in tqdm.tqdm(range(num_func)):
        np.random.seed(func * 5)
        store_x0, matrix_test = mtv3.function_parameters_quad(p, d, lambda_1,
                                                              lambda_2)
        func_args = p, store_x0, matrix_test
        task = metod_numerical_exp_quad(f, g, func_args, d, beta=beta_t,
                                        m=m_t, option=option_t, met=met_t)
        result = dask.compute(task, num_workers=num_workers)
        unique_number_of_minima_alg, extra_descents, time_taken_alg = result[0]
        number_minimas_per_func_metod[func] = unique_number_of_minima_alg
        number_extra_descents_per_func_metod[func] = extra_descents
        time_metod[func] = time_taken_alg
    table = pd.DataFrame({
                         "number_minimas_per_func_metod":
                         number_minimas_per_func_metod,
                         "number_extra_descents_per_func_metod":
                         number_extra_descents_per_func_metod,
                         "time_metod": time_metod})
    table.to_csv(table.to_csv
                 ('quad_testing_minimize_met_%s_beta_%s_m=%s_d=%s_p=%s.csv' %
                  (met_t, beta_t, m_t, d, p)))
