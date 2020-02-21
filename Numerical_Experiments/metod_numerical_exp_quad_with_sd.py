import dask
import numpy as np
import tqdm
import time
import sys
import pandas as pd

import metod_testing as mtv3

@dask.delayed
def metod_numerical_exp_quad(f, g, func_args, d, num_points = 1000, beta = 0.01, tolerance = 0.00001, projection = True, const = 0.1,m = 3, option = 'minimize', met='Nelder-Mead', initial_guess = 0.05):
    
    t0 = time.time()

    unique_minimas, unique_number_of_minima_alg, func_vals_of_minimas, extra_descents, store_its, des_x_points, des_z_points, starting_points = mtv3.metod_indepth(f, g, func_args, d, num_points, beta, tolerance, projection, const, m, option, met, initial_guess)

    for minima in unique_minimas:
        position_minimum, norm_with_minima = mtv3.calc_pos(minima, *func_args)
        assert(norm_with_minima < 0.1)

    t1 = time.time()
    time_taken_alg = t1-t0


    t0 = time.time()
    store_pos_minima = np.zeros((num_points))
    store_start_end_pos = np.zeros((num_points))
    store_dist_minima = np.zeros((num_points))
    for j in range(num_points):
        x = starting_points[j,:].reshape(d,)
        iterations_of_sd, its = mtv3.apply_sd_until_stopping_criteria(x, d, projection, tolerance, option, met, initial_guess, func_args, f, g)

        pos_minima, norm_with_minima = mtv3.calc_pos(iterations_of_sd[its].reshape(d,), *func_args)
        assert(norm_with_minima < 0.1)
        store_pos_minima[j] = pos_minima

    t1 = time.time()
    time_taken_des = t1-t0
        
    #checking steepest descent iterations
    for k in range(num_points):  
        pos_start_point, norm_with_minima_sp = mtv3.calc_pos(starting_points[k,:].reshape(d,),*func_args)

        if store_pos_minima[k] != pos_start_point:
            store_start_end_pos[k] = 1
        
    assert(np.all(store_start_end_pos == 0))

    
    unique_number_desended_minima = np.unique(store_pos_minima).shape[0]

    return unique_number_desended_minima, unique_number_of_minima_alg,  extra_descents, time_taken_alg, time_taken_des


if __name__ == "__main__":
    d = int(sys.argv[1])
    p = int(sys.argv[2])
    lambda_1 = int(sys.argv[3])
    lambda_2 = int(sys.argv[4])
    f = mtv3.quad_function
    g = mtv3.quad_gradient
    m_t = int(sys.argv[5])
    beta_t = float(sys.argv[6])

    num_func = 100
    num_workers = 1

    number_minimas_per_func_metod = np.zeros((num_func))
    number_extra_descents_per_func_metod = np.zeros((num_func))
    number_minimas_per_func_multistart = np.zeros((num_func))
    number_extra_descents_per_func = np.zeros((num_func))
    time_metod = np.zeros((num_func))
    time_multistart = np.zeros((num_func))

    for func in tqdm.tqdm(range(num_func)):
        np.random.seed(func * 5)
        store_x0, matrix_test = mtv3.function_parameters_quad(p, d, lambda_1, lambda_2)
        func_args = p, store_x0, matrix_test

        task = metod_numerical_exp_quad(f, g, func_args, d, beta = beta_t, m = m_t)
        result = dask.compute(task, num_workers=num_workers) 
        unique_number_desended_minima, unique_number_of_minima_alg, extra_descents, time_taken_alg, time_taken_des =  result[0]
    
        number_minimas_per_func_metod[func] = unique_number_of_minima_alg
        number_extra_descents_per_func_metod[func] = extra_descents
        number_minimas_per_func_multistart[func] = unique_number_desended_minima
        time_metod[func] = time_taken_alg
        time_multistart[func] = time_taken_des



    table = pd.DataFrame({
    "number_minimas_per_func_metod": number_minimas_per_func_metod,
    "number_extra_descents_per_func_metod": number_extra_descents_per_func_metod,

    "number_minimas_per_func_multistart": number_minimas_per_func_multistart,
    "time_metod": time_metod,
    "time_multistart": time_multistart})
    table.to_csv(table.to_csv('quad_testing_minimize_met_%s_beta_%s_m=%s_d=%s_p=%s.csv' % ('Nelder-Mead', beta_t, m_t, d, p)))


