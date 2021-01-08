import dask
import numpy as np
import tqdm
import time
import sys
import pandas as pd
from time import process_time
from time import perf_counter

import metod as mt
from metod import objective_functions as mt_obj


@dask.delayed
def metod_numerical_exp_quad(f_t, g_t, func_args_t, d_t,
                             num_p_t, beta_t, m_t, option_t,
                             met_t, initial_guess_t, set_x_t):
    """Apply METOD algorithm with specified parameters.

    Parameters
    ----------
    f_t : Minimum of several Quadratic forms objective function.

          ``f(x, *func_args) -> float``

          where ``x`` is a 1-D array with shape(d, ) and func_args is a
          tuple of arguments needed to compute the function value.
    g_t : Minimum of several Quadratic forms gradient.

         ``g(x, *func_args) -> 1-D array with shape (d, )``

          where ``x`` is a 1-D array with shape(d, ) and func_args is a
          tuple of arguments needed to compute the gradient.
    func_args_t : tuple
                  Arguments passed to f and g.
    d_t : integer
          Size of dimension.
    num_p_t : integer
              Number of random points generated.
    beta_t : float or integer
             Small constant step size to compute the partner points.
    m_t : integer
          Number of iterations of steepest descent to apply to point
          x before making decision on terminating descents.
    option_t : string
               Choose from 'minimize' or 'minimize_scalar'. For more
               information about each option see
               https://docs.scipy.org/doc/scipy/reference/optimize.html.
    met_t : string
           Choose method for option. For more information see
           - https://docs.scipy.org/doc/scipy/reference/generated/
           scipy.optimize.minimize.html#scipy.optimize.minimize
           - https://docs.scipy.org/doc/scipy/reference/generated/
           scipy.optimize.minimize_scalar.html#scipy.optimize.minimize_scalar
    initial_guess_t : float or integer
                      Initial guess passed to scipy.optimize.minimize and the
                      upper bound for the bracket interval when using the
                      'Brent' or 'Golden' method for
                      scipy.optimize.minimize_scalar. This
                      is recommended to be small.
    set_x_t : string
              If set_x = 'random', random starting points
              are generated for the METOD algorithm. If set_x = 'sobol'
              is selected, then a numpy.array with shape
              (num points * 2, d) of Sobol sequence samples are generated
              using SALib [1], which are randomly shuffled and used
              as starting points for the METOD algorithm.

    Returns
    -------
    unique_number_of_minimizers: integer
                                 Total number of unique minimizers found.
    extra_descents : integer
                     Number of excessive descents. Occurs when
                     [1, Eq. 9] does not hold for trajectories
                     that belong to the region of attraction
                     of the same local minimizer.
    time_taken_alg: float
                    Amount of time in seconds the METOD algorithm takes.

    References
    ----------
    1) Zilinskas, A., Gillard, J., Scammell, M., Zhigljavsky, A.: Multistart
       with early termination of descents. Journal of Global Optimization pp.
       1–16 (2019)

    """
    start_process_time = process_time()
    start_perf_counter = perf_counter()
    t0 = time.time()
    (unique_minimizers, unique_number_of_minimizers_alg,
     func_vals_of_minimizers, extra_descents,
     starting_points) = mt.metod(f=f_t, g=g_t, func_args=func_args_t, d=d_t,
                                 num_points=num_p_t, beta=beta_t, m=m_t,
                                 option=option_t, met=met_t, 
                                 initial_guess=initial_guess_t,
                                 set_x=set_x_t, bounds_set_x=(0, 1))
    end_process_time = process_time()
    end_perf_counter = perf_counter()
    t1 = time.time()
    time_taken_alg = t1-t0
    time_taken_alg_perf_count = end_perf_counter - start_perf_counter
    time_taken_alg_process_t = end_process_time - start_process_time
    for minimizer in unique_minimizers:
        position_minimum, norm_with_minimizer = mt_obj.calc_pos(minimizer,
                                                                *func_args)
        assert(norm_with_minimizer < 0.1)
    return (unique_number_of_minimizers_alg, extra_descents, time_taken_alg,
            time_taken_alg_perf_count, time_taken_alg_process_t)


if __name__ == "__main__":
    d = int(sys.argv[1])
    p = int(sys.argv[2])
    lambda_1 = int(sys.argv[3])
    lambda_2 = int(sys.argv[4])
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    m = int(sys.argv[5])
    beta = float(sys.argv[6])
    met = str(sys.argv[7])
    option = str(sys.argv[8])
    initial_guess = float(sys.argv[9])
    set_x = str(sys.argv[10])
    num_p = int(sys.argv[11])
    num_func = 100
    num_workers = 1
    number_minimizers_per_func_metod = np.zeros((num_func))
    number_extra_descents_per_func_metod = np.zeros((num_func))
    number_extra_descents_per_func = np.zeros((num_func))
    time_metod = np.zeros((num_func))
    time_perf_metod = np.zeros((num_func))
    time_process_metod = np.zeros((num_func))
    for func in tqdm.tqdm(range(num_func)):
        np.random.seed(func * 5)
        store_x0, matrix_test = (mt_obj.function_parameters_several_quad
                                 (p, d, lambda_1, lambda_2))
        func_args = p, store_x0, matrix_test
        task = metod_numerical_exp_quad(f, g, func_args, d, num_p, beta,
                                        m, option, met, initial_guess,
                                        set_x)
        result = dask.compute(task, num_workers=num_workers)
        (unique_number_of_minimizers_alg,
         extra_descents, time_taken_alg,
         time_taken_alg_perf_count, time_taken_alg_process_t) = result[0]
        number_minimizers_per_func_metod[func] = unique_number_of_minimizers_alg
        number_extra_descents_per_func_metod[func] = extra_descents
        time_metod[func] = time_taken_alg
        time_perf_metod[func] = time_taken_alg_perf_count
        time_process_metod[func] = time_taken_alg_process_t
    table = pd.DataFrame({
                         "number_minimas_per_func_metod":
                         number_minimizers_per_func_metod,
                         "number_extra_descents_per_func_metod":
                         number_extra_descents_per_func_metod,
                         "time_metod": time_metod,
                         "perf_counter": time_perf_metod,
                         "process_time": time_process_metod})
    table.to_csv(table.to_csv
                 ('quad_testing_minimize_met_%s_beta_%s_m=%s_d=%s_p=%s'
                  '_%s_%s_%s.csv'
                  % (met, beta, m, d, p, initial_guess, set_x,
                     num_p)))
