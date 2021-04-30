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
def metod_numerical_exp_shekel(f_t, g_t, func_args_t, d_t,
                               num_p_t, beta_t, m_t, option_t,
                               met_t, tolerance_t, initial_guess_t,
                               set_x_t):
    """Apply METOD algorithm with specified parameters.

    Parameters
    ----------
    f_t : Shekel function.

          ``f(x, *func_args) -> float``

          where ``x`` is a 1-D array with shape(d, ) and func_args is a
          tuple of arguments needed to compute the function value.
    g_t : Shekel gradient.

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
    tolerance_t : integer or float
                  Stopping condition for steepest descent iterations. Apply
                  steepest descent iterations until the norm
                  of g(point, *func_args) is less than some tolerance.
                  Also check that the norm of the gradient at a starting point
                  is larger than some tolerance.
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
    time_taken_alg_perf_count: float
                               Amount of time in seconds the METOD algorithm
                               takes using perf_counter.
    time_taken_alg_process_t: float
                               Amount of time in seconds the METOD algorithm
                               takes using process_time.
    References
    ----------
    1) Zilinskas, A., Gillard, J., Scammell, M., Zhigljavsky, A.: Multistart
       with early termination of descents. Journal of Global Optimization pp.
       1–16 (2019)

    """
    t0 = time.time()
    (unique_minimizers_metod,
     unique_number_of_minimizers_metod,
     func_vals_of_minimizers_metod,
     extra_descents,
     starting_points) = mt.metod(f=f_t, g=g_t, func_args=func_args_t, d=d_t,
                                num_points=num_p_t, tolerance=tolerance_t,
                                beta=beta_t, m=m_t, option=option_t,
                                met=met_t, initial_guess=initial_guess_t,
                                set_x=set_x_t, bounds_set_x=(0, 10))
    t1 = time.time()
    time_taken_metod = t1-t0

    check_unique_minimizers = np.zeros((unique_number_of_minimizers_metod))
    index = 0
    for minimizer in unique_minimizers_metod:
        position_minimum, norm_with_minimizer = (mt_obj.calc_minimizer_shekel
                                                 (minimizer, *func_args))
        check_unique_minimizers[index] = position_minimum
        assert(norm_with_minimizer < 0.25)
        index += 1
    assert(np.unique(check_unique_minimizers).shape[0] == 
           unique_number_of_minimizers_metod)
    return (unique_number_of_minimizers_metod, extra_descents, time_taken_metod,
            np.min(func_vals_of_minimizers_metod))


if __name__ == "__main__":
    d = 4
    lambda_1 = int(sys.argv[1])
    lambda_2 = int(sys.argv[2])
    p = int(sys.argv[3])
    f = mt_obj.shekel_function
    g = mt_obj.shekel_gradient
    m = int(sys.argv[4])
    beta = float(sys.argv[5])
    met = 'None'
    option = 'forward_backward_tracking'
    initial_guess = 0.005
    set_x = str(sys.argv[6])
    num_p = int(sys.argv[7])
    num_func = 100
    num_workers = 1
    tolerance = 0.001
    projection = False
    number_minimizers_per_func_metod = np.zeros((num_func))
    number_extra_descents_per_func_metod = np.zeros((num_func))
    number_extra_descents_per_func = np.zeros((num_func))
    time_metod = np.zeros((num_func))
    func_val_metod = np.zeros((num_func))
    for func in tqdm.tqdm(range(num_func)):
        np.random.seed(func * 5)
        matrix_test, C, b = (mt_obj.function_parameters_shekel(
                                       lambda_1, lambda_2, p))
        func_args = p, matrix_test, C, b
        task = metod_numerical_exp_shekel(f, g, func_args, d,
                                          num_p, beta, m, option,
                                          met, tolerance, initial_guess,
                                          set_x)
        result = dask.compute(task, num_workers=num_workers)
        (number_minimizers_per_func_metod[func],
         number_extra_descents_per_func_metod[func],
         time_metod[func],
         func_val_metod[func]) = result[0]
    table = pd.DataFrame({
                         "number_minimas_per_func_metod":
                         number_minimizers_per_func_metod,
                         "number_extra_descents_per_func_metod":
                         number_extra_descents_per_func_metod,
                         "time_metod": time_metod,
                         "func_val_metod": func_val_metod})
    table.to_csv(table.to_csv
                 ('shekel_metod_beta_%s_m=%s_d=%s'
                 '_p=%s_%s_%s.csv.csv'
                  % (beta, m, d, p, set_x,
                     num_p)))
