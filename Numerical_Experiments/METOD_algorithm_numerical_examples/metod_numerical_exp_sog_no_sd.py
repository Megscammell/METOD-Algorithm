import dask
import numpy as np
import tqdm
import time
import sys
import pandas as pd
from SALib.sample import sobol_sequence

import metod as mt
from metod import objective_functions as mt_obj


@dask.delayed
def metod_numerical_exp_sog(f_t, g_t, func_args_t, d_t,
                            num_p_t, beta_t, m_t, option_t,
                            met_t):
    """Apply METOD algorithm with specified parameters.

    Parameters
    ----------
    f_t : Sum of Gaussians objective function.

          ``f(x, *func_args) -> float``

          where ``x`` is a 1-D array with shape(d, ) and func_args is a
          tuple of arguments needed to compute the function value.
    g_t : Sum of Gaussians gradient.

         ``g(x, *func_args) -> 1-D array with shape (d, )``

          where ``x`` is a 1-D array with shape(d, ) and func_args is a
          tuple of arguments needed to compute the gradient.
    func_args_t : tuple
                  Arguments passed to f and g.
    d_t : integer
          Size of dimension.
    num_p_t : integer
              Number of random points generated. The Default is
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
    set_x_t = 'random'
    t0 = time.time()
    (unique_minimizers, unique_number_of_minimizers_alg,
     func_vals_of_minimizers, extra_descents,
     starting_points) = mt.metod(f=f_t, g=g_t, func_args=func_args_t, d=d_t,
                                 num_points=num_p_t, beta=beta_t, m=m_t,option=option_t, met=met_t, set_x=set_x_t,
                                 bounds_set_x=(0, 1))
    for minimizer in unique_minimizers:
        pos_minimizer, min_dist = mt_obj.calc_minimizer(minimizer, *func_args)
        assert(min_dist < 0.35)
    t1 = time.time()
    time_taken_alg = t1-t0
    return unique_number_of_minimizers_alg, extra_descents, time_taken_alg


if __name__ == "__main__":
    d = int(sys.argv[1])
    p = int(sys.argv[2])
    sigma_sq = float(sys.argv[3])
    lambda_1 = int(sys.argv[4])
    lambda_2 = int(sys.argv[5])
    f = mt_obj.sog_function
    g = mt_obj.sog_gradient
    m_t = int(sys.argv[6])
    beta_t = float(sys.argv[7])
    met_t = str(sys.argv[8])
    option_t = str(sys.argv[9])
    num_p_t = 100
    num_func = 100
    num_workers = 1
    number_minimizers_per_func_metod = np.zeros((num_func))
    number_extra_descents_per_func_metod = np.zeros((num_func))
    number_extra_descents_per_func = np.zeros((num_func))
    time_metod = np.zeros((num_func))
    for func in tqdm.tqdm(range(num_func)):
        np.random.seed(func * 10)
        store_x0, matrix_test, store_c = (mt_obj.function_parameters_sog
                                          (p, d, lambda_1, lambda_2))
        func_args = p, sigma_sq, store_x0, matrix_test, store_c
        task = metod_numerical_exp_sog(f, g, func_args, d, num_p_t, beta_t,
                                       m_t, option_t, met_t)
        result = dask.compute(task, num_workers=num_workers)
        (unique_number_of_minimizers_alg,
         extra_descents, time_taken_alg) = result[0]
        number_minimizers_per_func_metod[func] = unique_number_of_minimizers_alg
        number_extra_descents_per_func_metod[func] = extra_descents
        time_metod[func] = time_taken_alg
    table = pd.DataFrame({
                         "number_minimas_per_func_metod":
                         number_minimizers_per_func_metod,
                         "number_extra_descents_per_func_metod":
                         number_extra_descents_per_func_metod,
                         "time_metod": time_metod})
    table.to_csv(table.to_csv
                 ('sog_testing_minimize_met_%s_beta_%s_m=%s_d=%s'
                  '_p=%s_random.csv' %
                  (met_t, beta_t, m_t, d, p)))
