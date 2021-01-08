import dask
import numpy as np
import tqdm
import time
import sys
import pandas as pd

import metod as mt
from metod import objective_functions as mt_obj
from metod import metod_algorithm_functions as mt_alg


@dask.delayed
def metod_numerical_exp_styb(f_t, g_t, func_args_t, d_t,
                             num_p_t, beta_t, m_t, option_t,
                             met_t, tolerance_t, projection_t,
                             initial_guess_t, set_x_t):
    """Apply METOD algorithm with specified parameters and also apply
    multistart.

    Parameters
    ----------
    f_t : Styblinski-Tang function.

          ``f(x, *func_args) -> float``

          where ``x`` is a 1-D array with shape(d, ) and func_args is a
          tuple of arguments needed to compute the function value.
    g_t : Styblinski-Tang gradient.

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
    tolerance_t: float
                 Stopping condition for steepest descent iterations. Apply
                 steepest descent iterations until the norm
                 of g(point, *func_args) is less than some tolerance.
                 Also check that the norm of the gradient at a starting point
                 is larger than some tolerance.
    projection_t : boolean
                   If projection is True, this projects points back to
                   bounds_set_x. If projection is False, points are
                   kept the same.
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
    unique_number_desended_minimizers: integer
                                       Total number of unique minimizers found
                                       by applying multistart.
    unique_number_of_minimizers_alg: integer
                                    Total number of unique minimizers found by
                                    applying METOD.
    extra_descents : integer
                     Number of excessive descents. Occurs when
                     [1, Eq. 9] does not hold for trajectories
                     that belong to the region of attraction
                     of the same local minimizer.
    time_taken_alg: float
                    Amount of time in seconds the METOD algorithm takes.
    time_taken_des: float
                    Amount of time in seconds multistart takes.

    References
    ----------
    1) Zilinskas, A., Gillard, J., Scammell, M., Zhigljavsky, A.: Multistart
       with early termination of descents. Journal of Global Optimization pp.
       1–16 (2019)

    """
    t0 = time.time()
    (unique_minimizers, unique_number_of_minimizers_alg,
     func_vals_of_minimizers, extra_descents,
     starting_points) = mt.metod(f=f_t, g=g_t, func_args=func_args_t, d=d_t,
                                 num_points=num_p_t, tolerance=tolerance_t, 
                                 beta=beta_t, m=m_t, option=option_t,
                                 met=met_t, initial_guess=initial_guess_t,
                                 set_x=set_x_t, bounds_set_x=(-5, 5))
    t1 = time.time()
    time_taken_alg = t1-t0

    check_unique_minimizers = np.zeros((unique_number_of_minimizers_alg))
    index = 0
    for minimizer in unique_minimizers:
        position_minimum, norm_with_minimizer = (mt_obj.calc_minimizer_styb
                                                 (minimizer, *func_args))
        assert(norm_with_minimizer < 0.5)
        check_unique_minimizers[index] = position_minimum
        index += 1
    assert(np.unique(check_unique_minimizers).shape[0] == 
           unique_number_of_minimizers_alg)

    t0 = time.time()
    store_pos_minimizer = np.zeros((num_p_t))
    store_start_end_pos = np.zeros((num_p_t))
    store_minimizer_des = np.zeros((num_p_t, d))
    for j in range(num_p_t):
        x = starting_points[j].reshape(d,)
        iterations_of_sd, its = (mt_alg.apply_sd_until_stopping_criteria
                                 (x, d=d_t, projection=projection_t,
                                  tolerance=tolerance_t, option=option_t,
                                  met=met_t, initial_guess=initial_guess_t,
                                  func_args=func_args_t, f=f_t, g=g_t,
                                  bound_1=-5, bound_2=5,
                                  usage='metod_algorithm', relax_sd_it=1))
        store_minimizer_des[j, :] = iterations_of_sd[its, :]

    t1 = time.time()
    time_taken_des = t1-t0
    for k in range(num_p_t):
        pos_minimizer, norm_with_minimizer = (mt_obj.calc_minimizer_styb
                                              (store_minimizer_des[k, :], 
                                               *func_args))
        assert(norm_with_minimizer < 0.5)
        store_pos_minimizer[k] = pos_minimizer

    unique_number_desended_minimizers = np.unique(store_pos_minimizer).shape[0]
    return (unique_number_desended_minimizers, unique_number_of_minimizers_alg,
            extra_descents, time_taken_alg, time_taken_des)


if __name__ == "__main__":
    d = int(sys.argv[1])
    f = mt_obj.styblinski_tang_function
    g = mt_obj.styblinski_tang_gradient
    m = int(sys.argv[2])
    beta = float(sys.argv[3])
    met = str(sys.argv[4])
    option = str(sys.argv[5])
    initial_guess = float(sys.argv[6])
    set_x = str(sys.argv[7])
    num_p = int(sys.argv[8])
    num_func = 100
    num_workers = 1
    tolerance = 0.00001
    projection = False
    number_minimizers_per_func_metod = np.zeros((num_func))
    number_extra_descents_per_func_metod = np.zeros((num_func))
    number_minimizers_per_func_multistart = np.zeros((num_func))
    number_extra_descents_per_func = np.zeros((num_func))
    time_metod = np.zeros((num_func))
    time_multistart = np.zeros((num_func))

    for func in tqdm.tqdm(range(num_func)):
        np.random.seed(func * 5)
        func_args = (d, )
        task = metod_numerical_exp_styb(f, g, func_args, d, num_p, beta,
                                        m, option, met, tolerance, 
                                        projection, initial_guess, set_x)
        result = dask.compute(task, num_workers=num_workers)
        (unique_number_desended_minimizers,
         unique_number_of_minimizers_alg,
         extra_descents, time_taken_alg,
         time_taken_des) = result[0]
        number_minimizers_per_func_metod[func] = unique_number_of_minimizers_alg
        number_extra_descents_per_func_metod[func] = extra_descents
        number_minimizers_per_func_multistart[func] = unique_number_desended_minimizers
        time_metod[func] = time_taken_alg
        time_multistart[func] = time_taken_des
    table = pd.DataFrame({
                         "number_minimas_per_func_metod":
                         number_minimizers_per_func_metod,
                         "number_extra_descents_per_func_metod":
                         number_extra_descents_per_func_metod,
                         "number_minimas_per_func_multistart":
                         number_minimizers_per_func_multistart,
                         "time_metod": time_metod,
                         "time_multistart": time_multistart})
    table.to_csv(table.to_csv
                 ('stybl_tang_sd_minimize_met_%s_beta_%s_m=%s_d=%s'
                 '_%s_%s_%s.csv' %
                  (met, beta, m, d, initial_guess, set_x, num_p)))
