import dask
import numpy as np
import tqdm
import time
import sys
import pandas as pd

import metod as mt
import metod.objective_functions as mt_obj
import metod.metod_algorithm as mt_alg


@dask.delayed
def metod_numerical_exp_quad(f_t, g_t, func_args_t, d_t,
                             num_p_t, beta_t, m_t, option_t,
                             met_t, no_inequals, tolerance_t, projection_t,
                             initial_guess_t):
    """Apply METOD algorithm with specified parameters and also apply
    multistart.

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
    no_inequals : string
                  Evaluate METOD algroithm condition with all
                  iterations ('All') or two iterations
                  ('Two').
    tolerance_t: float
                 Stopping condition for steepest descent iterations.
    projection_t : boolean
                   If projection is True, this projects points back to
                   bounds_set_x. If projection is False, points are
                   kept the same.
    initial_guess_t : float or integer
                      Initial guess passed to scipy.optimize.minimize. This
                      is recommended to be small.

    Returns
    -------
    unique_number_desended_minima: integer
                                   Total number of unique minima found by
                                   applying multistart.
    unique_number_of_minima_alg: integer
                                 Total number of unique minima found by
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
    set_x_t = np.random.uniform(0, 1, (num_p_t, d))
    t0 = time.time()
    (unique_minimas, unique_number_of_minima_alg,
     func_vals_of_minimas, extra_descents) = mt.metod(f=f_t, g=g_t,
                                                      func_args=func_args_t,
                                                      d=d_t,
                                                      num_points=num_p_t,
                                                      beta=beta_t, m=m_t,
                                                      option=option_t,
                                                      met=met_t,
                                                      no_inequals_to_compare=no_inequals,
                                                      set_x=set_x_t)
    t1 = time.time()
    time_taken_alg = t1-t0
    for minima in unique_minimas:
        position_minimum, norm_with_minima = mt_obj.calc_pos(minima,
                                                             *func_args)
        assert(norm_with_minima < 0.1)
    t0 = time.time()
    store_pos_minima = np.zeros((num_p_t))
    store_start_end_pos = np.zeros((num_p_t))
    store_minima_des = np.zeros((num_p_t, d))
    for j in range(num_p_t):
        x = set_x_t[j, :].reshape(d,)
        iterations_of_sd, its = (mt_alg.apply_sd_until_stopping_criteria
                                 (x, d=d_t, projection=projection_t,
                                  tolerance=tolerance_t, option=option_t,
                                  met=met_t, initial_guess=initial_guess_t,
                                  func_args=func_args_t, f=f_t, g=g_t,
                                  bound_1=0, bound_2=1,
                                  usage='metod_algorithm', relax_sd_it=1))
        store_minima_des[j, :] = iterations_of_sd[its, :]
    t1 = time.time()
    time_taken_des = t1-t0
    for k in range(num_p_t):
        pos_minima, norm_with_minima = mt_obj.calc_pos(store_minima_des[k, :].
                                                       reshape(d,), *func_args)
        assert(norm_with_minima < 0.1)
        store_pos_minima[k] = pos_minima
        pos_start_point, norm_with_minima_sp = mt_obj.calc_pos(set_x_t[k,
                                                               :].reshape(d,),
                                                               *func_args)
        if store_pos_minima[k] != pos_start_point:
            store_start_end_pos[k] = 1
    assert(np.all(store_start_end_pos == 0))
    unique_number_desended_minima = np.unique(store_pos_minima).shape[0]
    return (unique_number_desended_minima, unique_number_of_minima_alg,
            extra_descents, time_taken_alg, time_taken_des)


if __name__ == "__main__":
    d = int(sys.argv[1])
    p = int(sys.argv[2])
    lambda_1 = int(sys.argv[3])
    lambda_2 = int(sys.argv[4])
    f = mt_obj.quad_function
    g = mt_obj.quad_gradient
    m_t = int(sys.argv[5])
    beta_t = float(sys.argv[6])
    met_t = str(sys.argv[7])
    option_t = str(sys.argv[8])
    no_i_t_c_t = str(sys.argv[9])
    num_p_t = 1000
    num_func = 100
    num_workers = 1
    tolerance_t = 0.00001
    projection_t = False
    initial_guess_t = 0.05
    number_minimas_per_func_metod = np.zeros((num_func))
    number_extra_descents_per_func_metod = np.zeros((num_func))
    number_minimas_per_func_multistart = np.zeros((num_func))
    number_extra_descents_per_func = np.zeros((num_func))
    time_metod = np.zeros((num_func))
    time_multistart = np.zeros((num_func))

    for func in tqdm.tqdm(range(num_func)):
        np.random.seed(func * 5)
        store_x0, matrix_test = mt_obj.function_parameters_quad(p, d, lambda_1,
                                                                lambda_2)
        func_args = p, store_x0, matrix_test
        task = metod_numerical_exp_quad(f, g, func_args, d, num_p_t, beta_t,
                                        m_t, option_t, met_t, no_i_t_c_t,
                                        tolerance_t, projection_t,
                                        initial_guess_t)
        result = dask.compute(task, num_workers=num_workers)
        (unique_number_desended_minima, unique_number_of_minima_alg,
         extra_descents, time_taken_alg, time_taken_des) = result[0]
        number_minimas_per_func_metod[func] = unique_number_of_minima_alg
        number_extra_descents_per_func_metod[func] = extra_descents
        number_minimas_per_func_multistart[func] = unique_number_desended_minima
        time_metod[func] = time_taken_alg
        time_multistart[func] = time_taken_des
    table = pd.DataFrame({
                         "number_minimas_per_func_metod":
                         number_minimas_per_func_metod,
                         "number_extra_descents_per_func_metod":
                         number_extra_descents_per_func_metod,
                         "number_minimas_per_func_multistart":
                         number_minimas_per_func_multistart,
                         "time_metod": time_metod,
                         "time_multistart": time_multistart})
    table.to_csv(table.to_csv
                 ('quad_sd_minimize_met_%s_beta_%s_m=%s_d=%s_p=%s_%s.csv' %
                  (met_t, beta_t, m_t, d, p, no_i_t_c_t)))
