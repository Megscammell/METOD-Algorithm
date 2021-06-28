import dask
import numpy as np
import tqdm
import time
import sys
import pandas as pd

import metod_alg as mt
from metod_alg import objective_functions as mt_obj
from metod_alg import metod_analysis as mt_ays


def check_sp_fp(starting_points, store_minimizer_des, num_p, func_args):
    """
    Checks that the local minimizer at a starting point is the same as the
    local minimizer at the final point. 

    Parameters
    ----------
    starting_points : 2-d array with shape (num_p, d)
                      Array containing starting points.
    store_minimizer_des : list
                          List containing local minimizers, found after
                          applying local descent at each starting point.
    num_p : integer
            Number of starting points.
    func_args : tuple
                Function arguments passed to f and g.
    """

    count = 0
    for k in range(num_p):
        pos_minimizer = (mt_ays.calc_minimizer_sev_quad_no_dist_check
                         (store_minimizer_des[k], *func_args))
        pos_sp = (mt_ays.calc_minimizer_sev_quad_no_dist_check
                  (starting_points[k], *func_args))
        if pos_minimizer != pos_sp:
            count += 1
    assert(count == 0)


@dask.delayed
def metod_numerical_exp_quad(f, g, func_args, d,
                            num_p, beta, tolerance, projection,
                            const, m, option, met, initial_guess,
                            set_x, bounds_set_x, relax_sd_it, sd_its,
                            check_func):
    """
    Apply the METOD algorithm with specified parameters. If sd_its =
    True, multistart will also be applied with the same starting points
    as METOD.

    Parameters
    ----------
    f : Objective function.

        ``f(x, *func_args) -> float``

        where ``x`` is a 1-D array with shape (d, ) and func_args is a
        tuple of arguments needed to compute the function value.
    g : Gradient.

         ``g(x, *func_args) -> 1-D array with shape (d, )``

          where ``x`` is a 1-D array with shape (d, ) and func_args is a
          tuple of arguments needed to compute the gradient.
    func_args : tuple
                Arguments passed to f and g.
    d : integer
        Size of dimension.
    num_p : integer
            Number of random points generated.
    beta : float or integer
           Small constant step size to compute the partner points.
    tolerance: float
               Stopping condition for steepest descent iterations. Apply
               steepest descent iterations until the norm
               of g(point, *func_args) is less than some tolerance.
               Also check that the norm of the gradient at a starting point
               is larger than some tolerance.
    projection : boolean
                 If projection is True, points are projected back to
                 bounds_set_x. If projection is False, points are
                 not projected.
    const : float or integer
            In order to classify a point as a new local minimizer, the
            euclidean distance between the point and all other discovered local
            minimizers must be larger than const.
    m : integer
        Number of iterations of steepest descent to apply to a point
        before making decision on terminating descents.
    option : string
             Choose from 'minimize', 'minimize_scalar' or
             'forward_backward_tracking'. For more
             information on 'minimize' or 'minimize_scalar' see
             https://docs.scipy.org/doc/scipy/reference/optimize.html.
    met : string
           If option = 'minimize' or option = 'minimize_scalar', choose
           appropiate method. For more information see
           - https://docs.scipy.org/doc/scipy/reference/generated/
           scipy.optimize.minimize.html#scipy.optimize.minimize
           - https://docs.scipy.org/doc/scipy/reference/generated/
           scipy.optimize.minimize_scalar.html#scipy.optimize.minimize_scalar.
           If option = 'forward_backward_tracking', then met does not need to
           be specified.
    initial_guess : float or integer
                    Initial guess passed to scipy.optimize.minimize and the
                    upper bound for the bracket interval when using the
                    'Brent' or 'Golden' method for
                    scipy.optimize.minimize_scalar. Also the initial guess
                    for option='forward_backward_tracking'. This
                    is recommended to be small.
    set_x : string
            If set_x = 'random', random starting points
            are generated for the METOD algorithm. If set_x = 'sobol',
            then a numpy.array with shape (num points * 2, d) of Sobol
            sequence samples are generated using SALib [1], which are
            randomly shuffled and used as starting points for the METOD
            algorithm.
    bounds_set_x : tuple
                   Bounds used for set_x = 'random', set_x = 'sobol' and
                   also for projection = True.
    relax_sd_it : float or integer
                  Multiply the step size by a small constant in [0, 2], to
                  obtain a new step size for steepest descent iterations. This
                  process is known as relaxed steepest descent [2].
    sd_its : boolean
             If sd_its = True, multistart is applied with the same starting
             points as METOD. If sd_its = False, only METOD is applied.
    check_func : function
                 A function which checks the local minimizers obtained by
                 METOD or multistart with the true local minimizers of the
                 objective function.
            
    Returns
    -------
    if sd_its == True:
        unique_number_of_minimizers_mult: integer
                                          Total number of unique minimizers
                                          found by applying multistart.
        unique_number_of_minimizers_metod: integer
                                           Total number of unique minimizers
                                           found by applying METOD.
        extra_descents : integer
                         Number of excessive descents. Occurs when
                         [3, Eq. 9] does not hold for trajectories
                         that belong to the region of attraction
                         of the same local minimizer.
        time_taken_metod: float
                          Amount of time (in seconds) the METOD algorithm
                          takes.
        time_taken_des: float
                        Amount of time (in seconds) multistart takes.
        np.min(func_vals_of_minimizers_metod) : float
                                                Minimum function value found
                                                using METOD.
        np.min(store_func_vals_mult) : float
                                       Minimum function value found using
                                       multistart.
        grad_evals_metod : 1-D array with shape (num_p,)
                           Number of gradient evaluations used either to reach
                           a local minimizer if [3, Eq. 9] does not hold or the 
                           number of gradient evaluations used during the warm up
                           period.
        grad_evals_mult : 1-D array with shape (num_p,)
                          Number of gradient evaluations used to reach a local
                          minimizer for each starting point when using
                          Multistart.
        store_grad_norms : 1-D array with shape (num_p,)
                           Euclidean norm of the gradient at each starting point.
        starting_points : 2-D array with shape (num_p, d)
                          Each row contains each starting point used by METOD
                          and Multistart.

    else:
        unique_number_of_minimizers_metod: integer
                                           Total number of unique minimizers
                                           found by applying METOD.
        extra_descents : integer
                         Number of excessive descents. Occurs when
                         [3, Eq. 9] does not hold for trajectories
                         that belong to the region of attraction
                         of the same local minimizer.
        time_taken_metod: float
                          Amount of time (in seconds) the METOD algorithm
                          takes.
        np.min(func_vals_of_minimizers_metod) : float
                                                Minimum function value found
                                                using METOD.
        grad_evals_metod : 1-D array with shape (num_p,)
                           Number of gradient evaluations used either to reach
                           a local minimizer if [3, Eq. 9] does not hold or the 
                           number of gradient evaluations used during the warm up
                           period.
        store_grad_norms : 1-D array with shape (num_p,)
                           Euclidean norm of the gradient at each starting point.
        starting_points : 2-D array with shape (num_p, d)
                          Each row contains each starting point used by METOD.


    References
    ----------
    1) Herman et al, (2017), SALib: An open-source Python library for 
       Sensitivity Analysis, Journal of Open Source Software, 2(9), 97, doi:10.
       21105/joss.00097
    2) Raydan, M., Svaiter, B.F.: Relaxed steepest descent and
       cauchy-barzilai- borwein method. Computational Optimization and
       Applications 21(2), 155–167 (2002)
    3) Zilinskas, A., Gillard, J., Scammell, M., Zhigljavsky, A.: Multistart
       with early termination of descents. Journal of Global Optimization pp.
       1–16 (2019)

    """

    t0 = time.time()
    (unique_minimizers_metod,
     unique_number_of_minimizers_metod,
     func_vals_of_minimizers_metod,
     extra_descents,
     starting_points,
     grad_evals_metod) = mt.metod(f, g, func_args, d, num_p, beta,
                                 tolerance, projection, const, m,
                                 option, met, initial_guess,
                                 set_x, bounds_set_x, relax_sd_it)
    t1 = time.time()
    time_taken_metod = t1-t0

    mt_obj.check_unique_minimizers(unique_minimizers_metod,
                                   unique_number_of_minimizers_metod,
                                   check_func, func_args)

    store_grad_norms = np.zeros((num_p))
    for j in range(num_p):
        store_grad_norms[j] = np.linalg.norm(g(starting_points[j], *func_args))

    if sd_its == True:
        (unique_minimizers_mult,
         unique_number_of_minimizers_mult,
         store_func_vals_mult,
         time_taken_des,
         store_minimizer_des,
         grad_evals_mult) = mt.multistart(f, g, func_args, d, starting_points,
                                 num_p, tolerance, projection, const,
                                 option, met, initial_guess,
                                 bounds_set_x, relax_sd_it)

        mt_obj.check_unique_minimizers(store_minimizer_des,
                                       unique_number_of_minimizers_mult,
                                       check_func, func_args)

        mt_obj.check_minimizers_mult_metod(unique_minimizers_metod,
                                           unique_minimizers_mult)

        check_sp_fp(starting_points, store_minimizer_des, num_p, func_args)

        return (unique_number_of_minimizers_mult,
                unique_number_of_minimizers_metod,
                extra_descents,
                time_taken_metod,
                time_taken_des,
                np.min(func_vals_of_minimizers_metod),
                np.min(store_func_vals_mult),
                grad_evals_metod,
                grad_evals_mult,
                store_grad_norms,
                starting_points)

    else:
        return (unique_number_of_minimizers_metod,
                extra_descents,
                time_taken_metod,
                np.min(func_vals_of_minimizers_metod),
                grad_evals_metod,
                store_grad_norms,
                starting_points)


if __name__ == "__main__":
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    check_func = mt_obj.calc_minimizer_sev_quad

    d = int(sys.argv[1])
    num_p = int(sys.argv[2])
    beta = float(sys.argv[3])
    m = int(sys.argv[4])
    set_x = str(sys.argv[5])
    sd_its = eval(sys.argv[6])
    p = int(sys.argv[7])
    option = str(sys.argv[8])
    initial_guess = float(sys.argv[9])

    tolerance = 0.01
    projection = False
    const = 0.1
    if option == 'minimize_scalar':
        met = 'Brent'
    elif option == 'forward_backward_tracking':
        met = 'None'
    else:
        raise ValueError('Incorrect option.')
    bounds_set_x = (0, 1)
    relax_sd_it = 1

    lambda_1 = 1
    lambda_2 = 10
    num_func = 100
    num_workers = 1

    number_minimizers_per_func_metod = np.zeros((num_func))
    number_extra_descents_per_func_metod = np.zeros((num_func))
    number_extra_descents_per_func = np.zeros((num_func))
    time_metod = np.zeros((num_func))
    func_val_metod = np.zeros((num_func))
    store_grad_norms = np.zeros((num_func, num_p))
    store_grad_evals_metod = np.zeros((num_func, num_p))
    if sd_its == True:
        number_minimizers_per_func_multistart = np.zeros((num_func))
        time_multistart = np.zeros((num_func))
        func_val_multistart = np.zeros((num_func))
        store_grad_evals_mult = np.zeros((num_func, num_p))

    for func in tqdm.tqdm(range(num_func)):
        np.random.seed(func * 5)
        store_x0, matrix_test = (mt_obj.function_parameters_several_quad
                                 (p, d, lambda_1, lambda_2))
        func_args = (p, store_x0, matrix_test)
        task = metod_numerical_exp_quad(f, g, func_args, d,
                                        num_p, beta, tolerance, projection,
                                        const, m, option, met, initial_guess,
                                        set_x, bounds_set_x, relax_sd_it, sd_its,
                                        check_func)
        result = dask.compute(task, num_workers=num_workers)

        if sd_its == True:
            (number_minimizers_per_func_multistart[func],
             number_minimizers_per_func_metod[func],
             number_extra_descents_per_func_metod[func],
             time_metod[func],
             time_multistart[func] ,
             func_val_metod[func],
             func_val_multistart[func],
             store_grad_evals_metod[func],
             store_grad_evals_mult[func],
             store_grad_norms[func],
             starting_points) = result[0]
            if func == 0:
                store_starting_points = np.array(starting_points)
            else:
                store_starting_points = np.vstack([store_starting_points,
                                                 np.array(starting_points)])
        else:
            (number_minimizers_per_func_metod[func],
             number_extra_descents_per_func_metod[func],
             time_metod[func],
             func_val_metod[func],
             store_grad_evals_metod[func],
             store_grad_norms[func],
             starting_points) = result[0]
            if func == 0:
                store_starting_points = np.array(starting_points)
            else:
                store_starting_points = np.vstack([store_starting_points,
                                                 np.array(starting_points)])

    np.savetxt('quad_grad_norm_beta_%s_m=%s_d=%s'
                '_p=%s_%s_%s_%s_%s.csv' %
                (beta, m, d, p, set_x, num_p, option[0], initial_guess),
                 store_grad_norms,
                 delimiter=',')

    np.savetxt('quad_grad_evals_metod_beta_%s_m=%s_d=%s'
                'p=%s_%s_%s_%s_%s.csv' %
                (beta, m, d, p, set_x, num_p, option[0], initial_guess),
                 store_grad_evals_metod,
                 delimiter=',')

    if sd_its == True:        
        table = pd.DataFrame({
                            "number_minimizers_per_func_metod":
                            number_minimizers_per_func_metod,
                            "number_extra_descents_per_func_metod":
                            number_extra_descents_per_func_metod,
                            "number_minimizers_per_func_multistart":
                            number_minimizers_per_func_multistart,
                            "time_metod": time_metod,
                            "time_multistart": time_multistart,
                            "min_func_val_metod": func_val_metod,
                            "min_func_val_multistart": func_val_multistart})
        table.to_csv(table.to_csv
                    ('quad_sd_metod_beta_%s_m=%s_d=%s_p=%s'
                    '_%s_%s_%s_%s.csv' %
                    (beta, m, d, p, set_x,
                     num_p, option[0], initial_guess)))
        
        np.savetxt('quad_grad_evals_mult_beta_%s_m=%s_d=%s'
                    'p=%s_%s_%s_%s_%s.csv' %
                    (beta, m, d, p, set_x, num_p, option[0], initial_guess),
                     store_grad_evals_mult,
                     delimiter=',')
        np.savetxt('quad_sd_start_p_beta_%s_m=%s_d=%s'
                    '_p=%s_%s_%s_%s_%s.csv' %
                    (beta, m, d, p, set_x, num_p, option[0], initial_guess),
                    store_starting_points,
                    delimiter=',')
    else:
        table = pd.DataFrame({
                            "number_minimizers_per_func_metod":
                            number_minimizers_per_func_metod,
                            "number_extra_descents_per_func_metod":
                            number_extra_descents_per_func_metod,
                            "time_metod": time_metod,
                            "min_func_val_metod": func_val_metod})
        table.to_csv(table.to_csv
                    ('quad_metod_beta_%s_m=%s_d=%s_p=%s'
                    '_%s_%s_%s_%s.csv' %
                    (beta, m, d, p, set_x,
                     num_p, option[0], initial_guess)))

        np.savetxt('quad_start_p_beta_%s_m=%s_d=%s'
                    '_p=%s_%s_%s_%s_%s.csv' %
                    (beta, m, d, p, set_x, num_p, option[0], initial_guess),
                    store_starting_points,
                    delimiter=',')
