import dask
import numpy as np
import tqdm
import time
import sys
import pandas as pd

import metod_alg as mt
from metod_alg import objective_functions as mt_obj
from metod_alg import metod_analysis as mt_ays
from metod_alg import check_metod_class as prev_mt_alg


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


def check_classification_points_metod(classification_points,
                                      unique_minimizers_metod,
                                      check_func,
                                      func_args):
    """
    Matches the disovered local minimizers from METOD (found either by local
    descent or early termination of descents) with true local minimizers.

    Parameters
    ----------
    classification_points : 1-d array with shape (num_p,)
                            Array containing minimizer index number in which a
                            point belongs to (found either by local descent or
                            early termination of descents within the METOD
                            algorithm).
    unique_minimizers_metod : list
                              Unique minimizers found by the METOD algorithm.
    func_args : integer
                Arguments passed to f and g.
    """

    class_store_x0 = np.zeros((len(classification_points)))
    for j in range(len(classification_points)):
        class_store_x0[j] = (check_func
                             (unique_minimizers_metod[int(classification_points[j])],
                              *func_args))
    return class_store_x0


def metod_numerical_exp_quad_new(f, g, func_args, d,
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
                           number of gradient evaluations used during the warm
                           up period.
        grad_evals_mult : 1-D array with shape (num_p,)
                          Number of gradient evaluations used to reach a local
                          minimizer for each starting point when using
                          Multistart.
        store_grad_norms : 1-D array with shape (num_p,)
                           Euclidean norm of the gradient at each starting
                           point.
        starting_points : 2-D array with shape (num_p, d)
                          Each row contains each starting point used by METOD
                          and Multistart.
        missed_minimizers : float
                            Total number of minimizers missed from [3, Eq. 9]
                            holding for points belonging to regions
                            of attraction of different local minimizers.
        total_checks : float
                       Total number of times condition [3, Eq. 9]
                       holds.

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
                           number of gradient evaluations used during the warm
                           up period.
        store_grad_norms : 1-D array with shape (num_p,)
                           Euclidean norm of the gradient at each starting
                           point.
        starting_points : 2-D array with shape (num_p, d)
                          Each row contains each starting point used by METOD.
        missed_minimizers : float
                            Total number of minimizers missed from [3, Eq. 9]
                            holding for points belonging to regions
                            of attraction of different local minimizers.
        total_checks : float
                       Total number of times condition [3, Eq. 9]
                       holds.

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
     grad_evals_metod,
     missed_minimizers,
     total_checks) = (prev_mt_alg.metod_without_class(
                      f, g, func_args, d, num_p, beta,
                      tolerance, projection, const, m,
                      option, met, initial_guess,
                      set_x, bounds_set_x, relax_sd_it))
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
                starting_points,
                missed_minimizers,
                total_checks)

    else:
        return (unique_number_of_minimizers_metod,
                extra_descents,
                time_taken_metod,
                np.min(func_vals_of_minimizers_metod),
                grad_evals_metod,
                store_grad_norms,
                starting_points,
                missed_minimizers,
                total_checks)


def metod_numerical_exp_quad_original(f, g, func_args, d,
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
                           number of gradient evaluations used during the warm
                           up period.
        grad_evals_mult : 1-D array with shape (num_p,)
                          Number of gradient evaluations used to reach a local
                          minimizer for each starting point when using
                          Multistart.
        store_grad_norms : 1-D array with shape (num_p,)
                           Euclidean norm of the gradient at each starting
                           point.
        starting_points : 2-D array with shape (num_p, d)
                          Each row contains each starting point used by METOD
                          and Multistart.
        prop_class_sd_metod : float
                              Proportion of times the classification of a point
                              using the METOD algorithm is different to the
                              true classification using Multistart.
        count_gr_2 : integer
                     Number of times inequality [3, Eq. 9]  is satisfied for
                     more than one region of attraction.
        missed_minimizers : float
                            Total number of minimizers missed from [3, Eq. 9]
                            holding for points belonging to regions
                            of attraction of different local minimizers.
        total_checks : float
                       Total number of times condition [3, Eq. 9]
                       holds.
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
                           number of gradient evaluations used during the warm
                           up period.
        store_grad_norms : 1-D array with shape (num_p,)
                           Euclidean norm of the gradient at each starting
                           point.
        starting_points : 2-D array with shape (num_p, d)
                          Each row contains each starting point used by METOD.
        count_gr_2 : integer
                     Number of times inequality [3, Eq. 9]  is satisfied for
                     more than one region of attraction.
        missed_minimizers : float
                            Total number of minimizers missed from [3, Eq. 9]
                            holding for points belonging to regions
                            of attraction of different local minimizers.
        total_checks : float
                       Total number of times condition [3, Eq. 9]
                       holds.
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
     grad_evals_metod,
     classification_points,
     count_gr_2, missed_minimizers,
     total_checks) = prev_mt_alg.metod_class(f, g, func_args, d, num_p, beta,
                                             tolerance, projection, const, m,
                                             option, met, initial_guess,
                                             set_x, bounds_set_x, relax_sd_it)
    t1 = time.time()
    time_taken_metod = t1-t0
    mt_obj.check_unique_minimizers(unique_minimizers_metod,
                                   unique_number_of_minimizers_metod,
                                   check_func, func_args)

    class_store_x0 = check_classification_points_metod(classification_points,
                                                       unique_minimizers_metod,
                                                       check_func,
                                                       func_args)

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

        prop_class_sd_metod = (prev_mt_alg.check_classification_sd_metod
                               (store_minimizer_des, class_store_x0,
                                check_func, func_args))

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
                starting_points,
                prop_class_sd_metod,
                count_gr_2, missed_minimizers,
                total_checks)

    else:
        return (unique_number_of_minimizers_metod,
                extra_descents,
                time_taken_metod,
                np.min(func_vals_of_minimizers_metod),
                grad_evals_metod,
                store_grad_norms,
                starting_points,
                count_gr_2, missed_minimizers,
                total_checks)


@dask.delayed
def all_functions_metod(f, g, p, lambda_1, lambda_2, d,
                        num_p, beta, tolerance, projection,
                        const, m, option, met, initial_guess,
                        set_x, bounds_set_x, relax_sd_it, sd_its,
                        check_func, num_func, random_seed, type_func):
    """
    Generate each function required for the METOD algorithm and save outputs
    to csv files.

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
    p : integer
        Number of local minima.
    lambda_1 : integer
               Smallest eigenvalue of diagonal matrix.
    lambda_2 : integer
               Largest eigenvalue of diagonal matrix.
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
    num_func : integer
               Number of random functions to generate.
    random_seed : integer
                  Value to initialize pseudo-random number generator.
    type_func : string
                Indicate version of objective function.
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
    number_minimizers_per_func_metod = np.zeros((num_func))
    number_extra_descents_per_func_metod = np.zeros((num_func))
    time_metod = np.zeros((num_func))
    func_val_metod = np.zeros((num_func))
    store_grad_norms = np.zeros((num_func, num_p))
    store_grad_evals_metod = np.zeros((num_func, num_p))
    store_count_gr_2 = np.zeros((num_func))
    store_missed_minimizers = np.zeros((num_func))
    store_total_checks = np.zeros((num_func))
    if sd_its == True:
        number_minimizers_per_func_multistart = np.zeros((num_func))
        time_multistart = np.zeros((num_func))
        func_val_multistart = np.zeros((num_func))
        store_grad_evals_mult = np.zeros((num_func, num_p))
        store_prop_class_sd_metod = np.zeros((num_func))

    np.random.seed(random_seed)
    for func in tqdm.tqdm(range(num_func)):
        store_A = np.zeros((p, d, d))
        store_x0 = np.zeros((p, d))
        store_rotation = np.zeros((p, d, d))
        for i in range(p):
            diag_vals = np.zeros(d)
            diag_vals[:2] = np.array([lambda_1, lambda_2])
            diag_vals[2:] = np.random.uniform(lambda_1 + 1,
                                              lambda_2 - 1, (d - 2))
            store_A[i] = np.diag(diag_vals)
            store_x0[i] = np.random.uniform(0, 1, (d,))
            store_rotation[i] = mt_obj.calculate_rotation_matrix(d, 3)
        matrix_test = (np.transpose(store_rotation, (0, 2, 1)) @ store_A @
                       store_rotation)
        func_args = (p, store_x0, matrix_test)
        if sd_its == True:
            if type_func == 'old':
                (number_minimizers_per_func_multistart[func],
                 number_minimizers_per_func_metod[func],
                 number_extra_descents_per_func_metod[func],
                 time_metod[func],
                 time_multistart[func],
                 func_val_metod[func],
                 func_val_multistart[func],
                 store_grad_evals_metod[func],
                 store_grad_evals_mult[func],
                 store_grad_norms[func],
                 starting_points,
                 store_prop_class_sd_metod[func],
                 store_count_gr_2[func],
                 store_missed_minimizers[func],
                 store_total_checks[func]) = (metod_numerical_exp_quad_original
                                              (f, g, func_args, d,
                                               num_p, beta, tolerance,
                                               projection, const, m, option,
                                               met, initial_guess, set_x,
                                               bounds_set_x, relax_sd_it,
                                               sd_its, check_func))
            else:
                (number_minimizers_per_func_multistart[func],
                 number_minimizers_per_func_metod[func],
                 number_extra_descents_per_func_metod[func],
                 time_metod[func],
                 time_multistart[func],
                 func_val_metod[func],
                 func_val_multistart[func],
                 store_grad_evals_metod[func],
                 store_grad_evals_mult[func],
                 store_grad_norms[func],
                 starting_points,
                 store_missed_minimizers[func],
                 store_total_checks[func]) = (metod_numerical_exp_quad_new
                                              (f, g, func_args, d,
                                               num_p, beta, tolerance,
                                               projection, const, m, option,
                                               met, initial_guess, set_x,
                                               bounds_set_x, relax_sd_it,
                                               sd_its, check_func))
            if func == 0:
                store_starting_points = np.array(starting_points)
            else:
                store_starting_points = np.vstack([store_starting_points,
                                                   np.array(starting_points)])
        else:
            if type_func == 'old':
                (number_minimizers_per_func_metod[func],
                 number_extra_descents_per_func_metod[func],
                 time_metod[func],
                 func_val_metod[func],
                 store_grad_evals_metod[func],
                 store_grad_norms[func],
                 starting_points,
                 store_count_gr_2[func],
                 store_missed_minimizers[func],
                 store_total_checks[func]) = (metod_numerical_exp_quad_original
                                              (f, g, func_args, d,
                                               num_p, beta, tolerance,
                                               projection, const, m, option,
                                               met, initial_guess, set_x,
                                               bounds_set_x, relax_sd_it,
                                               sd_its, check_func))
            else:
                (number_minimizers_per_func_metod[func],
                 number_extra_descents_per_func_metod[func],
                 time_metod[func],
                 func_val_metod[func],
                 store_grad_evals_metod[func],
                 store_grad_norms[func],
                 starting_points,
                 store_missed_minimizers[func],
                 store_total_checks[func]) = (metod_numerical_exp_quad_new
                                              (f, g, func_args, d,
                                               num_p, beta, tolerance,
                                               projection, const, m, option,
                                               met, initial_guess, set_x,
                                               bounds_set_x, relax_sd_it,
                                               sd_its, check_func))
            if func == 0:
                store_starting_points = np.array(starting_points)
            else:
                store_starting_points = np.vstack([store_starting_points,
                                                   np.array(starting_points)])

    np.savetxt('quad_grad_norm_beta_%s_m=%s_d=%s'
               '_p=%s_%s_%s_%s_%s_%s.csv' %
               (beta, m, d, p, set_x, num_p, option[0], initial_guess,
                type_func),
               store_grad_norms,
               delimiter=',')

    np.savetxt('quad_grad_evals_metod_beta_%s_m=%s_d=%s'
               'p=%s_%s_%s_%s_%s_%s.csv' %
               (beta, m, d, p, set_x, num_p, option[0], initial_guess,
                type_func),
               store_grad_evals_metod,
               delimiter=',')

    if sd_its == True:
        if type_func == 'old':
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
                                "min_func_val_multistart": func_val_multistart,
                                "prop_class": store_prop_class_sd_metod,
                                "greater_than_one_region": store_count_gr_2,
                                "total_times_minimizer_missed":
                                store_missed_minimizers,
                                "total_no_times_inequals_sat":
                                store_total_checks})
        else:
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
                                "min_func_val_multistart": func_val_multistart,
                                "total_times_minimizer_missed":
                                store_missed_minimizers,
                                "total_no_times_inequals_sat":
                                store_total_checks})
        table.to_csv(table.to_csv
                     ('quad_sd_metod_beta_%s_m=%s_d=%s_p=%s'
                      '_%s_%s_%s_%s_%s.csv' %
                      (beta, m, d, p, set_x,
                       num_p, option[0], initial_guess, type_func)))

        np.savetxt('quad_grad_evals_mult_beta_%s_m=%s_d=%s'
                   'p=%s_%s_%s_%s_%s_%s.csv' %
                   (beta, m, d, p, set_x, num_p, option[0], initial_guess,
                    type_func),
                   store_grad_evals_mult,
                   delimiter=',')
        np.savetxt('quad_sd_start_p_beta_%s_m=%s_d=%s'
                   '_p=%s_%s_%s_%s_%s_%s.csv' %
                   (beta, m, d, p, set_x, num_p, option[0], initial_guess,
                    type_func),
                   store_starting_points,
                   delimiter=',')
    else:
        if type_func == 'old':
            table = pd.DataFrame({
                                "number_minimizers_per_func_metod":
                                number_minimizers_per_func_metod,
                                "number_extra_descents_per_func_metod":
                                number_extra_descents_per_func_metod,
                                "time_metod": time_metod,
                                "min_func_val_metod": func_val_metod,
                                "greater_than_one_region": store_count_gr_2,
                                "total_times_minimizer_missed":
                                store_missed_minimizers,
                                "total_no_times_inequals_sat":
                                store_total_checks})
        else:
            table = pd.DataFrame({
                                "number_minimizers_per_func_metod":
                                number_minimizers_per_func_metod,
                                "number_extra_descents_per_func_metod":
                                number_extra_descents_per_func_metod,
                                "time_metod": time_metod,
                                "min_func_val_metod": func_val_metod,
                                "total_times_minimizer_missed":
                                store_missed_minimizers,
                                "total_no_times_inequals_sat":
                                store_total_checks})
        table.to_csv(table.to_csv
                     ('quad_metod_beta_%s_m=%s_d=%s_p=%s'
                      '_%s_%s_%s_%s_%s.csv' %
                      (beta, m, d, p, set_x,
                       num_p, option[0], initial_guess, type_func)))

        np.savetxt('quad_start_p_beta_%s_m=%s_d=%s'
                   '_p=%s_%s_%s_%s_%s_%s.csv' %
                   (beta, m, d, p, set_x, num_p, option[0], initial_guess,
                    type_func),
                   store_starting_points,
                   delimiter=',')


if __name__ == "__main__":
    """
    To obtain the same results as in [1] or in thesis, set optional
    input parameters to the following:

    d : set the dimension to either 50, 100 or 200.
    num_p : 1000.
    beta : set beta to be either 0.005, 0.01, 0.05 or 0.1.
    m : set warm up period to be either 1, 2 or 3.
    set_x : 'random'.
    sd_its : True.
    p : 50.
    option : either option = 'minimize_scalar' to obtain results in thesis or
             option = 'minimize' to obtain results in [1].
    met : either met = 'Brent' to obtain results in thesis or
          met = 'Nelder-Mead' to obtain results in [1].
    initial_guess : set initial_guess=0.05 for results in [1]. Otherwise,
                    set initial_guess=0.005 for results in thesis.
    random_seed : either random_seed = 1997 when d = 50 or
                  random_seed = 121 when d = 100 or d = 200.
    type_func : either type_func = 'new' to obtain results in thesis or
                type_func = 'old' to obtain results in [1].
    References
    ----------
    1) Zilinskas, A., Gillard, J., Scammell, M., Zhigljavsky, A.: Multistart
       with early termination of descents. Journal of Global Optimization pp.
       1–16 (2019)
    """
    d = int(sys.argv[1])
    num_p = int(sys.argv[2])
    beta = float(sys.argv[3])
    m = int(sys.argv[4])
    set_x = str(sys.argv[5])
    sd_its = eval(sys.argv[6])
    p = int(sys.argv[7])
    option = str(sys.argv[8])
    met = str(sys.argv[9])
    initial_guess = float(sys.argv[10])
    random_seed = int(sys.argv[11])
    type_func = str(sys.argv[12])
    type_results = str(sys.argv[13])

    if type_func == 'old':
        if type_results == 'paper':
            f = prev_mt_alg.quad_function
            g = prev_mt_alg.quad_gradient
            check_func = prev_mt_alg.calc_minimizer_quad
        if type_results == 'thesis':
            f = mt_obj.several_quad_function
            g = mt_obj.several_quad_gradient
            check_func = mt_obj.calc_minimizer_sev_quad
    elif type_func == 'new':
        f = mt_obj.several_quad_function
        g = mt_obj.several_quad_gradient
        check_func = mt_obj.calc_minimizer_sev_quad

    tolerance = 0.001
    projection = False
    const = 0.1
    bounds_set_x = (0, 1)
    relax_sd_it = 1

    lambda_1 = 1
    lambda_2 = 10
    num_func = 100

    num_workers = 1

    task = all_functions_metod(f, g, p, lambda_1, lambda_2, d,
                               num_p, beta, tolerance, projection,
                               const, m, option, met, initial_guess,
                               set_x, bounds_set_x, relax_sd_it, sd_its,
                               check_func, num_func, random_seed, type_func)
    result = dask.compute(task, num_workers=num_workers)
