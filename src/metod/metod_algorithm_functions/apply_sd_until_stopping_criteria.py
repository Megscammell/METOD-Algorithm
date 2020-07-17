import numpy as np
from numpy import linalg as LA

import metod.metod_algorithm_functions as mt_alg


def apply_sd_until_stopping_criteria(point, d, projection, tolerance, option,
                                     met, initial_guess, func_args, f, g,
                                     bound_1, bound_2, usage, relax_sd_it):
    """Apply steepest descent iterations until the euclidean
    norm of the gradient is smaller than tolerance.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point to apply steepest descent iterations.
    d : integer
        Size of dimension.
    projection : boolean
                 If projection is True, this projects points back to
                 (bound_1, bound_2). If projection is False, points are
                 kept the same.
    tolerance : integer or float (optional)
                Stopping condition for steepest descent iterations.
                Can either apply steepest descent iterations until the norm of
                g(point, *func_args) is less than some tolerance (usage =
                metod_algorithm) or until the total number of steepest descent
                iterations is greater than some tolerance  (usage =
                metod_analysis)
    tolerance : float
                Stopping condition for steepest descent iterations.
    option : string
             Choose from 'minimize' or 'minimize_scalar'. For more
             information about each option see
             https://docs.scipy.org/doc/scipy/reference/optimize.html.
    met : string
         Choose method for option. For more information see
         - https://docs.scipy.org/doc/scipy/reference/generated/
         scipy.optimize.minimize.html#scipy.optimize.minimize
         - https://docs.scipy.org/doc/scipy/reference/generated/
         scipy.optimize.minimize_scalar.html#scipy.optimize.minimize_scalar
    initial_guess : float or integer
                    Initial guess passed to scipy.optimize.minimize. This
                    is recommended to be small.
    func_args : tuple
                Arguments passed to f and g.
    f : objective function.

        ``f(point, *func_args) -> float``

        where ``point`` is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the function value.
    g : gradient of objective function.

       ``g(point, *func_args) -> 1-D array with shape (d, )``

        where ``point`` is a 1-D array with shape (d, ) and func_args is
        a tuple of arguments needed to compute the gradient.
    bounds_1 : integer
               Lower bound used for projection.
    bounds_2 : integer
               Upper bound used for projection.
    usage : string
            Used to decide stopping criterion for steepest descent
            iterations. Should be either usage == 'metod_algorithm' or
            usage == 'metod_analysis'.
    relax_sd_it : float or integer
                  Small constant in [0, 2] to multiply the step size by for a
                  steepest descent iteration. This process is known as relaxed
                  steepest descent [1].


    Returns
    -------
    sd_iterations : 2-D array with shape (its, d)
                    Each row of sd_iterations contains a new point
                    after a steepest descent iteration.
    its: integer
         Total number of steepest descent iterations.

    References
    ----------
    1) Raydan, M., Svaiter, B.F.: Relaxed steepest descent and
       cauchy-barzilai- borwein method. Computational Optimization and
       Applications 21(2), 155–167 (2002)

    """
    its = 0
    sd_iterations = np.zeros((1, d))
    sd_iterations[0, :] = point.reshape(1, d)
    x_iteration = mt_alg.sd_iteration(point, projection, option, met,
                                      initial_guess, func_args, f, g, bound_1,
                                      bound_2, relax_sd_it)
    sd_iterations = np.vstack([sd_iterations, x_iteration.reshape((1, d))])
    its += 1
    point = x_iteration

    if usage == 'metod_algorithm':
        while LA.norm(g(point, *func_args)) >= tolerance:
            x_iteration = mt_alg.sd_iteration(point, projection, option, met,
                                              initial_guess, func_args, f, g,
                                              bound_1, bound_2, relax_sd_it)
            sd_iterations = np.vstack([sd_iterations, x_iteration.reshape
                                      ((1, d))])
            its += 1
            point = x_iteration
            if its > 200:
                break
        if its >= 200:
            raise ValueError('Number of iterations has exceeded 200.'
                             ' Try anothermethod and/or option.')
    elif usage == 'metod_analysis':
        while its < tolerance:
            x_iteration = mt_alg.sd_iteration(point, projection, option, met,
                                              initial_guess, func_args, f, g,
                                              bound_1, bound_2, relax_sd_it)
            sd_iterations = np.vstack([sd_iterations, x_iteration.reshape
                                      ((1, d))])
            its += 1
            point = x_iteration
    return sd_iterations, its
