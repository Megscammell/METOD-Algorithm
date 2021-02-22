import numpy as np
from numpy import linalg as LA

from metod import metod_algorithm_functions as mt_alg


def apply_sd_until_stopping_criteria(point, d, projection, tolerance, option,
                                     met, initial_guess, func_args, f, g,
                                     bound_1, bound_2, usage, relax_sd_it):
    """
    Apply steepest descent iterations until some stopping condition has
    been met.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            Apply steepest descent iterations to point.
    d : integer
        Size of dimension.
    projection : boolean
                 If projection is True, points are projected back to
                 (bound_1, bound_2). If projection is False, points are
                 kept the same.
    tolerance : integer or float
                Stopping condition for steepest descent iterations.
                Can either apply steepest descent iterations until the norm
                ||g(point, *func_args)|| is less than some tolerance (usage =
                metod_algorithm) or until the total number of steepest descent
                iterations is greater than some tolerance (usage =
                metod_analysis)
    tolerance : float
                Stopping condition for steepest descent iterations. Apply
                steepest descent iterations until the norm
                of g(point, *func_args) is less than some tolerance.
                Also check that the norm of the gradient at a starting point
                is larger than some tolerance.
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
                    Initial guess passed to scipy.optimize.minimize and the
                    upper bound for the bracket interval when using the
                    'Brent' or 'Golden' method for
                    scipy.optimize.minimize_scalar. This
                    is recommended to be small.
    func_args : tuple
                Arguments passed to f and g.
    f : objective function.

        `f(point, *func_args) -> float`

        where point` is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the function value.
    g : gradient of objective function.

       `g(point, *func_args) -> 1-D array with shape (d, )`

        where `point` is a 1-D array with shape (d, ) and func_args is
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
                  Multiply the step size by a small constant in [0, 2], to 
                  obtain a new step size for steepest descent iterations. This 
                  process is known as relaxed steepest descent [1].

    Returns
    -------
    sd_iterations : 2-D array with shape (its, d)
                    Each steepest descent iteration is stored in each row of
                    sd_iterations.
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
                raise ValueError('Number of iterations has exceeded 200.'
                                 ' Try another method and/or option.')
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
