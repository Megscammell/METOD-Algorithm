import numpy as np

import metod.metod_algorithm as mt_alg


def apply_sd_until_warm_up(point, d, m, beta, projection, option, met,
                           initial_guess, func_args, f, g, bound_1, bound_2):
    """Computes m iterations of steepest descent and the corresponding
    partner points

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point to apply steepest descent iterations.
    d : integer
        Size of dimension.
    m : integer
        Number of iterations of steepest descent to apply to point
        x before making decision on terminating descents.
    beta : float or integer
           Small constant step size to compute the partner points.
    projection : boolean
                 If projection is True, this projects points back to
                 (bound_1, bound_2). If projection is False, points are
                 kept the same.
    option : string (optional)
             Choose from 'minimize' or 'minimize_scalar'. For more
             information about each option see
             https://docs.scipy.org/doc/scipy/reference/optimize.html.
    met : string (optional)
         Choose method for option. For more information see
         - https://docs.scipy.org/doc/scipy/reference/generated/
         scipy.optimize.minimize.html#scipy.optimize.minimize
         - https://docs.scipy.org/doc/scipy/reference/generated/
         scipy.optimize.minimize_scalar.html#scipy.optimize.minimize_scalar.
    initial_guess : float or integer (optional)
                    Initial guess passed to scipy.optimize.minimize. This
                    is recommended to be small.
    func_args : tuple
                Arguments passed to f and g.
    f : objective function.

        ``f(x, *func_args) -> float``

        where ``x`` is a 1-D array with shape(d, ) and func_args is a
        tuple of arguments needed to compute the function value.
    g : gradient of objective function.

       ``g(x, *func_args) -> 1-D array with shape (d, )``

        where ``x`` is a 1-D array with shape(d, ) and func_args is a
        tuple of arguments needed to compute the gradient.
    bounds_1 : integer
               Lower bound used for projection.
    bounds_2 : integer
               Upper bound used for projection.

    Returns
    -------
    sd_iterations : 2-D array with shape (m, d)
                    Each row of sd_iterations contains a new point
                    after a steepest descent iteration.
    sd_iterations_partner_points: 2-D array with shape (m, d)
                                  Corresponding partner points for
                                  sd_iterations.

    """
    its = 0
    sd_iterations = np.zeros((1, d))
    sd_iterations[0, :] = point.reshape(1, d)
    while its < m:
        x_iteration = mt_alg.sd_iteration(point, projection, option, met,
                                          initial_guess, func_args, f, g,
                                          bound_1, bound_2)
        sd_iterations = np.vstack([sd_iterations, x_iteration.reshape((1, d))])
        its += 1
        point = x_iteration
    sd_iterations_partner_points = mt_alg.partner_point_each_sd(sd_iterations,
                                                                d, beta, its,
                                                                g, func_args)
    return sd_iterations, sd_iterations_partner_points
