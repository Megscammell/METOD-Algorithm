import numpy as np

from metod import metod_algorithm_functions as mt_alg


def apply_sd_until_warm_up(point, d, m, beta, projection, option, met,
                           initial_guess, func_args, f, g, bound_1, bound_2,
                           relax_sd_it):
    """
    Computes m iterations of steepest descent and the corresponding
    partner points.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            Apply steepest descent iterations to point.
    d : integer
        Size of dimension.
    m : integer
        Number of steepest descent iterations to apply to a point
        before making a decision on terminating descents.
    beta : float or integer
           Small constant step size to compute the partner points.
    projection : boolean
                 If projection is True, points are projected back to
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

        `f(x, *func_args) -> float`

        where `x` is a 1-D array with shape(d, ) and func_args is a
        tuple of arguments needed to compute the function value.
    g : gradient of objective function.

       `g(x, *func_args) -> 1-D array with shape (d, )`

        where `x` is a 1-D array with shape(d, ) and func_args is a
        tuple of arguments needed to compute the gradient.
    bounds_1 : integer
               Lower bound used for projection.
    bounds_2 : integer
               Upper bound used for projection.
    relax_sd_it : float or integer
                  Multiply the step size by a small constant in [0, 2], to
                  obtain a new step size for steepest descent iterations. This
                  process is known as relaxed steepest descent [1].

    Returns
    -------
    sd_iterations : 2-D array with shape (m, d)
                    Each steepest descent iteration is stored in each row of
                    sd_iterations.
    sd_iterations_partner_points: 2-D array with shape (m, d)
                                  Corresponding partner points for
                                  sd_iterations.

    References
    ----------
    1) Raydan, M., Svaiter, B.F.: Relaxed steepest descent and
       cauchy-barzilai- borwein method. Computational Optimization and
       Applications 21(2), 155–167 (2002)

    """
    its = 0
    sd_iterations = np.zeros((1, d))
    sd_iterations[0, :] = point.reshape(1, d)
    while its < m:
        x_iteration = mt_alg.sd_iteration(point, projection, option, met,
                                          initial_guess, func_args, f, g,
                                          bound_1, bound_2, relax_sd_it)
        sd_iterations = np.vstack([sd_iterations, x_iteration.reshape((1, d))])
        its += 1
        point = x_iteration
    sd_iterations_partner_points = mt_alg.partner_point_each_sd(sd_iterations,
                                                                d, beta, its,
                                                                g, func_args)
    return sd_iterations, sd_iterations_partner_points
