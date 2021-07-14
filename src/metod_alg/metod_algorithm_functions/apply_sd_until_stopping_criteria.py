import numpy as np
from numpy import linalg as LA

from metod_alg import metod_algorithm_functions as mt_alg


def apply_sd_until_stopping_criteria(point, d, projection, tolerance, option,
                                     met, initial_guess, func_args, f, g,
                                     bound_1, bound_2, usage, relax_sd_it,
                                     init_grad):
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
                metod_analysis).
   option : string
            Used to find the step size for each iteration of steepest
            descent.
            Choose from 'minimize' or 'minimize_scalar'. For more
            information on 'minimize' or 'minimize_scalar' see
            https://docs.scipy.org/doc/scipy/reference/optimize.html.
    met : string
           If option = 'minimize' or option = 'minimize_scalar', choose
           appropiate method. For more information see
           - https://docs.scipy.org/doc/scipy/reference/generated/
           scipy.optimize.minimize.html#scipy.optimize.minimize
           - https://docs.scipy.org/doc/scipy/reference/generated/
           scipy.optimize.minimize_scalar.html#scipy.optimize.minimize_scalar.
    initial_guess : float or integer
                    Initial guess passed to scipy.optimize.minimize and the
                    upper bound for the bracket interval when using the
                    'Brent' or 'Golden' method for
                    scipy.optimize.minimize_scalar. This
                    is recommended to be small.
    func_args : tuple
                Arguments passed to the function f and gradient g.
    f : objective function.

        `f(point, *func_args) -> float`

        where point` is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the function value.
    g : gradient of objective function.

       `g(point, *func_args) -> 1-D array with shape (d, )`

        where `point` is a 1-D array with shape (d, ) and func_args is
        a tuple of arguments needed to compute the gradient.
    bound_1 : integer
               Lower bound used for projection.
    bound_2 : integer
               Upper bound used for projection.
    usage : string
            Used to decide stopping condition for steepest descent
            iterations. Should be either usage == 'metod_algorithm' or
            usage == 'metod_analysis'.
    relax_sd_it : float or integer
                  Multiply the step size by a small constant in [0, 2], to
                  obtain a new step size for steepest descent iterations. This
                  process is known as relaxed steepest descent [1].
    init_grad : either None or 1-D array
                If local descent starts from some starting point, then
                init_grad will be a 1-D array of the gradient at the
                starting point. If local descent starts from another point
                (i.e a point in which a warm up period m has been applied),
                then init_grad = None and the gradient at that point will
                be computed.
    Returns
    -------
    sd_iterations : 2-D array with shape (its + 1, d)
                    Each steepest descent iteration is stored in each row of
                    sd_iterations.
    its: integer
         Total number of steepest descent iterations.

    store_grad : 2-D array with shape (its + 1, d)
                 Gradient of each point in sd_iterations.

    References
    ----------
    1) Raydan, M., Svaiter, B.F.: Relaxed steepest descent and
       cauchy-barzilai- borwein method. Computational Optimization and
       Applications 21(2), 155–167 (2002)

    """
    its = 0
    sd_iterations = np.zeros((1, d))
    store_grad = np.zeros((1, d))
    sd_iterations[0, :] = point.reshape(1, d)
    if init_grad is None:
        grad = g(point, *func_args)
    else:
        grad = init_grad
    store_grad[0, :] = grad.reshape(1, d)
    if usage == 'metod_algorithm':
        while LA.norm(grad) >= tolerance:
            x_iteration = mt_alg.sd_iteration(point, projection, option, met,
                                              initial_guess, func_args, f,
                                              grad, bound_1, bound_2,
                                              relax_sd_it)
            sd_iterations = np.vstack([sd_iterations, x_iteration.reshape
                                      ((1, d))])
            its += 1
            point = x_iteration
            if its > 1000:
                raise ValueError('Number of iterations has exceeded 1000.'
                                 ' Try another method and/or option.')
            grad = g(point, *func_args)
            store_grad = np.vstack([store_grad, grad.reshape
                                   ((1, d))])
        return sd_iterations, its, store_grad

    elif usage == 'metod_analysis':
        while its < tolerance:
            x_iteration = mt_alg.sd_iteration(point, projection, option, met,
                                              initial_guess, func_args, f,
                                              grad, bound_1, bound_2,
                                              relax_sd_it)
            sd_iterations = np.vstack([sd_iterations, x_iteration.reshape
                                      ((1, d))])
            its += 1
            point = x_iteration
            grad = g(point, *func_args)
            store_grad = np.vstack([store_grad, grad.reshape
                                   ((1, d))])
        return sd_iterations, its, store_grad
