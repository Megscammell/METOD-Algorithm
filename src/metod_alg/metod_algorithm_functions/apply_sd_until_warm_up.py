import numpy as np

from metod_alg import metod_algorithm_functions as mt_alg


def apply_sd_until_warm_up(point, d, m, beta, projection, option, met,
                           initial_guess, func_args, f, g, bound_1, bound_2,
                           relax_sd_it, init_grad):
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
    init_grad : 1-D array
                If local descent starts from some starting point, then
                init_grad will be a 1-D array of the gradient at the
                starting point.

    Returns
    -------
    sd_iterations : 2-D array with shape (m + 1, d)
                    Each steepest descent iteration is stored in each row of
                    sd_iterations.
    sd_iterations_partner_points: 2-D array with shape (m, d)
                                  Corresponding partner points for
                                  sd_iterations.
    store_grad : 2-D array with shape (m + 1, d)
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
    grad = init_grad
    store_grad[0, :] = grad.reshape(1,d)
    while its < m:
        x_iteration = mt_alg.sd_iteration(point, projection, option, met,
                                          initial_guess, func_args, f, grad,
                                          bound_1, bound_2, relax_sd_it)
        sd_iterations = np.vstack([sd_iterations, x_iteration.reshape((1, d))])
        its += 1
        point = x_iteration
        grad = g(point, *func_args)
        store_grad = np.vstack([store_grad, grad.reshape
                                ((1, d))])
    sd_iterations_partner_points = mt_alg.partner_point_each_sd(sd_iterations,
                                                                beta,
                                                                store_grad)
    return sd_iterations, sd_iterations_partner_points, store_grad
