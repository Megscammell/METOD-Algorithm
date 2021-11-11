from metod_alg import metod_algorithm_functions as mt_alg
from metod_alg import check_metod_class as check_mt_alg


def check_if_new_minimizer(x_2, d, projection, tolerance,
                           option, met, initial_guess,
                           func_args, f, g, bound_1, bound_2,
                           usage, relax_sd_it, grad,
                           discovered_minimizers, const):
    """
    Checks if local minimizer has been missed.

    Parameters
    ----------
    x_2 : 1-D array with shape (d, )
          Point with m iterations of steepest descent.
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
    grad : 1-D array
           Gradient at x.
    discovered_minimizers : list
                            Previously identified minimizers.
    const : float or integer
            In order to classify a point as a new local minimizer, the
            euclidean distance between the point and all other discovered
            local minimizers must be larger than const.
    Returns
    -------
    1 or 0 : integer
             Will be 1 if a new local minimizer has been missed. Otherwise,
             will be set to 0 if local minimizer has already been discovered.

    References
    ----------
    1) Raydan, M., Svaiter, B.F.: Relaxed steepest descent and
       cauchy-barzilai- borwein method. Computational Optimization and
       Applications 21(2), 155–167 (2002)

    """
    (iterations_of_sd,
     its,
     store_grad) = (mt_alg.apply_sd_until_stopping_criteria
                    (x_2, d, projection, tolerance,
                     option, met, initial_guess,
                     func_args, f, g, bound_1, bound_2,
                     usage, relax_sd_it, grad))
    c = check_mt_alg.check_des_points(iterations_of_sd,
                                      discovered_minimizers,
                                      const)
    if c == None:
        return 1
    else:
        return 0
