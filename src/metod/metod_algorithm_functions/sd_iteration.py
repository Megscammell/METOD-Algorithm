import numpy as np
import scipy
from scipy import optimize

from metod import metod_algorithm_functions as mt_alg


def sd_iteration(point, projection, option, met, initial_guess, func_args, f,
                 g, bound_1, bound_2, relax_sd_it):
    """
    Compute iteration of steepest descent.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            Apply steepest descent iterations to point.
    projection : boolean
                 If projection is True, points are projected back to
                 (bound_1, bound_2). If projection is False, points are
                 kept the same.
    option : string
             Choose from 'minimize' or 'minimize_scalar'. For more
             information about each option see
             https://docs.scipy.org/doc/scipy/reference/optimize.html.
    met : string
         Choose method for option. For more information see
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
                Arguments passed to f and g.
    f : objective function.

        `f(point, *func_args) -> float`

        where `point` is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the function value.
    g : gradient of objective function.

       `g(point, *func_args) -> 1-D array with shape (d, )`

        where `point` is a 1-D array with shape (d, ) and func_args is
        a tuple of arguments needed to compute the gradient.
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
    new_point : 1-D array with shape (d, )
                Compute a steepest descent iteration. That is,
                x = x - gamma * g(x, *func_args), where gamma > 0, is computed by exact line search.

    References
    ----------
    1) Raydan, M., Svaiter, B.F.: Relaxed steepest descent and
       cauchy-barzilai- borwein method. Computational Optimization and
       Applications 21(2), 155–167 (2002)

    """

    if option == 'minimize':
        met_list_minimize = (['Nelder-Mead', 'Powell', 'CG', 'BFGS',
                              'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP',
                              'trust-constr'])
        if met not in met_list_minimize:
            raise ValueError('Please choose correct method for minimize '
                             'option')
        t = scipy.optimize.minimize(mt_alg.minimize_function, initial_guess,
                                    args=(point, f, g, *func_args), method=met)
        if float(t.x) <= 0:
            raise ValueError('Step size less than or equal to 0. Please '
                             'choose different option, method or initial_guess.')
        new_point = point - relax_sd_it * float(t.x) * g(point, *func_args)
        if projection is True:
            new_point = np.clip(new_point, bound_1, bound_2)

    elif option == 'minimize_scalar':
        met_list_minimize_scalar = (['golden', 'brent', 'Golden', 'Brent'])
        if met not in met_list_minimize_scalar:
            raise ValueError('Please choose correct method for minimize_scalar'
                             ' option')
        else:
            t = scipy.optimize.minimize_scalar(mt_alg.minimize_function,
                                               bracket=(0,initial_guess),
                                               args=(point, f, g, *func_args),
                                               method=met)
        if float(t.x) <= 0:
            raise ValueError('Step size less than or equal to 0. Please choose'
                             ' different option, method or initial_guess.')
        new_point = point - relax_sd_it * float(t.x) * g(point, *func_args)

        if projection is True:
            new_point = np.clip(new_point, bound_1, bound_2)

    else:
        raise ValueError('Please select valid option')
    return new_point
