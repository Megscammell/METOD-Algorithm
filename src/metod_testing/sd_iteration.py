import numpy as np
import scipy
from scipy.optimize import minimize

import metod_testing as mtv3


def sd_iteration(point, projection, option, met, initial_guess, func_args, f,
                 g, bound_1, bound_2):
    """Compute iteration of steepest descent.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point to apply steepest descent iterations.
    projection : boolean
                 If projection is True, this projects points back to
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

    Returns
    -------
    new_point : 1-D array with shape (d, )
                Compute a steepest descent iteration, that is,
                x = x - gamma * g(x, *func_args), where gamma
                is calculated by finding gamma > 0 such that
                min(f(x - gamma * g(x, *func_args))).

    """

    if option == 'minimize':
        met_list_minimize = (['Nelder-Mead', 'Powell', 'CG', 'BFGS',
                              'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP',
                              'trust-constr'])
        if met not in met_list_minimize:
            raise ValueError('Please choose correct method for minimize '
                             'option')
        t = scipy.optimize.minimize(mtv3.minimise_function, initial_guess,
                                    args=(point, f, g, *func_args), method=met)
        if float(t.x) <= 0:
            raise ValueError('Step size less than or equal to 0. Please '
                             'choose different option and/or method')
        new_point = point - float(t.x) * g(point, *func_args)
        # if t.success == True:
        if projection is True:
            new_point = np.clip(new_point, bound_1, bound_2)
        # else:
        # raise ValueError('Optimizer to calculate step size did not exit'
        #                  ' successfully')

    elif option == 'minimize_scalar':
        met_list_minimize_scalar = (['golden', 'brent', 'Golden', 'Brent',
                                    'bounded', 'Bounded'])
        if met not in met_list_minimize_scalar:
            raise ValueError('Please choose correct method for minimize_scalar'
                             ' option')
        if met == 'Bounded' or met == 'bounded':
            t = scipy.optimize.minimize_scalar(mtv3.minimise_function,
                                               args=(point, f, g, *func_args),
                                               method='bounded',
                                               bounds=(0.00001, 10000))
        else:
            t = scipy.optimize.minimize_scalar(mtv3.minimise_function,
                                               args=(point, f, g, *func_args),
                                               method=met)
        if float(t.x) <= 0:
            raise ValueError('Step size less than or equal to 0. Please choose'
                             ' different option and/or method')
        new_point = point - float(t.x) * g(point, *func_args)
        # if t.success == True:
        if projection is True:
            new_point = np.clip(new_point, bound_1, bound_2)
        # else:
        #     raise ValueError('Optimizer to calculate step size did not exit
        #                        ' successfully')
    else:
        raise ValueError('Please select valid option')
    return new_point
