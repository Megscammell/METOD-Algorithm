import numpy as np
import scipy
from scipy.optimize import minimize

import metod_testing as mtv3

def sd_iteration(point, projection, option, met, initial_guess, func_args, f, g):
    """Minimise quadratic function with respect to gamma

    Keyword arguments:
    point -- is a (d,) array
    projection -- is a boolean variable. If projection = True, this projects points back to the [0,1]^d cube
    option -- choose from 'line_search', 'minimize' or 'minimize_scalar'
    met -- if chosen 'minimize' or  'minimize_scalar' choose method to use
    initial guess -- if chosen 'minimize', choose an initial guess
    func_args - args passed to gradient and function in order to compute the function and gradient
    f -- user defined function
    g -- user defined gradient
    """
    change_point = False
    if option == 'line_search':
        pk = -g(point, *func_args)
        alpha = scipy.optimize.line_search(f, g, point, pk, args = (func_args))
        if (alpha[0] is None) or (alpha[3] is None) or (alpha[5] is None):
            change_point = True
            new_point = point
        else:
            new_point = point - float(alpha[0]) * g(point, *func_args)
            if projection == True:
                new_point = np.clip(new_point, 0, 1)

    elif option =='minimize':

        met_list_minimize = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr']
        if met not in met_list_minimize:
            raise ValueError('Please choose correct method for minimize option')
        t = scipy.optimize.minimize(mtv3.minimise_function, initial_guess, args =(point, f, g, *func_args), method = met)
        if float(t.x) <= 0:
            raise ValueError('Step size less than or equal to 0. Please choose different option and/or method')
        new_point = point - float(t.x) * g(point, *func_args)
        if t.success == True:
            if projection == True:
                new_point = np.clip(new_point, 0, 1)
        else:
            raise ValueError('Optimizer to calculate step size did not exit successfully')

    elif option == 'minimize_scalar':
        met_list_minimize_scalar = ['golden', 'brent', 'Golden', 'Brent', 'bounded', 'Bounded']
        if met not in met_list_minimize_scalar:
            raise ValueError('Please choose correct method for minimize_scalar option')
        if met == 'Bounded' or met == 'bounded':
            t = scipy.optimize.minimize_scalar(mtv3.minimise_function, args = (point, f, g, *func_args), method = 'bounded', bounds = (0.00001, 10000))
        else:
            t = scipy.optimize.minimize_scalar(mtv3.minimise_function, args = (point, f, g, *func_args), method = met)
        if float(t.x) <= 0:
            raise ValueError('Step size less than or equal to 0. Please choose different option and/or method')
        new_point = point - float(t.x) * g(point, *func_args)
        if t.success == True:
            if projection == True:
                new_point = np.clip(new_point, 0, 1)
        else:
            raise ValueError('Optimizer to calculate step size did not exit successfully')

    else:
        raise ValueError('Please select valid option')
        
    return new_point, change_point
