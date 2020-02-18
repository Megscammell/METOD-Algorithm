import numpy as np
import scipy
from scipy.optimize import minimize

import metod_testing as mtv3

def sd_iteration(point, projection, option, met, initial_guess, func_args, f, g):
    """Find step size gamma by either using exact line search or using strong Wolfe conditions.

    Keyword arguments:
    point -- is a (d,) array
    projection -- is a boolean variable. If projection = True, this projects points back to the [0,1]^d cube
    option -- choose from 'minimize' or 'minimize_scalar' and must input as a string
    met -- choose method to use
    initial guess -- if chosen 'minimize', choose an initial guess
    func_args - arguments passed to gradient and function in order to compute the function and gradient
    f -- function
    g -- gradient
    """

    if option =='minimize':
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
        
    return new_point
