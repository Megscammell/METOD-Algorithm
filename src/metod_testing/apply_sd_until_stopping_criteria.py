import numpy as np
from numpy import linalg as LA

import metod_testing as mtv3

def apply_sd_until_stopping_criteria(initial_point, point, d, projection, tolerance, option, met, initial_guess,                                          func_args, f, g):
    """Apply steepest descent iterations until stopping criteria has been met.

    Keyword arguments:
    point -- is a (d,) array
    d -- is dimension
    projection -- is a boolean variable. If projection = True, this projects points back to the [0,1]^d cube
    tolerance -- small constant in which norm of gradient has to be smaller 
    option -- choose from 'line_search', 'minimize' or 'minimize_scalar'
    met -- if chosen 'minimize' or  'minimize_scalar' choose method to use
    initial guess -- if chosen 'minimize', choose an initial guess
    func_args - args passed to gradient and function in order to compute the function and gradient
    f -- user defined function
    g -- user defined gradient
    """

    its = 0
    sd_iterations = np.zeros((1, d))
    sd_iterations[0,:] = point.reshape(1, d)
    sd_iterations, x_iteration, its, flag = mtv3.iterations_check(point, d, sd_iterations, projection, its,                                            option, met, initial_guess,                                                func_args, f, g)
    point = x_iteration

    while LA.norm(g(point, *func_args)) >= tolerance:
        sd_iterations, x_iteration, its, flag = mtv3.iterations_check(point, d, sd_iterations, projection,                                             its, option, met,                                                      initial_guess, func_args, f, g)
        point = x_iteration
        if its > 200:
            break
        
        if flag == True and initial_point == False:
            break

    if its >= 200:
        raise ValueError('Number iterations has exceeded 200. Try another method and/or option chosen as point gets stuck.')   

    if flag == True and initial_point == False:
        raise ValueError('Cannot replace point when steepest descent iterations are applied to find a minimizer. Please choose different option and method') 

    return sd_iterations, its
