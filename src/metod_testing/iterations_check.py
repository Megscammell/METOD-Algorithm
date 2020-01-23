import numpy as np

import metod_testing as mtv3


def iterations_check(point, d, points_x, projection, its, option, met, initial_guess, func_args, f, g):

    """Check step sizes when line search option is chosen

    Keyword arguments:
    point -- is a (d,) array
    d -- is dimension
    points_x -- stored iterations of steepest descent for a point
    projection -- is a boolean variable. If projection = True, this projects points back to the [0,1]^d cube
    its -- number of iterations of steepest descent that have occured so far
    option -- choose from 'line_search', 'minimize' or 'minimize_scalar'
    met -- if chosen 'minimize' or  'minimize_scalar' choose method to use
    initial guess -- if chosen 'minimize', choose an initial guess
    func_args - args passed to gradient and function in order to compute the function and gradient
    f -- function
    g -- gradient
    """
    break_true = False
    flag = False
    x_iteration, change_point = mtv3.sd_iteration(point, projection, option, met, initial_guess, func_args, f, g)
    count = 0
    while change_point == True:
        flag = True
        its = 0
        points_x = np.zeros((1, d))
        point = np.random.uniform(0, 1,(d,))
        points_x[0,:] = point.reshape(1, d)
        x_iteration, change_point =  mtv3.sd_iteration(point, projection,option, met, initial_guess, func_args, f, g)
        count += 1
        if count > 1000:
            break_true = True
            break

    if break_true == True:
        raise ValueError('Cannot get suitable step size using line_search. Please try another option')
    
    points_x = np.vstack([points_x, x_iteration.reshape((1, d))])
    its += 1
    return points_x, x_iteration, its, flag
