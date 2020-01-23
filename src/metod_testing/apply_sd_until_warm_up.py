import numpy as np

import metod_testing as mtv3

def apply_sd_until_warm_up(point, d, m, beta, projection, option, met, initial_guess, func_args, f, g):
    """Calculates m and m-1 iterations of steepest descent for a point and calculates the corresponding partner points

    Keyword arguments:
    point -- is a (d,) array
    d -- is dimension
    m -- is warm up
    beta -- partner point step size
    projection -- is a boolean variable. If projection = True, this projects points back to the [0,1]^d cube
    option -- choose from 'line_search', 'minimize' or 'minimize_scalar'
    met -- if chosen 'minimize' or  'minimize_scalar' choose method to use
    initial guess -- if chosen 'minimize', choose an initial guess
    func_args - args passed to gradient and function in order to compute the function and gradient
    f -- function
    g -- gradient
    """
    its = 0
    sd_iterations = np.zeros((1, d))
    sd_iterations[0,:] = point.reshape(1, d)
    while its < m:
        sd_iterations, x_iteration, its, flag = mtv3.iterations_check(
                                                point, d, sd_iterations, projection, its, option, met, initial_guess, func_args, f, g)
        point = x_iteration    

    sd_iterations_partner_points = mtv3.partner_point_each_sd(sd_iterations, d, beta, its, g, func_args)

    return sd_iterations, sd_iterations_partner_points 
