import numpy as np

import metod_testing as mtv3

def apply_sd_until_warm_up(point, d, m, beta, projection, option, met, initial_guess, func_args, f, g, bound_1, bound_2):
    """Calculates m and m-1 iterations of steepest descent for a point and calculates the corresponding partner points

    Keyword arguments:
    point -- is a (d,) array
    d -- is dimension
    m -- is warm up
    beta -- partner point step size
    projection -- is a boolean variable. If projection = True, this projects points back to the [0,1]^d cube
    option -- choose from minimize' or 'minimize_scalar'
    met -- choose method to use
    initial guess -- if chosen 'minimize', choose an initial guess
    func_args - args passed to gradient and function in order to compute the function and gradient
    f -- function
    g -- gradient
    """
    its = 0
    sd_iterations = np.zeros((1, d))
    sd_iterations[0,:] = point.reshape(1, d)
    while its < m:
        x_iteration = mtv3.sd_iteration(point, projection, option, met,      initial_guess, func_args, f, g, bound_1, bound_2)
        sd_iterations = np.vstack([sd_iterations, x_iteration.reshape((1, d))])
        its += 1
        point = x_iteration    

    sd_iterations_partner_points = mtv3.partner_point_each_sd(sd_iterations, d,beta, its, g, func_args)

    return sd_iterations, sd_iterations_partner_points 
