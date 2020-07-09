import numpy as np
from numpy import linalg as LA

import metod_testing as mtv3


def apply_sd_until_stopping_criteria(point, d, projection, tolerance, option,
                                     met, initial_guess, func_args, f, g,
                                     bound_1, bound_2):
    """Apply steepest descent iterations until stopping criteria has been met.

    Keyword arguments:
    point -- is a (d,) array
    d -- is dimension
    projection -- is a boolean variable. If projection = True, this projects
    points back to the [0,1]^d cube
    tolerance -- small constant in which norm of gradient has to be smaller
    option -- choose from 'minimize' or 'minimize_scalar'
    met -- choose method to use
    initial guess -- if chosen 'minimize', choose an initial guess
    func_args - args passed to gradient and function in order to compute the
    function and gradient
    f -- function
    g -- gradient
    """
    its = 0
    sd_iterations = np.zeros((1, d))
    sd_iterations[0, :] = point.reshape(1, d)
    x_iteration = mtv3.sd_iteration(point, projection, option, met,
                                    initial_guess, func_args, f, g, bound_1,
                                    bound_2)
    sd_iterations = np.vstack([sd_iterations, x_iteration.reshape((1, d))])
    its += 1
    point = x_iteration
    while LA.norm(g(point, *func_args)) >= tolerance:
        x_iteration = mtv3.sd_iteration(point, projection, option, met,
                                        initial_guess, func_args, f, g,
                                        bound_1, bound_2)
        sd_iterations = np.vstack([sd_iterations, x_iteration.reshape((1, d))])
        its += 1
        point = x_iteration
        if its > 200:
            break
    if its >= 200:
        raise ValueError('Number of iterations has exceeded 200.'
                         'Try anothermethod and/or option.')
    return sd_iterations, its
