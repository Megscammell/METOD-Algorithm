import numpy as np

import metod_testing as mtv3


def partner_point_each_sd(all_iterations_of_sd, d, beta, iterations, g,                                  func_args):
    """Compute all corresponding partner points for each iteration of steepest descent for point

    Keyword arguments:
    all_iterations_of_sd -- is an array of size (iterations + 1) x d                                   containing all iterations of steepest descent of a                         point
    d -- is dimension
    beta -- fixed small step size 
    iterations -- total number of steepest descent iterations
    g -- user defined gradient
    func_args -- parameters needed to compute the function and                              gradient
    """
    all_iterations_of_sd_partner_points = np.zeros((iterations + 1, d))
    for its in range(iterations + 1):
        point = all_iterations_of_sd[its, :].reshape((d,))
        partner_point = mtv3.partner_point(point, beta, d, g, func_args)
        all_iterations_of_sd_partner_points[its,:] = partner_point.reshape(1,d)
    return all_iterations_of_sd_partner_points