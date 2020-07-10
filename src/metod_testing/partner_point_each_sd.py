import numpy as np

import metod_testing as mtv3


def partner_point_each_sd(all_iterations_of_sd, d, beta, iterations, g,
                          func_args):
    """Compute all corresponding partner points for all iterations of
    steepest descent.

    Parameters
    ----------
    all_iterations_of_sd : 2-D array with shape (iterations, d), where
                           iterations is the total number of steepest
                           descent iterations.
    d : integer
        Size of dimension.
    beta : float or integer
           Small constant step size to compute the partner points.
    iterations : integer
                 Total number of steepest descent iterations.
    g : gradient of objective function.

       ``g(x, *func_args) -> 1-D array with shape (d, )``

        where ``x`` is a 1-D array with shape(d, ) and func_args is a
        tuple of arguments needed to compute the gradient.
    func_args : tuple
                Arguments passed to g.
    Returns
    -------
    all_iterations_of_sd_partner_points : 2-D array with shape
                                          (iterations, d)
                                          Computation of corresponding partner
                                          points for all_iterations_of_sd.

    """
    all_iterations_of_sd_partner_points = np.zeros((iterations + 1, d))
    for its in range(iterations + 1):
        point = all_iterations_of_sd[its, :].reshape((d, ))
        partner_point = mtv3.partner_point(point, beta, d, g, func_args)
        all_iterations_of_sd_partner_points[its, :] = (partner_point.reshape
                                                       (1, d))
    return all_iterations_of_sd_partner_points
