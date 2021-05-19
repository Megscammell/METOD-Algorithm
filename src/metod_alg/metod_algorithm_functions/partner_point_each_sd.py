import numpy as np


def partner_point_each_sd(all_iterations_of_sd, beta, store_grad):
    """
    Compute all corresponding partner points for all iterations of
    steepest descent.

    Parameters
    ----------
    all_iterations_of_sd : 2-D array with shape (iterations, d), where
                           iterations is the total number of steepest
                           descent iterations.
    beta : float or integer
           Small constant step size to compute the partner points.
    store_grad : 2-D array with shape (iterations, d)
                 store of all gradients of each point of all_iterations_of_sd.

    Returns
    -------
    all_iterations_of_sd_partner_points : 2-D array with shape
                                          (iterations, d)
                                          Computation of corresponding partner
                                          points for all_iterations_of_sd.

    """
    all_iterations_of_sd_partner_points = (all_iterations_of_sd -
                                           beta * store_grad)
    return all_iterations_of_sd_partner_points