import numpy as np

from metod import metod_analysis as mt_ays


def evaluate_quantities_with_points(beta, x_tr, y_tr, min_x, min_y, d,
                                    func_args):
    """For trajectories x^(k_x) and y^(k_y), where k_x = (0,...,K_x) and k_y =
    (0,...,K_y), evaluate quantites.

    Parameters
    ----------
    beta : float or integer
        Small constant step size to compute the partner points.
    x_tr :   2-D array with shape (iterations + 1, d)
             First array containing steepest descent iterations from the first
             starting point.
    y_tr :   2-D array with shape (iterations + 1, d)
             Second array containing steepest descent iterations from the
             second starting point.
    min_x : integer
            Region of attraction index of x_tr.
    min_y : integer
            Region of attraction index of y_tr.
    d : integer
        Size of dimension.
    func_args : tuple
                Arguments passed to f and g.


    Returns
    -------
    store_quantites : 2-D array with shape (iterations, 5)
                 Computation of each of the 5 quantites at each iteration.
    sum_quantities : 1-D array of shape iterations
               Sum of all quantites at each iteration.

    """
    store_beta = np.zeros((4, 5))
    sum_beta = np.zeros((4))
    index = 0
    for j in range(1, 3):
        for k in range(1, 3):
            x = x_tr[j, :].reshape(d, )
            y = y_tr[k, :].reshape(d, )
            store_beta[index, :], sum_beta[index] = (mt_ays.quantities
                                                     (x, y, min_x, min_y, beta,
                                                      *func_args))

            calc = mt_ays.check_quantities(beta, x, y, func_args)
            assert(np.round(calc, 5) == np.round(sum_beta[index], 5))
            index += 1
    return store_beta, sum_beta
