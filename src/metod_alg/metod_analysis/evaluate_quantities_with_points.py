import numpy as np

from metod_alg import metod_analysis as mt_ays


def evaluate_quantities_with_points(beta, x_tr, y_tr, d,
                                    g, func_args):
    """
    For trajectories x^(k_x) and y^(k_y), where k_x = (0,...,K_x) and k_y =
    (0,...,K_y), evaluate quantites.

    Parameters
    ----------
    beta : float or integer
           Small constant step size to compute the partner points.
    x_tr :   2-D array with shape (iterations + 1, d)
             Array containing steepest descent iterations from the first
             starting point.
    y_tr :   2-D array with shape (iterations + 1, d)
             Array containing steepest descent iterations from the
             second starting point.
    min_x : integer
            Region of attraction index of x_tr.
    min_y : integer
            Region of attraction index of y_tr.
    d : integer
        Size of dimension.
    g : gradient of objective function.

       ``g(x, *func_args) -> 1-D array with shape (d, )``

        where ``x`` is a 1-D array with shape(d, ) and func_args is a
        tuple of arguments needed to compute the gradient.
    func_args : tuple
                Arguments passed to g.

    Returns
    -------
    store_beta : 2-D array with shape (4, 2)
                 Computation of c1 and c2 at each iteration, where
                 c_1 = ||b||^2, c_2 = b^T (x - y) and
                 b = beta * (g(y, *func_args) - g(x, *func_args)).
     sum_beta : 1-D array of shape 4
                Compute c_1 + c_2 at each iteration.
    """
    store_beta = np.zeros((4, 2))
    sum_beta = np.zeros((4))
    index = 0
    for j in range(1, 3):
        for k in range(1, 3):
            x = x_tr[j, :].reshape(d, )
            y = y_tr[k, :].reshape(d, )
            store_beta[index, 0] = (beta ** 2 *
                                    (np.linalg.norm(g(y, *func_args) -
                                     g(x, *func_args)) ** 2))
            store_beta[index, 1] = (2 * beta * (g(y, *func_args) -
                                                g(x, *func_args)).T @ (x - y))
            calc = mt_ays.check_quantities(beta, x, y, g, func_args)
            assert(np.round(calc, 5) == np.round(np.sum(store_beta[index]), 5))
            sum_beta[index] = calc
            index += 1
    return store_beta, sum_beta
