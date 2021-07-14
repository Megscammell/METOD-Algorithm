import numpy as np

from metod_alg import metod_analysis as mt_ays


def evaluate_quantities_with_points_quad(beta, x_tr, y_tr, min_x, min_y, d,
                                         g, func_args):
    """
    For trajectories x^(k_x) and y^(k_y), where k_x = (0,...,K_x) and k_y =
    (0,...,K_y), evaluate quantites for the minimum of several quadratic
    forms function.

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
    g : gradient of objective function.

       ``g(x, *func_args) -> 1-D array with shape (d, )``

        where ``x`` is a 1-D array with shape(d, ) and func_args is a
        tuple of arguments needed to compute the gradient.
    func_args : tuple
                Arguments passed to g.


    Returns
    -------
    store_quantites : 2-D array with shape (iterations, 5)
                      store_quantites[:,:3], contains terms from the expansion
                      of c_1 and store_quantites[:,3:] contains terms from the
                      expansion of c_2, where c_1 = ||b||^2, c_2 = b^T (x - y)
                      and b = beta * (g(y, *func_args) - g(x, *func_args)).
    sum_quantities : 1-D array of shape iterations
                     Compute c_1 + c_2 at each iteration.

    """
    store_quantites = np.zeros((4, 5))
    sum_quantites = np.zeros((4))
    index = 0
    for j in range(1, 3):
        for k in range(1, 3):
            x = x_tr[j, :].reshape(d, )
            y = y_tr[k, :].reshape(d, )
            (store_quantites[index, :],
             sum_quantites[index]) = (mt_ays.quantities
                                      (x, y, min_x, min_y, beta,
                                       *func_args))

            calc = mt_ays.check_quantities(beta, x, y, g, func_args)
            assert(np.round(calc, 5) == np.round(sum_quantites[index], 5))
            index += 1
    return store_quantites, sum_quantites
