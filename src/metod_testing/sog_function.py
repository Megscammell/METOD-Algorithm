import numpy as np

def sog_function(point, p, sigma_sq, store_x0, matrix_test, store_c):
    """Compute sum of Gaussians at a given point with given args.

    Keyword arguments:
    point -- is a (d,) array and the function is evaluated at point.
    args -- parameters needed to compute the function (i.e a set of symmetic matrices, set of minima, set of constants)
    """
    function_val = 0
    for i in range(p):
        function_val += store_c[i] * np.exp((- 1 / (2 * sigma_sq)) 
                                * np.transpose(point - store_x0[i])
                                @ matrix_test[i] @ (point - store_x0[i]))
    return float( -function_val)
