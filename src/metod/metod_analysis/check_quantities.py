from numpy import linalg as LA

from metod import objective_functions as mt_obj


def check_quantities(beta, x, y, func_args):
    """
    Check that the sum of results from quantities.py is the same as
    b ** 2 + 2 * b.T (x - y), where b = beta * (g(y, *func_args) - g(x,
    *func_args)).

    Parameters
    ----------
    beta : float or integer (optional)
           Small constant step size to compute the partner points.
    x : 1-D array of shape (d, )
        First point.
    y : 1-D array of shape (d, )
        Second point.
    func_args : tuple
                Arguments passed to f and g.

    Returns
    -------
    calc : float
           Calculation of b ** 2 + 2 * b.T (x - y).

    """
    b = beta * (mt_obj.several_quad_gradient(y, *func_args) -
                mt_obj.several_quad_gradient(x, *func_args))
    calc = float(LA.norm(b) ** 2 + 2 * b.T @ (x - y))
    return calc
