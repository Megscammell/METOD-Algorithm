from numpy import linalg as LA


def check_quantities(beta, x, y, g, func_args):
    """
    Check that the sum of results from quantities.py is the same as
    b ** 2 + 2 * b.T (x - y), where b = beta * (g(y, *func_args) - g(x,
    *func_args)).

    Parameters
    ----------
    beta : float or integer
           Small constant step size to compute the partner points.
    x : 1-D array of shape (d, )
        First point.
    y : 1-D array of shape (d, )
        Second point.
    g : gradient of objective function.

       ``g(x, *func_args) -> 1-D array with shape (d, )``

        where ``x`` is a 1-D array with shape(d, ) and func_args is a
        tuple of arguments needed to compute the gradient.
    func_args : tuple
                Arguments passed to g.

    Returns
    -------
    calc : float
           Calculation of b ** 2 + 2 * b.T (x - y).

    """
    b = beta * (g(y, *func_args) -
                g(x, *func_args))
    calc = float(LA.norm(b) ** 2 + 2 * b.T @ (x - y))
    return calc
