def partner_point(point, beta, d, g, func_args):
    """Compute partner point

    Parameters
    ----------
    point : 1-D array with shape (d, )
            Compute corresponding partner_point for point.
    beta : float or integer
           Small constant step size to compute the partner point.
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
    partner_point : 1-D array with shape (d, )
                    Computation of partner point, that is,
                    x = x - beta * g(point, *func_args).

    """
    partner_point = point - beta * g(point, *func_args)
    return partner_point
