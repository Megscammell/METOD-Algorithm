def partner_point(point, beta, d, g, func_args):
    """Compute corresponding partner point

    Keyword arguments:
    point -- is a (d,) array
    beta -- fixed small step size
    d -- is dimension
    g -- gradient
    func_args -- paramters passed to the function and gradient
    """
    partner_point = point - beta * g(point, *func_args)
    return partner_point
