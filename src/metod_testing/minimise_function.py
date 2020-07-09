def minimise_function(gamma, point, f, g, *func_args):
    """Minimise quadratic function with respect to gamma

    Keyword arguments:
    gamma - minimse with respect to gamma
    point -- is a (d,) array and the function is evaluated at point.
    f -- function
    g -- gradient
    func_args-- parameters needed to compute the function and gradient
    """
    func_val = f((point - gamma * g(point, *func_args)), *func_args)
    return func_val
