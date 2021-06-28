def minimize_function(gamma, point, f, grad, *func_args):
    """
    Function used to apply scipy.optimize.minimize or scipy.optimize.
    minimize_scalar to find step size gamma.

    Parameters
    ----------
    gamma : float
            Step size.
    point : 1-D array with shape (d, )
            A point used to evaluate the objective function.
    f : objective function.

        `f(point, *func_args) -> float`

        where `point` is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the function value.
    grad : gradient of objective function at point.

           `g(point, *func_args) -> 1-D array with shape (d, )`

            where `point` is a 1-D array with shape (d, ) and func_args is
            a tuple of arguments needed to compute the gradient.
    func_args : tuple
                Arguments passed to f and g.

    Returns
    -------
    func_val : float

    """
    func_val = f((point - gamma * grad), *func_args)
    return func_val
