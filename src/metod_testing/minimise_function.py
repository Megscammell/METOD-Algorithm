import numpy as np


def minimise_function(gamma, point, f, g, *func_args):
    """Minimise quadratic function with respect to gamma

    Keyword arguments:
    gamma - minimse the quadratic function with respect to the constant gamma
    point -- is a (d,) array and the function is evaluated at point.
    f -- user defined function
    g -- user defined gradient
    func_args-- parameters needed to compute the function and                              gradient
    """
    func_val = f((point - gamma * g(point, *func_args)), *func_args)
    return func_val
