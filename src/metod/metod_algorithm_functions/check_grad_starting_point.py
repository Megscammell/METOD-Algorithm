import numpy as np
from numpy import linalg as LA
from warnings import warn


def check_grad_starting_point(x, point_index, no_points, bounds_set_x, sobol_points, d,
                              g, func_args, set_x, tolerance, num_points):
    """
    Check the norm of the gradient of a starting point is larger than some 
    tolerance. If not, the starting point will be changed.

    Parameters
    ----------
    x : 1-D array
        Original starting point.
    point_index : integer
                  Number of starting points changed due to the norm of the 
                  gradient at a starting point being less than some tolerance.
    no_points : integer
                Starting point index.
    bounds_set_x : tuple
                    Bounds for numpy.random.uniform distribution.
    sobol_points : 2-D array
                    If set_x='sobol', then a numpy.array with shape
                    (num points * 2, d) of Sobol sequence samples are generated
                    using SALib [1], which are randomly shuffled and used
                    as starting points for the METOD algorithm.
    d : integer
        Size of dimension.
    g : gradient of objective function.

        `g(x, *func_args) -> 1-D array with shape (d, )`

        where `x` is a 1-D array with shape(d, ) and func_args is a
        tuple of arguments needed to compute the gradient.
    func_args : tuple
                Arguments passed to g.
    set_x : 'random' or 'sobol'
            If set_x = 'random', random starting points
            are generated for the METOD algorithm. If set_x = 'sobol'
            is selected, then a numpy.array with shape
            (num points * 2, d) of Sobol sequence samples are generated
            using SALib [1], which are randomly shuffled and used
            as starting points for the METOD algorithm. The Default is
            set_x = 'sobol'.
    tolerance : integer or float
                Stopping condition for steepest descent iterations. Apply
                steepest descent iterations until the norm
                of g(point, *func_args) is less than some tolerance.
                Also check that the norm of the gradient at a starting point
                is larger than some tolerance.
    num_points : integer
                 Number of random points generated.

    Returns
    -------
    point_index : integer
                  Updated count of the number of starting points changed.
    x : 1-D array
        If the norm of the gradient at a starting point is greater than 
        tolerance, then the original starting point is used. Otherwise, the 
        starting point is changed until the norm of the gradient is greater 
        than tolerance.

    References
    ----------
    1) Herman et al, (2017), SALib: An open-source Python library for 
       Sensitivity Analysis, Journal of Open Source Software, 2(9), 97, doi:10.
       21105/joss.00097

    """
    while np.linalg.norm(g(x, *func_args)) < tolerance:
        warn('Norm of gradient at starting point is too small. A new '    
             'starting point will be used.', RuntimeWarning)
        point_index += 1
        if set_x == 'random':
            x = np.random.uniform(*bounds_set_x, (d, ))
        else:
            x = sobol_points[point_index + no_points]
        if point_index > int(num_points * 0.1):
            raise ValueError('Norm of the gradient at many starting points is' 
                             ' too small. Please change function parameters or'
                             ' set_x.')
    return point_index, x
