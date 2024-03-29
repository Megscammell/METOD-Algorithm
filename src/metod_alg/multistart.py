import numpy as np
import time

from metod_alg import metod_algorithm_functions as mt_alg


def multistart(f, g, func_args, d, starting_points, num_points,
               tolerance, projection, const, option, met, initial_guess,
               bounds_set_x, relax_sd_it):

    """
    Apply multistart with specified parameters. Purpose of code is to
    compare results with METOD. Hence, a list of starting points will
    need to be provided, which have been used by the METOD algorithm.

    Parameters
    ----------
    f : objective function.

        `f(x, *func_args) -> float`

        where `x` is a 1-D array with shape(d, ) and func_args is a
        tuple of arguments needed to compute the function value.
    g : gradient of objective function.

       `g(x, *func_args) -> 1-D array with shape (d, )`

        where `x` is a 1-D array with shape(d, ) and func_args is a
        tuple of arguments needed to compute the gradient.
    func_args : tuple
                Arguments passed to f and g.
    d : integer
        Size of dimension.
    starting_points : list
                      Starting points used by METOD. Same starting points will
                      be used for multistart to provide a fair comparison.
    num_points : integer
                 Number of random points generated.
    tolerance : integer or float
                Stopping condition for steepest descent iterations. Apply
                steepest descent iterations until the norm
                of g(point, *func_args) is less than some tolerance.
                Also check that the norm of the gradient at a starting point
                is larger than some tolerance.
    projection : boolean
                 If projection is True, points are projected back to
                 bounds_set_x. If projection is False, points are
                 kept the same.
    const : float or integer
            In order to classify a point as a new local minimizer, the
            euclidean distance between the point and all other discovered local
            minimizers must be larger than const.
    option : string
             Used to find the step size for each iteration of steepest
             descent.
             Choose from 'minimize' or 'minimize_scalar'. For more
             information about each option see
             https://docs.scipy.org/doc/scipy/reference/optimize.html.
    met : string
         Choose method for option. For more information see
         - https://docs.scipy.org/doc/scipy/reference/generated/
         scipy.optimize.minimize.html#scipy.optimize.minimize
         - https://docs.scipy.org/doc/scipy/reference/generated/
         scipy.optimize.minimize_scalar.html#scipy.optimize.minimize_scalar.
    initial_guess : float or integer
                    Initial guess passed to scipy.optimize.minimize and the
                    upper bound for the bracket interval when using the
                    'Brent' or 'Golden' method for
                    scipy.optimize.minimize_scalar. This
                    is recommended to be small.
    bounds_set_x : tuple
                   Bounds used for set x = 'random', set x='sobol' and
                   also for projection = True.
    relax_sd_it : float or integer
                  Multiply the step size by a small constant in [0, 2], to
                  obtain a new step size for steepest descent iterations. This
                  process is known as relaxed steepest descent [1].
    Returns
    -------
    unique_minimizers : list
                        Contains all unique minimizers.
    unique_number_of_minimizers: integer
                                 Total number of unique minimizers found.
    func_vals_of_minimizers : list
                              Function value at each unique minimizer.
    time taken_des : float
                     Time taken to find all local minimizers.
    copy_store_minimizer_des : list
                               List of all local minimizers found for each
                               starting point.
    no_grad_evals : 1-D array with shape (num_points,)
                    Array containing the number of gradient evaluations used
                    for each starting point.

    References
    ----------
    1) Raydan, M., Svaiter, B.F.: Relaxed steepest descent and
       cauchy-barzilai- borwein method. Computational Optimization and
       Applications 21(2), 155–167 (2002)

    """
    usage_choice = 'metod_algorithm'
    bound_1 = bounds_set_x[0]
    bound_2 = bounds_set_x[1]
    t0 = time.time()
    store_minimizer_des = []
    no_grad_evals = np.zeros((num_points))
    for j in (range(num_points)):
        (iterations_of_sd,
         its,
         store_grad) = (mt_alg.apply_sd_until_stopping_criteria
                        (starting_points[j].reshape(d, ),
                         d, projection,
                         tolerance, option,
                         met, initial_guess,
                         func_args, f, g,
                         bound_1,
                         bound_2,
                         usage_choice,
                         relax_sd_it, None))
        store_minimizer_des.append(iterations_of_sd[-1].reshape(d,))
        no_grad_evals[j] = len(store_grad)
    copy_store_minimizer_des = store_minimizer_des.copy()
    (unique_minimizers_mult,
     unique_number_of_minimizers_mult) = (mt_alg.check_unique_minimizers
                                          (store_minimizer_des, const))
    store_func_vals_mult = ([f(element, *func_args) for element in
                             unique_minimizers_mult])
    t1 = time.time()
    time_taken_des = t1-t0

    return (unique_minimizers_mult, unique_number_of_minimizers_mult,
            store_func_vals_mult, time_taken_des, copy_store_minimizer_des,
            no_grad_evals)
