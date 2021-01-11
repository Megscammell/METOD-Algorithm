from warnings import warn
import numpy as np

from metod import metod_algorithm_functions as mt_alg


def metod(f, g, func_args, d, num_points=1000, beta=0.01,
          tolerance=0.00001, projection=False, const=0.1, m=3,
          option='minimize_scalar', met='Brent', initial_guess=0.005,
          set_x='sobol', bounds_set_x=(0, 1), relax_sd_it=1):

    """Apply METOD algorithm with specified parameters.

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
    num_points : integer (optional)
                 Number of random points generated. The Default is
                 num_points=1000.
    beta : float or integer (optional)
           Small constant step size to compute the partner points.
           The Default is beta=0.01.
    tolerance : integer or float (optional)
                Stopping condition for steepest descent iterations. Apply
                steepest descent iterations until the norm
                of g(point, *func_args) is less than some tolerance.
                Also check that the norm of the gradient at a starting point
                is larger than some tolerance.
                The Default is tolerance=0.00001.
    projection : boolean (optional)
                 If projection is True, points are projected back to
                 bounds_set_x. If projection is False, points are
                 kept the same. The Default is projection=False.
    const : float or integer (optional)
            In order to classify point x as a new local minimizer, the
            euclidean distance between x and all other discovered local
            minimizers must be larger than const. The Default is const=0.1.
    m : integer (optional)
        Number of iterations of steepest descent to apply to point
        x before making decision on terminating descents. The Default
        is m=3.
    option : string (optional)
             Choose from 'minimize' or 'minimize_scalar'. For more
             information about each option see
             https://docs.scipy.org/doc/scipy/reference/optimize.html.
             Default is 'minimize_scalar'.
    met : string (optional)
         Choose method for option. For more information see
         - https://docs.scipy.org/doc/scipy/reference/generated/
         scipy.optimize.minimize.html#scipy.optimize.minimize
         - https://docs.scipy.org/doc/scipy/reference/generated/
         scipy.optimize.minimize_scalar.html#scipy.optimize.minimize_scalar
         Default is 'Brent'.
    initial_guess : float or integer (optional)
                    Initial guess passed to scipy.optimize.minimize and the
                    upper bound for the bracket interval when using the
                    'Brent' or 'Golden' method for
                    scipy.optimize.minimize_scalar. This
                    is recommended to be small. The default is
                    initial_guess=0.005.
    set_x : string (optional)
            If set_x = 'random', random starting points
            are generated for the METOD algorithm. If set_x = 'sobol'
            is selected, then a numpy.array with shape
            (num points * 2, d) of Sobol sequence samples are generated
            using SALib [1], which are randomly shuffled and used
            as starting points for the METOD algorithm. The Default is
            set_x = 'sobol'.
    bounds_set_x : tuple (optional)
                   Bounds used for set x = 'random', set x='sobol' and
                   also for projection = True. The Default is
                   bounds_set_x=(0, 1).
    relax_sd_it : float or integer (optional)
                  Multiply the step size by a small constant in [0, 2], to
                  obtain a new step size for steepest descent iterations. This
                  process is known as relaxed steepest descent [2].

    Returns
    -------
    unique_minimizers : list
                        Contains all 1-D arrays with shape (d, ) of
                        unique minimizers.
    unique_number_of_minimizers: integer
                                 Total number of unique minimizers found.
    func_vals_of_minimizers : list
                              Function value at each unique minimizer.
    excessive_descents: integer
                        Number of excessive descents.
    starting_points: list
                     Starting points used by the METOD algorithm.

    References
    ----------
    1) Herman et al, (2017), SALib: An open-source Python library for 
       Sensitivity Analysis, Journal of Open Source Software, 2(9), 97, doi:10.
       21105/joss.00097
    2) Raydan, M., Svaiter, B.F.: Relaxed steepest descent and
       cauchy-barzilai- borwein method. Computational Optimization and
       Applications 21(2), 155–167 (2002)

    """
    if type(d) is not int:
        raise ValueError('d must be an integer.')
    if type(num_points) is not int:
        raise ValueError('num_points must be an integer.')
    if (type(beta) is not int) and (type(beta) is not float):
        raise ValueError('beta must be an integer or float.')
    if (type(tolerance) is not int) and (type(tolerance) is not float):
        raise ValueError('tolerance must be a float or integer.')
    if type(projection) is not bool:
        raise ValueError('projection must be boolean.')
    if (type(const) is not int) and (type(const) is not float):
        raise ValueError('const must be an integer or float.')
    if type(m) is not int:
        raise ValueError('m must be an integer.')
    if type(option) is not str:
        raise ValueError('option must be a string.')
    if type(met) is not str:
        raise ValueError('met must be a string.')
    if (type(initial_guess) is not int) and (type(initial_guess) is not float):
        raise ValueError('initial_guess must be a float.')
    if ((type(bounds_set_x[0]) is not int) and (type(bounds_set_x[0]) is not
                                                float)):
        raise ValueError('bounds_set_x does not contain bounds which are'
                         ' floats or integers .')
    if ((type(bounds_set_x[1]) is not int) and (type(bounds_set_x[1]) is not
                                                float)):
        raise ValueError('bounds_set_x does not contain bounds which are'
                         ' floats or integers .')
    if (type(relax_sd_it) is not int) and (type(relax_sd_it) is not float):
        raise ValueError('relax_sd_it must be a float.')
    if d < 2:
        raise ValueError('Must have d > 1')
    if m < 1:
        raise ValueError('Must have m >= 1')
    if len(bounds_set_x) != 2:
        raise ValueError('Length of bounds_set_x is less or greater than 2')
    if type(set_x) is not str:
        raise ValueError('set_x must be a string.')
    if (set_x != 'random' and 
        set_x != 'sobol'):
        raise ValueError('Please select valid set_x.')
    if beta >= 1:
        warn('beta too high and would require that the largest eigenvalue is'
             ' smaller than one. Default beta is used.', RuntimeWarning)
        beta = 0.01
    if tolerance > 0.1:
        warn('tolerance is too high.'
             'Default tolerance is used.', RuntimeWarning)
        tolerance = 0.00001
    if relax_sd_it < 0:
        raise ValueError('relax_sd_it is less than zero. This will change the'
                         ' direction of descent.')
    usage = 'metod_algorithm'
    no_inequals_to_compare = 'All'
    bound_1 = bounds_set_x[0]
    bound_2 = bounds_set_x[1]
    des_x_points = []
    des_z_points = []
    discovered_minimizers = []
    starting_points = []
    if set_x == 'random':
        sobol_points = None
        x = np.random.uniform(*bounds_set_x, (d, ))
    else:
        sobol_points = mt_alg.create_sobol_sequence_points(bound_1, 
                                                           bound_2, 
                                                           d, num_points)
        x = sobol_points[0]
    point_index = 0
    point_index, x = (mt_alg.check_grad_starting_point
                      (x, point_index, 0, bounds_set_x, sobol_points, d, g,
                       func_args, set_x, tolerance, num_points))
    starting_points.append(x)
    iterations_of_sd, its = mt_alg.apply_sd_until_stopping_criteria(
                            x, d, projection, tolerance, option, met,
                            initial_guess, func_args, f, g, bound_1, bound_2,
                            usage, relax_sd_it)
    if its <= m:
        raise ValueError('m is equal to or larger than the total number of '
                         'steepest descent iterations to find a minimizer. '
                         'Please change m or change tolerance.')
    des_x_points.append(iterations_of_sd)
    discovered_minimizers.append(iterations_of_sd[its].reshape(d,))
    sd_iterations_partner_points = (mt_alg.partner_point_each_sd
                                    (iterations_of_sd, d, beta, its, g,
                                     func_args))
    des_z_points.append(sd_iterations_partner_points)
    number_minima = 1
    for remaining_points in (range(num_points - 1)):
        if set_x == 'random':
            x = np.random.uniform(*bounds_set_x, (d, ))
        else:
            pos = (point_index + (remaining_points + 1))
            x = sobol_points[pos]

        point_index, x = (mt_alg.check_grad_starting_point
                          (x, point_index, remaining_points + 1, bounds_set_x,
                           sobol_points, d,  g, func_args, set_x, tolerance,
                           num_points))
        starting_points.append(x)
        warm_up_sd, warm_up_sd_partner_points = (mt_alg.apply_sd_until_warm_up
                                                 (x, d, m, beta, projection,
                                                  option, met, initial_guess,
                                                  func_args, f, g, bound_1,
                                                  bound_2, relax_sd_it))
        x_1 = warm_up_sd[m - 1].reshape(d,)
        z_1 = warm_up_sd_partner_points[m - 1].reshape(d, )
        x_2 = warm_up_sd[m].reshape(d,)
        z_2 = warm_up_sd_partner_points[m].reshape(d,)
        possible_regions = mt_alg.check_alg_cond(number_minima, x_1, z_1, x_2,
                                                 z_2, des_x_points,
                                                 des_z_points, m - 1, d,
                                                 no_inequals_to_compare)
        if possible_regions == []:
            iterations_of_sd_part, its = (mt_alg.
                                          apply_sd_until_stopping_criteria
                                          (x_2, d, projection, tolerance,
                                           option, met, initial_guess,
                                           func_args, f, g, bound_1, bound_2,
                                           usage, relax_sd_it))
            iterations_of_sd = np.vstack([warm_up_sd,
                                          iterations_of_sd_part[1:, ]])
            des_x_points.append(iterations_of_sd)
            discovered_minimizers.append(iterations_of_sd[its + m].reshape(d, ))
            sd_iterations_partner_points = (mt_alg.partner_point_each_sd
                                            (iterations_of_sd, d, beta, its +
                                             m, g, func_args))
            des_z_points.append(sd_iterations_partner_points)
            number_minima += 1
    (unique_minimizers,
     unique_number_of_minimizers) = (mt_alg.check_unique_minimizers
                                     (discovered_minimizers, const))
    func_vals_of_minimizers = ([f(element, *func_args) for element in
                                unique_minimizers])
    excessive_descents = (len(des_x_points) - unique_number_of_minimizers)
    return (unique_minimizers, unique_number_of_minimizers,
            func_vals_of_minimizers, excessive_descents, starting_points)
