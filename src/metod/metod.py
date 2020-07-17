from warnings import warn
import numpy as np

from metod import metod_algorithm_functions as mt_alg


def metod(f, g, func_args, d, num_points=1000, beta=0.01,
          tolerance=0.00001, projection=False, const=0.1, m=3,
          option='minimize', met='Nelder-Mead', initial_guess=0.05,
          set_x=np.random.uniform, bounds_set_x=(0, 1),
          no_inequals_to_compare='All', usage='metod_algorithm',
          relax_sd_it=1):
    """Apply METOD algorithm with specified parameters.

    Parameters
    ----------
    f : objective function.

        ``f(x, *func_args) -> float``

        where ``x`` is a 1-D array with shape(d, ) and func_args is a
        tuple of arguments needed to compute the function value.
    g : gradient of objective function.

       ``g(x, *func_args) -> 1-D array with shape (d, )``

        where ``x`` is a 1-D array with shape(d, ) and func_args is a
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
                Stopping condition for steepest descent iterations. Can
                either apply steepest descent iterations until the norm
                of g(point, *func_args) is less than some tolerance
                (usage = metod_algorithm) or until the total number of
                steepest descent iterations is greater than some
                tolerance (usage = metod_analysis).
                The Default is tolerance=0.00001, as default
                usage=metod_algorithm.
    projection : boolean (optional)
                 If projection is True, this projects points back to
                 bounds_set_x. If projection is False, points are
                 kept the same. The Default is projection=False.
    const : float or integer (optional)
            In order to classify point x as a new local minima, the
            euclidean distance between x and all other discovered local
            minima must be larger than const. The Default is const=0.1.
    m : integer (optional)
        Number of iterations of steepest descent to apply to point
        x before making decision on terminating descents. The Default
        is m=3.
    option : string (optional)
             Choose from 'minimize' or 'minimize_scalar'. For more
             information about each option see
             https://docs.scipy.org/doc/scipy/reference/optimize.html.
             Default is 'minimize'.
    met : string (optional)
         Choose method for option. For more information see
         - https://docs.scipy.org/doc/scipy/reference/generated/
         scipy.optimize.minimize.html#scipy.optimize.minimize
         - https://docs.scipy.org/doc/scipy/reference/generated/
         scipy.optimize.minimize_scalar.html#scipy.optimize.minimize_scalar
         Default is 'Nelder-Mead'.
    initial_guess : float or integer (optional)
                    Initial guess passed to scipy.optimize.minimize. This
                    is recommended to be small. The default is
                    initial_guess=0.05.
    set_x : numpy.random distribution, list or np.ndarray (optional)
            If numpy.random distribution is selected, generates random
            starting points for the METOD algorithm. If list or a numpy
            array of size num_points is passed, then the METOD algorithm
            uses these points as staring points. The Default is
            set_x=np.random.uniform.
    bounds_set_x : tuple (optional)
                   Bounds for numpy.random distribution. The Default is
                   bounds_set_x=(0, 1).
    no_inequals_to_compare : string (optional)
                             Evaluate METOD algroithm condition with all
                             iterations ('All') or two iterations
                             ('Two'). Default is
                             no_inequals_to_compare='All'.
    usage : string (optional)
            Used to decide stopping criterion for steepest descent
            iterations. Should be either usage='metod_algorithm' or
            usage='metod_analysis'. Default is usage='metod_algorithm'.
    relax_sd_it : float or integer (optional)
                  Small constant in [0, 2] to multiply the step size by
                  for a steepest descent iteration. This process is
                  known as relaxed steepest descent [1]. Default is
                  relax_sd_it=1.


    Returns
    -------
    unique_minima : list
                    Contains all 1-D arrays with shape (d, ) of
                    unique minima.
    unique_number_of_minima: integer
                             Total number of unique minima found (L).
    func_vals_of_minimas : list
                           Function values at each unique minima.
    (len(des_x_points) - unique_number_of_minima)): integer
                                                    Number of excessive
                                                    descents.

    References
    ----------
    1) Raydan, M., Svaiter, B.F.: Relaxed steepest descent and
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
    if (no_inequals_to_compare != 'All') and (no_inequals_to_compare != 'Two'):
        raise ValueError('no_inequals_to_compare is not specified correctly.')
    if (usage != 'metod_algorithm') and (usage != 'metod_analysis'):
        raise ValueError('usage is not specified correctly.')
    if (type(relax_sd_it) is not int) and (type(relax_sd_it) is not float):
        raise ValueError('relax_sd_it must be a float.')
    if (usage == 'metod_algorithm') and (tolerance > 0.1):
        warn('Tolerance is too high and replaced with default.',
             RuntimeWarning)
        tolerance = 0.00001
    if (usage == 'metod_analysis') and (tolerance < 10):
        warn('Tolerance is too small and replaced with 10.',
             RuntimeWarning)
        tolerance = 10
    if d < 2:
        raise ValueError('must have d > 1')
    if m < 1:
        raise ValueError('must have m >= 1')
    if len(bounds_set_x) != 2:
        raise ValueError('length of bounds_set_x is less or greater than 2')
    if type(set_x) is list or type(set_x) is np.ndarray:
        num_points = len(set_x)
        projection = False
        bound_1 = None
        bound_2 = None
    elif set_x == np.random.uniform:
        bound_1 = bounds_set_x[0]
        bound_2 = bounds_set_x[1]
    if beta >= 1:
        warn('beta too high and would require that the largest eigenvalue is'
             ' smaller than one. Default beta is used.', RuntimeWarning)
        beta = 0.01
    if relax_sd_it < 0:
        raise ValueError('relax_sd_it is less than zero. This will change the'
                         ' direction of the steepest descent iteration.')
    des_x_points = []
    des_z_points = []
    discovered_minimas = []
    if type(set_x) is list or type(set_x) is np.ndarray:
        x = set_x[0]
        if x.shape[0] != d:
            raise ValueError('set_x does not contain points which are of size'
                             ' d')
    else:
        x = set_x(*bounds_set_x, (d, ))
    iterations_of_sd, its = mt_alg.apply_sd_until_stopping_criteria(
                            x, d, projection, tolerance, option, met,
                            initial_guess, func_args, f, g, bound_1, bound_2,
                            usage, relax_sd_it)
    if its <= m:
        raise ValueError('m is equal to or larger than the total number of '
                         'steepest descent iterations to find a minimizer. '
                         'Please change m or change tolerance.')
    des_x_points.append(iterations_of_sd)
    discovered_minimas.append(iterations_of_sd[its].reshape(d,))
    sd_iterations_partner_points = (mt_alg.partner_point_each_sd
                                    (iterations_of_sd, d, beta, its, g,
                                     func_args))
    des_z_points.append(sd_iterations_partner_points)
    number_minimas = 1
    for remaining_points in range(num_points - 1):
        if type(set_x) is list or type(set_x) is np.ndarray:
            x = set_x[remaining_points + 1]
            if x.shape[0] != d:
                raise ValueError('set_x does not contain points which are of '
                                 'size d')
        else:
            x = set_x(*bounds_set_x, (d, ))
        warm_up_sd, warm_up_sd_partner_points = (mt_alg.apply_sd_until_warm_up
                                                 (x, d, m, beta, projection,
                                                  option, met, initial_guess,
                                                  func_args, f, g, bound_1,
                                                  bound_2, relax_sd_it))
        x_1 = warm_up_sd[m - 1].reshape(d,)
        z_1 = warm_up_sd_partner_points[m - 1].reshape(d, )
        x_2 = warm_up_sd[m].reshape(d,)
        z_2 = warm_up_sd_partner_points[m].reshape(d,)
        possible_regions = mt_alg.check_alg_cond(number_minimas, x_1, z_1, x_2,
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
            discovered_minimas.append(iterations_of_sd[its + m].reshape(d, ))
            sd_iterations_partner_points = (mt_alg.partner_point_each_sd
                                            (iterations_of_sd, d, beta, its +
                                             m, g, func_args))
            des_z_points.append(sd_iterations_partner_points)
            number_minimas += 1
    unique_minima, unique_number_of_minima = (mt_alg.check_unique_minimas
                                              (discovered_minimas, const))
    func_vals_of_minimas = ([f(element, *func_args) for element in
                            unique_minima])
    return (unique_minima, unique_number_of_minima, func_vals_of_minimas,
            (len(des_x_points) - unique_number_of_minima))