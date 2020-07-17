import numpy as np

import metod.metod_algorithm as mt_alg
import metod.objective_functions as mt_obj
import metod.metod_analysis as mt_ays


def compute_trajectories(num_points, d, projection, tolerance, option, met,
                         initial_guess, func_args, f, g, bounds_1, bounds_2,
                         usage, relax_sd_it):
    """Apply steepest descent iterations to each starting point, chosen
    uniformly at random from [0,1]^d. The number of starting points to
    generate is dependent on num_points.

    Parameters
    ----------
    num_points : integer
                 Total number of points to generate uniformly at random from
                 [0,1]^d.
    d : integer
        Size of dimension.
    projection : boolean
                 If projection is True, this projects points back to
                 [0, 1]^d. If projection is False, points are
                 kept the same.
    tolerance : integer or float
                Stopping condition for steepest descent iterations. Can
                either apply steepest descent iterations until the norm
                of g(point, *func_args) is less than some tolerance
                (usage = metod_algorithm) or until the total number of
                steepest descent iterations is greater than some
                tolerance (usage = metod_analysis).
    option : string
             Choose from 'minimize' or 'minimize_scalar'. For more
             information about each option see
             https://docs.scipy.org/doc/scipy/reference/optimize.html.
    met : string
          Choose method for option. For more information see
          - https://docs.scipy.org/doc/scipy/reference/generated/
          scipy.optimize.minimize.html#scipy.optimize.minimize
          - https://docs.scipy.org/doc/scipy/reference/generated/
          scipy.optimize.minimize_scalar.html#scipy.optimize.minimize_scalar
    initial_guess : float or integer
                    Initial guess passed to scipy.optimize.minimize. This
                    is recommended to be small.
    func_args : tuple
                Arguments passed to f and g.
    f : objective function.

        ``f(x, *func_args) -> float``

        where ``x`` is a 1-D array with shape(d, ) and func_args is a
        tuple of arguments needed to compute the function value.
    g : gradient of objective function.

       ``g(x, *func_args) -> 1-D array with shape (d, )``

        where ``x`` is a 1-D array with shape(d, ) and func_args is a
        tuple of arguments needed to compute the gradient.
    bounds_1 : integer
               Lower bound used for projection.
    bounds_2 : integer
               Upper bound used for projection.
    usage : string
            Used to decide stopping criterion for steepest descent
            iterations. Should be either usage == 'metod_algorithm' or
            usage == 'metod_analysis'.
    relax_sd_it : float or integer
                  Small constant in [0, 2] to multiply the step size by for a
                  steepest descent iteration. This process is known as relaxed
                  steepest descent [1].


    Returns
    -------
    store_x_values_list : list
                          Contains all trajectories from all random starting
                          points.
    store_minima : 1-D array
                   The region of attraction index of each trajectory.
    counter_non_match : integer
                        Total number of trajectories which belong to the
                        different regions of attraction.
    counter_match : integer
                    Total number of trajectories which belong to the same
                    region of attraction.

    References
    ----------
    1) Raydan, M., Svaiter, B.F.: Relaxed steepest descent and
       cauchy-barzilai- borwein method. Computational Optimization and
       Applications 21(2), 155–167 (2002)

    """
    store_x_values_list = []
    store_minima = np.zeros((num_points))
    for i in range((num_points)):
        x = np.random.uniform(0, 1, (d, ))
        points_x, its = (mt_alg.apply_sd_until_stopping_criteria
                         (x, d, projection, tolerance, option,
                          met, initial_guess, func_args, f, g,
                          bounds_1, bounds_2, usage, relax_sd_it))
        store_x_values_list.append(points_x)
        store_minima[i] = mt_obj.calc_pos(points_x[its].reshape(d, ),
                                          *func_args)[0]
        start_point_minima = mt_obj.calc_pos(points_x[0].reshape(d, ),
                                             *func_args)[0]
        assert(store_minima[i] == start_point_minima)
    counter_non_match = mt_ays.check_non_matchings(store_minima)
    counter_match = mt_ays.check_matchings(store_minima)
    return store_x_values_list, store_minima, counter_non_match, counter_match
