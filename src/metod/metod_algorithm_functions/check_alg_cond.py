import numpy as np

from metod import metod_algorithm_functions as mt_alg


def check_alg_cond(number_of_regions, x_1, z_1, x_2, z_2, x_points, z_points,
                   m, d, no_inequals_to_compare):
    """Checks [1, Eq. 9] of the METOD algorithm.

    Parameters
    ----------
    number_of_regions : integer
                        Number of regions of attraction of local minimizers
                        identified.
    x_1 : 1-D array with shape (d, )
        Point with m-1 iterations of steepest descent.
    z_1 : 1-D array with shape (d, )
        Partner point associated with x_1.
    x_2 : 1-D array with shape (d, )
        Point with m iterations of steepest descent.
    z_2 : 1-D array with shape (d, )
        Partner point associated with x_2.
    x_points : list
                Each array within x_points are steepest descent
                iterations that have been continued until
                g(x, *func_args) is smaller than some tolerance, where
                x is a 1-D array with shape (d, ).
    z_points : list
                Each array within z_points are corresponding partner
                points for each array in x_points.
    m : integer
        Number of iterations of steepest descent to apply to point
        before making decision on terminating descents.
    d : integer
        Size of dimension.

    Returns
    -------
    possible_region_numbers : list
                                Contains the indices of the regions of
                                attraction of local minimizers, where
                                the METOD algorithm condition holds.

    References
    ----------
    1) Zilinskas, A., Gillard, J., Scammell, M., Zhigljavsky, A.: Multistart
       with early termination of descents. Journal of Global Optimization pp.
       1–16 (2019)

    """
    possible_region_numbers = []
    for j in range(number_of_regions):
        iterations_of_sd_of_point = np.array(x_points[j])
        iterations_of_sd_of_partner_point = np.array(z_points[j])
        assert(iterations_of_sd_of_point.shape[0] ==
               iterations_of_sd_of_partner_point.shape[0])
        dist_x_1 = mt_alg.distances(iterations_of_sd_of_point, x_1, m,
                                    d, no_inequals_to_compare)
        dist_z_1 = mt_alg.distances(iterations_of_sd_of_partner_point,
                                    z_1, m, d, no_inequals_to_compare)
        dist_x_2 = mt_alg.distances(iterations_of_sd_of_point, x_2, m,
                                    d, no_inequals_to_compare)
        dist_z_2 = mt_alg.distances(iterations_of_sd_of_partner_point,
                                    z_2, m, d, no_inequals_to_compare)
        if all(dist_x_1 >= dist_z_1) and all(dist_x_2 >= dist_z_2):
            possible_region_numbers.append(j)
    return possible_region_numbers
