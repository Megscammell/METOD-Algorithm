import numpy as np

import metod_testing as mtv3


def check_alg_cond(number_of_regions, x_1, z_1, x_2, z_2, x_points, z_points,
                   m, d):
    """Checks condition of algorithm for a point with all other

    Keyword arguments:
    number_of_regions -- number of regions of attraction identified
    x_1 -- point with k-1 iterations of steepest descent
    z_1 -- partner point associated with x_1
    x_2 -- point with k iterations of steepest descent
    z_2 -- partner point associated with x_2
    x_points -- Each array within this list are steepest descent iterations to
     a local minima.
    z_points -- Each array within this list is the corresponding partner
      points for each array in x_points.
    m -- warm up number
    d -- is dimension
    """
    possible_region_numbers = []
    for j in range(number_of_regions):
        iterations_of_sd_of_point = np.array(x_points[j])
        iterations_of_sd_of_partner_point = np.array(z_points[j])
        assert(iterations_of_sd_of_point.shape[0] ==
               iterations_of_sd_of_partner_point.shape[0])
        dist_x_1 = mtv3.distances(iterations_of_sd_of_point, x_1, m, d)
        dist_z_1 = mtv3.distances(iterations_of_sd_of_partner_point, z_1, m, d)
        dist_x_2 = mtv3.distances(iterations_of_sd_of_point, x_2, m, d)
        dist_z_2 = mtv3.distances(iterations_of_sd_of_partner_point, z_2, m, d)
        if all(dist_x_1 >= dist_z_1) and all(dist_x_2 >= dist_z_2):
            possible_region_numbers.append(j)
    return possible_region_numbers
