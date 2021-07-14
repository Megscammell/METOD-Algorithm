import numpy as np


def check_des_points(iterations_of_sd, discovered_minimizers, const):
    """
    Check whether a local minimizer, found by applying local descent to a
    point, has already been identified by the METOD algorithm.

    Parameters
    ----------
    iterations_of_sd : 2-D array with shape (its, d), where 'its' is the number
                       of iterations of steepest descent.
    discovered_minimizers : list
                            Previously identified minimizers.
    const : float or integer
            In order to classify a point as a new local minimizer, the
            euclidean distance between the point and all other discovered
            local minimizers must be larger than const.

    Returns
    -------
    c : integer or None
        The region of attraction number which a local minimizer (found by
        applying local descent to a point) belongs to. If c = None, then a
        new region of attraction has been discovered.

    """
    minima_region = []
    all_minima_differences = []
    for j in range(len(discovered_minimizers)):
        dist = (np.linalg.norm(iterations_of_sd[-1] -
                               discovered_minimizers[j]))
        if (dist < const):
            minima_region.append(j)
            all_minima_differences.append(dist)

    if len(minima_region) == 1:
        c = minima_region[0]
        return c

    elif len(minima_region) >= 2:
        pos = np.argmin(np.array(all_minima_differences))
        c = minima_region[pos]
        return c

    else:
        return None
