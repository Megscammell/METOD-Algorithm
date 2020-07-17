from itertools import combinations


def check_non_matchings(store_minima):
    """Checks the number of trajectories which belong to different regions
    of attraction.

    Parameters
    ----------
    store_minima : 1-D array
                   The region of attraction index of each trajectory.

    Returns
    -------
    counter_non_matchings : integer
                            Total number of trajectories which belong to the
                            different regions of attraction.

    """
    counter_non_matchings = 0
    for min_1, min_2 in combinations(store_minima, 2):
        if min_1 != min_2:
            counter_non_matchings += 1
    return counter_non_matchings
