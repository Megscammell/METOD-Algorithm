from itertools import combinations


def check_matchings(store_minimizer):
    """
    Checks the number of trajectories which belong to the same region of
    attraction.

    Parameters
    ----------
    store_minimizer : 1-D array
                      The region of attraction index for each trajectory.

    Returns
    -------
    counter_matchings : integer
                        Total number of trajectories which belong to the same
                        region of attraction.

    """
    counter_matchings = 0
    for min_1, min_2 in combinations(store_minimizer, 2):
        if min_1 == min_2:
            counter_matchings += 1
    return counter_matchings
