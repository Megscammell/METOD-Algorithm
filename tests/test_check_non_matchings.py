import numpy as np

import metod.metod_analysis as mt_ays


def test_1():
    """Computational test, where number of pairwise points with a
    different region of attraction index is 18.
    """
    store_minima = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1])
    counter_non_matches = 0
    num_points = 9
    for j in range((num_points)):
        minima_1 = store_minima[j]
        for k in range(j + 1, num_points):
            minima_2 = store_minima[k]
            if minima_1 != minima_2:
                counter_non_matches += 1
    counter_non_matches_2 = mt_ays.check_non_matchings(store_minima)
    assert(counter_non_matches == 18)
    assert(counter_non_matches_2 == 18)


def test_2():
    """Computational test, where number of pairwise points with a
    different region of attraction index is 6.
    """
    store_minima = np.array([1, 0, 0, 0, 1])
    num_points = 5
    counter_non_matches = 0
    for j in range((num_points)):
        minima_1 = store_minima[j]
        for k in range(j + 1, num_points):
            minima_2 = store_minima[k]
            if minima_1 != minima_2:
                counter_non_matches += 1
    counter_non_matches_2 = mt_ays.check_non_matchings(store_minima)
    assert(counter_non_matches == 6)
    assert(counter_non_matches_2 == 6)


def test_3():
    """Computational test, where number of pairwise points with a
    different region of attraction index is 2.
    """
    store_minima = np.array([1, 0, 1])
    num_points = 3
    counter_non_matches = 0
    for j in range((num_points)):
        minima_1 = store_minima[j]
        for k in range(j + 1, num_points):
            minima_2 = store_minima[k]
            if minima_1 != minima_2:
                counter_non_matches += 1
    counter_non_matches_2 = mt_ays.check_non_matchings(store_minima)
    assert(counter_non_matches == 2)
    assert(counter_non_matches_2 == 2)
