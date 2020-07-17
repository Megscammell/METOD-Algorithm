import numpy as np

import metod.metod_analysis as mt_ays


def test_1():
    """Computational test, where number of pairwise points with the
    region of attraction index is 18.
    """
    store_minima = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1])
    num_points = 9
    counter_matches = 0
    for j in range((num_points)):
        minima_1 = store_minima[j]
        for k in range(j + 1, num_points):
            minima_2 = store_minima[k]
            if minima_1 == minima_2:
                counter_matches += 1
    counter_matches_2 = mt_ays.check_matchings(store_minima)
    assert(counter_matches == 18)
    assert(counter_matches_2 == 18)


def test_2():
    """Computational test, where number of pairwise points with the
    same region of attraction index is 1.
    """
    store_minima = np.array([1, 0, 1])
    num_points = 3
    counter_matches = 0
    for j in range((num_points)):
        minima_1 = store_minima[j]
        for k in range(j + 1, num_points):
            minima_2 = store_minima[k]
            if minima_1 == minima_2:
                counter_matches += 1
    counter_matches_2 = mt_ays.check_matchings(store_minima)
    assert(counter_matches == 1)
    assert(counter_matches_2 == 1)


def test_3():
    """Computational test, where number of pairwise points with the
    same region of attraction index is 4.
    """
    store_minima = np.array([1, 0, 0, 0, 1])
    num_points = 5
    counter_matches = 0
    for j in range((num_points)):
        minima_1 = store_minima[j]
        for k in range(j + 1, num_points):
            minima_2 = store_minima[k]
            if minima_1 == minima_2:
                counter_matches += 1
    counter_matches_2 = mt_ays.check_matchings(store_minima)
    assert(counter_matches == 4)
    assert(counter_matches_2 == 4)
