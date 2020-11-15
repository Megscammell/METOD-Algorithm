import numpy as np

from metod import metod_analysis as mt_ays


def test_1():
    """Computational test."""
    store_minimizer = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1])
    counter_non_matches = 0
    num_points = 9
    for j in range((num_points)):
        minimizer_1 = store_minimizer[j]
        for k in range(j + 1, num_points):
            minimizer_2 = store_minimizer[k]
            if minimizer_1 != minimizer_2:
                counter_non_matches += 1
    counter_non_matches_2 = mt_ays.check_non_matchings(store_minimizer)
    assert(counter_non_matches == 18)
    assert(counter_non_matches_2 == 18)


def test_2():
    """Computational test."""
    store_minimizer = np.array([1, 0, 0, 0, 1])
    num_points = 5
    counter_non_matches = 0
    for j in range((num_points)):
        minimizer_1 = store_minimizer[j]
        for k in range(j + 1, num_points):
            minimizer_2 = store_minimizer[k]
            if minimizer_1 != minimizer_2:
                counter_non_matches += 1
    counter_non_matches_2 = mt_ays.check_non_matchings(store_minimizer)
    assert(counter_non_matches == 6)
    assert(counter_non_matches_2 == 6)


def test_3():
    """Computational test.
    """
    store_minimizer = np.array([1, 0, 1])
    num_points = 3
    counter_non_matches = 0
    for j in range((num_points)):
        minimizer_1 = store_minimizer[j]
        for k in range(j + 1, num_points):
            minimizer_2 = store_minimizer[k]
            if minimizer_1 != minimizer_2:
                counter_non_matches += 1
    counter_non_matches_2 = mt_ays.check_non_matchings(store_minimizer)
    assert(counter_non_matches == 2)
    assert(counter_non_matches_2 == 2)
