import numpy as np

from metod_alg import metod_analysis as mt_ays


def test_1():
    """Computational test for mt_ays.check_matchings()."""
    store_minimizer = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1])
    num_points = 9
    counter_matches = 0
    for j in range((num_points)):
        minimizer_1 = store_minimizer[j]
        for k in range(j + 1, num_points):
            minimizer_2 = store_minimizer[k]
            if minimizer_1 == minimizer_2:
                counter_matches += 1
    counter_matches_2 = mt_ays.check_matchings(store_minimizer)
    assert(counter_matches == 18)
    assert(counter_matches_2 == 18)


def test_2():
    """Computational test for mt_ays.check_matchings()."""
    store_minimizer = np.array([1, 0, 1])
    num_points = 3
    counter_matches = 0
    for j in range((num_points)):
        minimizer_1 = store_minimizer[j]
        for k in range(j + 1, num_points):
            minimizer_2 = store_minimizer[k]
            if minimizer_1 == minimizer_2:
                counter_matches += 1
    counter_matches_2 = mt_ays.check_matchings(store_minimizer)
    assert(counter_matches == 1)
    assert(counter_matches_2 == 1)


def test_3():
    """Computational test for mt_ays.check_matchings()."""
    store_minimizer = np.array([1, 0, 0, 0, 1])
    num_points = 5
    counter_matches = 0
    for j in range((num_points)):
        minimizer_1 = store_minimizer[j]
        for k in range(j + 1, num_points):
            minimizer_2 = store_minimizer[k]
            if minimizer_1 == minimizer_2:
                counter_matches += 1
    counter_matches_2 = mt_ays.check_matchings(store_minimizer)
    assert(counter_matches == 4)
    assert(counter_matches_2 == 4)
