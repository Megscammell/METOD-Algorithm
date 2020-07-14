import numpy as np
from numpy import linalg as LA
from hypothesis import given, strategies as st

import metod.metod_algorithm as mt_alg


def test_1():
    """
    Simple computational example.
    """
    no_inequals_to_compare = 'All'
    x = np.array([0.1, 0.2, 0.5, 0.3, 0.6])
    test_points = np.array([[0, 0.1, 0.9, 0.7, 0.8],
                           [1, 0.9, 0.2, 0.3, 0.7]])
    dist = mt_alg.distances(test_points, x, 0, 5, no_inequals_to_compare)
    assert(np.round(dist[0], 5) == 0.61644)
    assert(np.round(dist[1], 5) == 1.18322)


@given(st.integers(3, 100), st.integers(1, 100))
def test_2(n, d):
    """
    Checks we get expected results from the distances function with
    a set of n points with dimension d and random set_number.
    """
    no_inequals_to_compare = 'All'
    set_number = np.random.randint(0, n - 1)
    points = np.random.uniform(0, 1, (n, d))
    x = np.random.uniform(0, 1, (d, ))
    dist_arr = np.zeros((n - set_number))
    for j in range(set_number, n):
        test = points[j, :].reshape(d, )
        cumulative = 0
        for i in range(d):
            cumulative += (test[i] - x[i]) ** 2
        norm = cumulative ** (0.5)
        dist_arr[j - set_number] = norm
    dist = mt_alg.distances(points, x, set_number, d, no_inequals_to_compare)
    assert(np.all(np.round(dist_arr, 10) == np.round(dist, 10)))
    assert(dist.shape[0] == n - set_number)


@given(st.integers(3, 100), st.integers(1, 100))
def test_3(n, d):
    """
    Same as test_2() but instead of using for loop to calculate euclidean
    distance, use LA.norm().
    """
    no_inequals_to_compare = 'All'
    set_number = np.random.randint(0, n - 1)
    points = np.random.uniform(0, 1, (n, d))
    x = np.random.uniform(0, 1, (d, ))
    dist_arr = np.zeros((n - set_number))
    for j in range(set_number, n):
        test = points[j, :].reshape(d,)
        norm = LA.norm(test - x)
        dist_arr[j - set_number] = norm
    dist = mt_alg.distances(points, x, set_number, d, no_inequals_to_compare)
    assert(np.all(np.round(dist_arr, 10) == np.round(dist, 10)))
    assert(dist.shape[0] == n - set_number)


def test_4():
    """
    Simple computational example.
    """
    no_inequals_to_compare = 'Two'
    x = np.array([0.1, 0.2, 0.5, 0.3, 0.6])
    test_points = np.array([[0, 0.1, 0.9, 0.7, 0.8],
                            [1, 0.9, 0.2, 0.3, 0.7]])
    dist = mt_alg.distances(test_points, x, 0, 5, no_inequals_to_compare)
    assert(np.round(dist[0], 5) == 0.61644)
    assert(np.round(dist[1], 5) == 1.18322)


@given(st.integers(5, 100), st.integers(1, 100))
def test_5(n, d):
    """
    This test ensures we get expected results from the distances function with
    a set of n points with dimension d and random set_number.
    """
    no_inequals_to_compare = 'Two'
    set_number = np.random.randint(0, n - 4)
    points = np.random.uniform(0, 1, (n, d))
    x = np.random.uniform(0, 1, (d,))
    dist_arr = np.zeros((2))
    for j in range(set_number, set_number + 2):
        test = points[j, :].reshape(d,)
        cumulative = 0
        for i in range(d):
            cumulative += (test[i] - x[i]) ** 2
        norm = cumulative ** (0.5)
        dist_arr[j - set_number] = norm
    dist = mt_alg.distances(points, x, set_number, d, no_inequals_to_compare)
    assert(np.all(np.round(dist_arr, 10) == np.round(dist, 10)))
    assert(dist.shape[0] == 2)


@given(st.integers(5, 100), st.integers(1, 100))
def test_6(n, d):
    """
    Same as test_2() but instead of using for loop to calculate euclidean
    distance, use LA.norm().
    """
    no_inequals_to_compare = 'Two'
    set_number = np.random.randint(0, n - 4)
    points = np.random.uniform(0, 1, (n, d))
    x = np.random.uniform(0, 1, (d, ))
    dist_arr = np.zeros((2))
    for j in range(set_number, set_number + 2):
        test = points[j, :].reshape(d, )
        norm = LA.norm(test - x)
        dist_arr[j - set_number] = norm
    dist = mt_alg.distances(points, x, set_number, d, no_inequals_to_compare)
    assert(np.all(np.round(dist_arr, 10) == np.round(dist, 10)))
    assert(dist.shape[0] == 2)
