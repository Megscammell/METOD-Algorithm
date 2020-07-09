import numpy as np
from numpy import linalg as LA
from hypothesis import given, strategies as st

import metod_testing as mtv3


def test_1():
    """
    Simple computational example.
    """
    x = np.array([0.1, 0.2, 0.5, 0.3, 0.6])
    test_points = np.array([[0, 0.1, 0.9, 0.7, 0.8],
                           [1, 0.9, 0.2, 0.3, 0.7]])
    dist = mtv3.distances(test_points, x, 0, 5)
    assert(np.round(dist[0], 5) == 0.61644)
    assert(np.round(dist[1], 5) == 1.18322)


@given(st.integers(3, 100), st.integers(1, 100))
def test_2(n, d):
    """
    This test ensures we get expected results from the distances function with
     a set of n points with dimension d and random set_number.
    """
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
    dist = mtv3.distances(points, x, set_number, d)
    assert(np.all(np.round(dist_arr, 10) == np.round(dist, 10)))
    assert(dist.shape[0] == n - set_number)


@given(st.integers(3, 100), st.integers(1, 100))
def test_3(n, d):
    """
    Same as test_2() but instead of using for loop to calculate euclidean
     distance, use LA.norm().
    """
    set_number = np.random.randint(0, n - 1)
    points = np.random.uniform(0, 1, (n, d))
    x = np.random.uniform(0, 1, (d, ))
    dist_arr = np.zeros((n - set_number))
    for j in range(set_number, n):
        test = points[j, :].reshape(d,)
        norm = LA.norm(test - x)
        dist_arr[j - set_number] = norm
    dist = mtv3.distances(points, x, set_number, d)
    assert(np.all(np.round(dist_arr, 10) == np.round(dist, 10)))
    assert(dist.shape[0] == n - set_number)
