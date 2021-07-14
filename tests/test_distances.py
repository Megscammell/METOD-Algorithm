import numpy as np
from numpy import linalg as LA
from hypothesis import given, strategies as st

from metod_alg import metod_algorithm_functions as mt_alg


def test_1():
    """
    Simple computational example for mt_alg.distances().
    """
    no_inequals_to_compare = 'All'
    x = np.array([0.1, 0.2, 0.5, 0.3, 0.6])
    test_points = np.array([[0, 0.1, 0.9, 0.7, 0.8],
                           [1, 0.9, 0.2, 0.3, 0.7]])
    dist = mt_alg.distances(test_points, x, 0, 5, no_inequals_to_compare)
    assert(np.round(dist[0], 5) == 0.61644)
    assert(np.round(dist[1], 5) == 1.18322)


def test_2():
    """
    Simple computational example for mt_alg.distances().
    """
    num = 1
    d = 2
    no_inequals_to_compare = 'All'
    x_1_iteration = np.array([0.15, 0.2])
    z_1_iteration = np.array([0.9, 0.9])
    x_2_iteration = np.array([0.1, 0.3])
    z_2_iteration = np.array([0.2, 0.8])
    x_tr_2 = np.array([[0.1, 0.4],
                      [0.3, 0.4],
                      [0.6, 0.7],
                      [0.2, 0.9],
                      [0.7, 0.3]])
    z_tr_2 = np.array([[0.9, 0.1],
                       [0.35, 0.45],
                       [0.55, 0.75],
                       [0.1, 0.65],
                       [0.2, 0.1]])
    dist_squared_test_x_1 = mt_alg.distances(x_tr_2, x_1_iteration, num, d,
                                             no_inequals_to_compare)
    dist_squared_test_z_1 = mt_alg.distances(z_tr_2, z_1_iteration, num, d,
                                             no_inequals_to_compare)
    dist_squared_test_x_2 = mt_alg.distances(x_tr_2, x_2_iteration, num, d,
                                             no_inequals_to_compare)
    dist_squared_test_z_2 = mt_alg.distances(z_tr_2, z_2_iteration, num, d,
                                             no_inequals_to_compare)
    assert(np.all(np.round(dist_squared_test_x_1, 3) ==
           np.array([0.250, 0.673, 0.702, 0.559])))
    assert(np.all(np.round(dist_squared_test_z_1, 3) ==
           np.array([0.711, 0.381, 0.838, 1.063])))
    assert(np.all(np.round(dist_squared_test_x_2, 3) ==
           np.array([0.224, 0.640, 0.608, 0.600])))
    assert(np.all(np.round(dist_squared_test_z_2, 3) ==
           np.array([0.381, 0.354, 0.180, 0.700])))


@given(st.integers(3, 100), st.integers(1, 100))
def test_3(n, d):
    """
    Checks that expected results are returned from mt_alg.distances(),
    where no_inequals_to_compare = 'All' and the input is a set of n points
    with dimension d and random set_number.
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
def test_4(n, d):
    """
    Same as test_3() with the exception that np.linalg.norm will be used to
    compute distances.
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


def test_5():
    """
    Simple computational example for mt_alg.distances().
    """
    no_inequals_to_compare = 'Two'
    x = np.array([0.1, 0.2, 0.5, 0.3, 0.6])
    test_points = np.array([[0, 0.1, 0.9, 0.7, 0.8],
                            [1, 0.9, 0.2, 0.3, 0.7]])
    dist = mt_alg.distances(test_points, x, 0, 5, no_inequals_to_compare)
    assert(np.round(dist[0], 5) == 0.61644)
    assert(np.round(dist[1], 5) == 1.18322)


@given(st.integers(5, 100), st.integers(1, 100))
def test_6(n, d):
    """
    Checks that expected results are returned from mt_alg.distances(),
    where no_inequals_to_compare = 'Two' and the input is a set of n points
    with dimension d and random set_number.
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
def test_7(n, d):
    """
    Same as test_6() with the exception that np.linalg.norm will be used to
    compute distances.
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
