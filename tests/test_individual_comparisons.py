import numpy as np

from metod_alg import metod_analysis as mt_ays
from metod_alg import metod_algorithm_functions as mt_alg


def test_1():
    """Example of how check inequals will work in individual_comparisons.py."""
    check_inequals = np.zeros((4))
    check_inequals[0] = 6 > 9
    check_inequals[1] = 5 > 4
    check_inequals[2] = 5 > 2
    check_inequals[3] = 5 > 3
    assert(np.all(check_inequals) == False)


def test_2():
    """Example of how check inequals will work in individual_comparisons.py."""
    check_inequals = np.zeros((4))
    check_inequals[0] = 5 > 1
    check_inequals[1] = 5 > 4
    check_inequals[2] = 5 > 2
    check_inequals[3] = 5 > 3
    assert(np.all(check_inequals) == True)


def test_3():
    """Testing outputs of individual_comparisons.py, where num = 1."""
    d = 2
    num = 1
    x_tr_1 = np.array([[1, 1],
                      [0.1, 0.3],
                      [0.15, 0.2],
                      [0.1, 0.1],
                      [1, 1.5]])

    z_tr_1 = np.array([[0.9, 0.9],
                      [0.2, 0.4],
                      [0.2, 0.8],
                      [0.9, 0.4],
                      [0.7, 0.4]])

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
    iterations = 4
    comparisons_check_2, total_check_2 = (mt_ays.individual_comparisons
                                          (d, x_tr_1, z_tr_1, x_tr_2, z_tr_2,
                                           iterations, num))
    assert(np.all(comparisons_check_2 == np.array([[1, 0, 1],
                                                  [1, 1, 1],
                                                  [1, 1, 1]])))
    assert(np.all(total_check_2 == np.array([[1, 1, 1],
                                            [1, 1, 1],
                                            [1, 1, 1]])))


def test_4():
    """Testing outputs of individual_comparisons.py, where num = 1."""
    d = 2
    num = 1
    x_tr_1 = np.array([[1, 1],
                      [0.1, 0.3],
                      [0.15, 0.2],
                      [0.1, 0.1]])

    z_tr_1 = np.array([[0.9, 0.9],
                      [0.3, 0.4],
                      [0.2, 0.3],
                      [0.9, 0.4]])

    x_tr_2 = np.array([[0.1, 0.4],
                      [0.3, 0.4],
                      [0.6, 0.7],
                      [0.2, 0.9]])

    z_tr_2 = np.array([[0.2, 0.1],
                      [0.35, 0.45],
                      [0.55, 0.75],
                      [0.1, 0.65]])
    tolerance = 3
    comparisons_check_2, total_check_2 = (mt_ays.individual_comparisons
                                          (d, x_tr_1, z_tr_1, x_tr_2, z_tr_2,
                                           tolerance, num))
    assert(np.all(comparisons_check_2 == np.array([[0, 0], [1, 1]])))
    assert(np.all(total_check_2 == np.array([[1, 1], [1, 1]])))


def test_5():
    """Testing outputs of individual_comparisons.py, where num = 2."""
    d = 2
    num = 2
    x_tr_1 = np.array([[1, 1],
                      [0.1, 0.3],
                      [0.15, 0.2],
                      [0.1, 0.1],
                      [1, 1.5]])

    z_tr_1 = np.array([[0.9, 0.9],
                      [0.2, 0.4],
                      [0.2, 0.8],
                      [0.9, 0.4],
                      [0.7, 0.4]])

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
    tolerance = 4
    comparisons_check_2, total_check_2 = (mt_ays.individual_comparisons
                                          (d, x_tr_1, z_tr_1, x_tr_2, z_tr_2,
                                           tolerance, num))
    assert(np.all(comparisons_check_2 == np.array([[1, 1], [1, 1]])))
    assert(np.all(total_check_2 == np.array([[1, 1], [1, 1]])))


def test_6():
    """Checking functionality of individual_comparisons.py."""
    num = 1
    d = 2
    i = 1
    iterations = 4
    no_inequals_to_compare = 'All'
    comparisons_check = np.zeros((1, iterations - num))
    total_checks = np.zeros((1, iterations - num))
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
    index = 0
    state = np.array([False, True, False, True, True, False, True, True,
                     False, False, True, False])
    for k in range(iterations - num):
        check_inequals = np.zeros((4))
        check_inequals[0] = (dist_squared_test_x_1[k] >=
                             dist_squared_test_z_1[k])
        assert(check_inequals[0] == state[index])
        index += 1
        check_inequals[1] = (dist_squared_test_x_1[k + 1] >=
                             dist_squared_test_z_1[k + 1])
        assert(check_inequals[1] == state[index])
        index += 1
        check_inequals[2] = (dist_squared_test_x_2[k] >=
                             dist_squared_test_z_2[k])
        assert(check_inequals[2] == state[index])
        index += 1
        check_inequals[3] = (dist_squared_test_x_2[k + 1] >=
                             dist_squared_test_z_2[k + 1])
        assert(check_inequals[3] == state[index])
        index += 1
        if np.all(check_inequals) is True:
            comparisons_check[i - num, k] = 0
            total_checks[i - num, k] += 1
        else:
            comparisons_check[i - num, k] += 1
            total_checks[i - num, k] += 1
    assert(np.all(comparisons_check == np.array([1, 1, 1])))
    assert(np.all(total_checks == np.array([1, 1, 1])))
