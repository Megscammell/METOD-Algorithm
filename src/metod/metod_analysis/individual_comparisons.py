import numpy as np

from metod import metod_algorithm_functions as mt_alg


def individual_comparisons(d, x_tr_1, z_tr_1, x_tr_2, z_tr_2, tolerance, num):
    """For trajectories x_i^(k_i), z_i^(k_i), x_j^(k_j) and z_j^(k_j), where
    k_i = (1,...,K_i) and k_j = (1,...,K_j), we observe where the METOD
    algroithm condition fails. A failure occurs when one of the following
    inequalities do not hold:
    - ||x_i^(k_i) - x_j^(k_j)|| >= ||z_i^(k_i) - z_j^(k_j)||
    - ||x_i^(k_i) - x_j^(k_j + 1)|| >= ||z_i^(k_i) - z_j^(k_j + 1)||
    - ||x_i^(k_i + 1) - x_j^(k_j)|| >= ||z_i^(k_i + 1) - z_j^(k_j)||
    - ||x_i^(k_i + 1) - x_j^(k_j + 1)|| >= ||z_i^(k_i + 1) - z_j^(k_j + 1)||
    If a failure does occur, then comparisons_check will obtain a 1 in the
    corresponding position, (k_i - num, k_j).

    Parameters
    ----------
    d : integer
        Size of dimension.
    x_tr_1 : 2-D array with shape (tolerance + 1, d)
             First array containing steepest descent iterations from a
             starting point.
    z_tr_1 : 2-D array with shape (tolerance + 1, d)
             Corresponding partner points for x_tr_1.
    x_tr_2 : 2-D array with shape (tolerance + 1, d)
             Second array containing steepest descent iterations from a
             starting point.
    z_tr_2 : 2-D array with shape (tolerance + 1, d)
             Corresponding partner points for x_tr_2.
    tolerance: integer
               Stopping condition for steepest descent iterations. Can
               either apply steepest descent iterations until the norm
               of g(point, *func_args) is less than some tolerance
               (usage = metod_algorithm) or until the total number of
               steepest descent iterations is greater than some
               tolerance (usage = metod_analysis).
    num: integer
         Iteration number to start comparing inequalities. E.g  k_i = (num,...,
         K_i) and k_j = (num,...,K_i).

    Returns
    -------
    comparisons_check : 2-D array with shape (tolerance - num, tolerance -
                                              num)
                        If inequalities fail at positions (k_i, k_j),
                        then a 1 will be placed in comparisons_check at
                        position (k_i - num, k_j).
    total_checks : 2-D array with shape (tolerance - num, tolerance -
                                         num)
                   If inequalities have been evaluated at positions
                   (k_i, k_j), then a 1 will be placed in total_checks
                   at position (k_i - num, k_j).

    """
    total_checks = np.zeros((tolerance - num, tolerance - num))
    comparisons_check = np.zeros((tolerance - num, tolerance - num))
    for i in range(num, tolerance):
        x_1_iteration = x_tr_1[i].reshape(d, )
        z_1_iteration = z_tr_1[i].reshape(d, )
        x_2_iteration = x_tr_1[i + 1].reshape(d, )
        z_2_iteration = z_tr_1[i + 1].reshape(d, )
        dist_squared_test_x_1 = mt_alg.distances(x_tr_2, x_1_iteration, num,
                                                 d, 'All')
        dist_squared_test_z_1 = mt_alg.distances(z_tr_2, z_1_iteration, num,
                                                 d, 'All')
        dist_squared_test_x_2 = mt_alg.distances(x_tr_2, x_2_iteration, num,
                                                 d, 'All')
        dist_squared_test_z_2 = mt_alg.distances(z_tr_2, z_2_iteration, num,
                                                 d, 'All')
        assert(dist_squared_test_x_1.shape[0] == (tolerance + 1) - num)
        assert(dist_squared_test_x_2.shape[0] == (tolerance + 1) - num)
        for k in range(tolerance - num):
            check_inequals = np.zeros((4))
            check_inequals[0] = (dist_squared_test_x_1[k] >=
                                 dist_squared_test_z_1[k])
            check_inequals[1] = (dist_squared_test_x_1[k + 1] >=
                                 dist_squared_test_z_1[k + 1])
            check_inequals[2] = (dist_squared_test_x_2[k] >=
                                 dist_squared_test_z_2[k])
            check_inequals[3] = (dist_squared_test_x_2[k + 1] >=
                                 dist_squared_test_z_2[k + 1])
            if np.all(check_inequals) == False:
                comparisons_check[i - num, k] += 1
                total_checks[i - num, k] += 1
            else:
                total_checks[i - num, k] += 1
    return comparisons_check, total_checks
