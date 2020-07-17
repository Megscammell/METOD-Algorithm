import numpy as np

from metod import metod_algorithm_functions as mt_alg


def test_1():
    """ Tests that operator all() works as expected """
    dist_1 = np.array([1, 4, 6, 7])
    dist_2 = np.array([5, 6, 7, 8])
    dist_3 = np.array([9, 8, 7, 6])
    dist_4 = np.array([3, 91, 3, 34])
    assert(all(dist_2 > dist_1) is True)
    assert(all(dist_4 > dist_3) is False)
    assert(True and False is False)


def test_2():
    """Checks that the inequalities are satisfied for des_1_x and
     des_1_z only, when check_alg_condition is applied.
     Hence, possible_region_numbers will contain the 0 element only.
     """
    d = 5
    no_inequals_to_compare = 'All'
    l_regions_x = []
    l_regions_z = []
    des_1_x = np.array([[1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 0]])
    des_2_x = np.array([[3, 7, 3, 9, 5],
                        [3, 5, 4, 4, 1]])
    des_1_z = np.array([[2, 3, 4, 5, 6],
                        [5, 2, 3, 4, 5]])
    des_2_z = np.array([[2, 5, 2, 1, 2],
                        [2, 5, 1, 1, 2]])
    l_regions_x.append(des_1_x)
    l_regions_x.append(des_2_x)
    l_regions_z.append(des_1_z)
    l_regions_z.append(des_2_z)
    x_1 = np.array([5, 6, 5, 6, 5]).reshape(d, )
    z_1 = np.array([4, 5, 4, 5, 4]).reshape(d, )
    x_2 = np.array([4, 5, 4, 5, 4]).reshape(d, )
    z_2 = np.array([3, 5, 2, 3, 6]).reshape(d, )
    possible_region_numbers = mt_alg.check_alg_cond(2, x_1, z_1, x_2, z_2,
                                                    l_regions_x, l_regions_z,
                                                    0, d,
                                                    no_inequals_to_compare)
    assert(possible_region_numbers == [0])


def test_3():
    """Checks that the inequalities are satisfied for des_1_x and
     des_1_z only, when check_alg_condition is applied.
     Hence, possible_region_numbers will contain the 0 element only.
     """
    d = 5
    no_inequals_to_compare = 'Two'
    l_regions_x = []
    l_regions_z = []
    des_1_x = np.array([[1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 0]])
    des_2_x = np.array([[3, 7, 3, 9, 5],
                        [3, 5, 4, 4, 1]])
    des_1_z = np.array([[2, 3, 4, 5, 6],
                        [5, 2, 3, 4, 5]])
    des_2_z = np.array([[2, 5, 2, 1, 2],
                        [2, 5, 1, 1, 2]])
    l_regions_x.append(des_1_x)
    l_regions_x.append(des_2_x)
    l_regions_z.append(des_1_z)
    l_regions_z.append(des_2_z)
    x_1 = np.array([5, 6, 5, 6, 5]).reshape(d, )
    z_1 = np.array([4, 5, 4, 5, 4]).reshape(d, )
    x_2 = np.array([4, 5, 4, 5, 4]).reshape(d, )
    z_2 = np.array([3, 5, 2, 3, 6]).reshape(d, )
    possible_region_numbers = mt_alg.check_alg_cond(2, x_1, z_1, x_2, z_2,
                                                    l_regions_x, l_regions_z,
                                                    0, d,
                                                    no_inequals_to_compare)
    assert(possible_region_numbers == [0])
