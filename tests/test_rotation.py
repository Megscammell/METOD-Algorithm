import math
import numpy as np

import metod_testing as mtv3


def test_d_2():
    """Checks we get the same rotation matrix for d = 2"""
    rs = 10
    d = 2
    np.random.seed(rs)
    num = 3
    theta = np.random.uniform(0, 2 * math.pi)
    rotation = np.identity(d)

    rotation[0, 0] = math.cos(theta)
    rotation[0, 1] = - math.sin(theta)
    rotation[1, 0] = math.sin(theta)
    rotation[1, 1] = math.cos(theta)

    np.random.seed(rs)
    rotation_function = mtv3.calculate_rotation_matrix(d, num)
    assert(np.all(rotation == rotation_function))


def test_d_3():
    """
    Checks we get same rotation matrix for d = 3 and number of
    rotations = 3.
    """
    rs = 20
    d = 3
    np.random.seed(rs)
    number_rotations = 3

    theta_1 = np.random.uniform(0, 2 * math.pi)
    rotation_1 = np.identity(d)
    pos_1 = np.random.randint(0, d - 1)
    pos_2 = np.random.randint(pos_1 + 1, d)
    rotation_1[pos_1, pos_1] = math.cos(theta_1)
    rotation_1[pos_1, pos_2] = - math.sin(theta_1)
    rotation_1[pos_2, pos_1] = math.sin(theta_1)
    rotation_1[pos_2, pos_2] = math.cos(theta_1)

    theta_2 = np.random.uniform(0, 2 * math.pi)
    rotation_2 = np.identity(d)
    pos_3 = np.random.randint(0, d - 1)
    pos_4 = np.random.randint(pos_3 + 1, d)
    rotation_2[pos_3, pos_3] = math.cos(theta_2)
    rotation_2[pos_3, pos_4] = - math.sin(theta_2)
    rotation_2[pos_4, pos_3] = math.sin(theta_2)
    rotation_2[pos_4, pos_4] = math.cos(theta_2)

    theta_3 = np.random.uniform(0, 2 * math.pi)
    rotation_3 = np.identity(d)
    pos_5 = np.random.randint(0, d - 1)
    pos_6 = np.random.randint(pos_5 + 1, d)
    rotation_3[pos_5, pos_5] = math.cos(theta_3)
    rotation_3[pos_5, pos_6] = - math.sin(theta_3)
    rotation_3[pos_6, pos_5] = math.sin(theta_3)
    rotation_3[pos_6, pos_6] = math.cos(theta_3)

    final_rotation = rotation_1 @ rotation_2 @ rotation_3
    np.random.seed(rs)
    rotation_function = (mtv3.calculate_rotation_matrix
                         (d, number_rotations))
    assert(np.all(final_rotation == rotation_function))
