import math

import numpy as np


def calculate_rotation_matrix(d, num_rotations):
    """Generate rotation matrix of size d x d

    Parameters
    ----------
    d : integer
        Size of dimension.
    num_rotations : integer
                    Number of rotation matrices to genererate, then
                    multiply together to obatin overall rotation
                    matrix for d > 2.

    Returns
    -------
    rotation : 2-D array with shape (d, d)
               Rotation matrix

    """
    if d == 2:
        all_rotations = np.zeros((num_rotations, d, d))
        theta = np.random.uniform(0, 2 * math.pi)
        initial_rotation = np.zeros((d, d))
        initial_rotation[0, 0] = math.cos(theta)
        initial_rotation[0, 1] = -math.sin(theta)
        initial_rotation[1, 0] = math.sin(theta)
        initial_rotation[1, 1] = math.cos(theta)
        return initial_rotation

    else:
        all_rotations = np.zeros((num_rotations, d, d))
        for num in range(num_rotations):
            theta = np.random.uniform(0, 2 * math.pi)
            initial_rotation = np.identity(d)
            pos_1 = np.random.randint(0, d - 1)
            pos_2 = np.random.randint(pos_1 + 1, d)
            initial_rotation[pos_1, pos_1] = math.cos(theta)
            initial_rotation[pos_1, pos_2] = -math.sin(theta)
            initial_rotation[pos_2, pos_1] = math.sin(theta)
            initial_rotation[pos_2, pos_2] = math.cos(theta)
            all_rotations[num] = initial_rotation

        rotation = all_rotations[0]
        for i in range(1, num_rotations):
            rotation = rotation @ all_rotations[i]
        return rotation
