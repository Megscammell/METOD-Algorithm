import numpy as np
from numpy import linalg as LA

def distances(set_of_points, point, set_of_points_num, d):
    """Compute Euclidean distances for each point in set_of_points with single_point

    Keyword arguments:
    set_of_points -- is an array of size N x d containing all iterations of steepest descent of a point
    point -- is a (d,) or d x 1 array
    set_of_points_num -- the row number of set_of_points to start computing distance with point
    d -- is dimension
    """

    set_of_points_dist = set_of_points[set_of_points_num:, :]
    set_of_points_dist_transpose = set_of_points_dist.T
    euclidean_distances = LA.norm(set_of_points_dist_transpose - 
                          point.reshape(d,1), ord = 2, axis=(0))
    return euclidean_distances
