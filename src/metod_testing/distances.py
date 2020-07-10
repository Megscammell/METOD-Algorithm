from numpy import linalg as LA


def distances(set_of_points, point, set_of_points_num, d):
    """Compute Euclidean distances for each point in set_of_points with
     single_point.


    Parameters
    ----------
    set_of_points : 2-D array of shape (iterations + 1, d)
                    Contains all iterations of steepest descent of a
                    point.
    point : 1-D array of shape (d,)
    set_of_points_num : integer
                        The row number of set_of_points. This is used
                        as an index to start from to compute euclidean
                        distances. That is
                        ||set_of_points[set_of_points_num] -
                        point||,
                        ||set_of_points[set_of_points_num + 1]
                        - point||, and so on until the last row
                        of set_of_points is reached.
    d : integer
        Size of dimension.

    Returns
    -------
    euclidean_distances : 1-D array of shape ((set_of_points.shape[0] -
                                               set_of_points_num), )
                          Array contains all euclidean distances,
                          ||set_of_points[set_of_points_num] - point||,
                          ||set_of_points[set_of_points_num + 1] - point||,
                          ...,
                          ||set_of_points[set_of_points.shape[0] - 1] - point||
    """

    set_of_points_dist = set_of_points[set_of_points_num:, :]
    set_of_points_dist_transpose = set_of_points_dist.T
    euclidean_distances = LA.norm(set_of_points_dist_transpose -
                                  point.reshape(d, 1), ord=2, axis=(0))
    return euclidean_distances
