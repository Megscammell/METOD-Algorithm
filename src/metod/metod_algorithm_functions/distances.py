from numpy import linalg as LA


def distances(set_of_points, point, set_of_points_num, d,
              no_inequals_to_compare):
    """
    Compute the distances of each point in set_of_points with
    point.


    Parameters
    ----------
    set_of_points : 2-D array of shape (iterations + 1, d)
                    Contains all iterations of steepest descent of a
                    point.
    point : 1-D array of shape (d,).
    set_of_points_num : integer
                        The row number of set_of_points. This is used
                        as an index to start computing euclidean
                        distances. That is
                        ||set_of_points[set_of_points_num] -
                        point||,
                        ||set_of_points[set_of_points_num + 1]
                        - point||, where set_of_points_num is increased by 1
                        until the last row of set_of_points is reached.
    d : integer
        Size of dimension.
    no_inequals_to_compare : string
                             The number of inequalities to compute and
                             compare.

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
    if no_inequals_to_compare == 'All':
        set_of_points_dist = set_of_points[set_of_points_num:, :]
    elif no_inequals_to_compare == 'Two':
        set_of_points_dist = (set_of_points[set_of_points_num:set_of_points_num
                              + 2, :])
    set_of_points_dist_transpose = set_of_points_dist.T
    euclidean_distances = LA.norm(set_of_points_dist_transpose -
                                  point.reshape(d, 1), ord=2, axis=(0))
    return euclidean_distances
