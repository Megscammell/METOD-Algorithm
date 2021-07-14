from itertools import combinations

import numpy as np
from numpy import linalg as LA


def check_unique_minimizers(discovered_minimizers, const):
    """
    Finds all unique minimizers.

    Parameters
    ----------
    discovered_minimizers : list
                            Each discovered local minimizer.
    const : float or integer
            In order to classify a point as a new local minimizer, the
            euclidean distance between the point and all other discovered
            local minimizers must be larger than const.

    Returns
    -------
    unique_minimizers : list
                        Contains all unique local minimizers.
    unique_number_minimizers: integer
                              Total number of unique local minimizers.

    """
    pos = np.arange(len(discovered_minimizers))
    for pos_1, pos_2 in combinations(pos, 2):
        if (np.all(discovered_minimizers[pos_1] is not None) and
                np.all(discovered_minimizers[pos_2] is not None)):
            if (LA.norm(discovered_minimizers[pos_1] - discovered_minimizers
                        [pos_2]) < const):
                discovered_minimizers[pos_2] = None
    unique_minimizers = ([element for element in discovered_minimizers
                          if np.all(element is not None)])
    unique_number_minimizers = len(unique_minimizers)
    return unique_minimizers, unique_number_minimizers
