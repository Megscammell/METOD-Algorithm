from itertools import combinations

import numpy as np
from numpy import linalg as LA


def check_unique_minimas(discovered_minimas, const):
    """Find all unique minimizers

    Parameters
    ----------
    discovered_minimas : list
                         Each discovered local minima.
    const : float or integer
            In order to classify a point as a new local minima, the
            euclidean distance between a point and all other discovered
            local minima must be larger than const. The Default is
            const=0.1.

    Returns
    -------
    unique_minima : list
                    Contains all 1-D arrays with shape (d, ) of
                    unique minima.
    unique_number_minimas: integer
                           Total number of unique local minima.

    """
    pos = np.arange(len(discovered_minimas))
    for pos_1, pos_2 in combinations(pos, 2):
        if (np.all(discovered_minimas[pos_1] is not None) and
                np.all(discovered_minimas[pos_2] is not None)):
            if (LA.norm(discovered_minimas[pos_1] - discovered_minimas
                        [pos_2]) < const):
                discovered_minimas[pos_2] = None
    unique_minimas = ([element for element in discovered_minimas if np.all
                      (element is not None)])
    unique_number_minimas = len(unique_minimas)
    return unique_minimas, unique_number_minimas
