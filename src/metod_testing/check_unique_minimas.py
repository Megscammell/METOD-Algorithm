
import itertools
from itertools import combinations

import numpy as np
from numpy import linalg as LA


def check_unique_minimas(discovered_minimas, const):
    """Checking minima of a point is not same as a local minima we have already discovered.

    Keyword arguments:
    discovered_minimas -- positions of each discovered minima
    const -- small positive constant
    """
    pos = np.arange(len(discovered_minimas))
    for pos_1, pos_2 in combinations(pos, 2):
        if np.all(discovered_minimas[pos_1] != None) and np.all(discovered_minimas[pos_2] != None):
            if LA.norm(discovered_minimas[pos_1] - discovered_minimas[pos_2]) < const:
                discovered_minimas[pos_2] = None
                
    unique_minimas = [element for element in discovered_minimas if np.all(element != None)]
    
    unique_number_minimas = len(unique_minimas)
    return unique_minimas, unique_number_minimas 
