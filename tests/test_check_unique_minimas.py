import numpy as np
from numpy import linalg as LA

import metod_testing as mtv3

def test_1():
    """Checking for loop version version of function with computational example
    """
    const = 0.1
    d = 3
    discovered_minimas = [np.array([0.1, 0.3, 0.5]).reshape(d,),
                         np.array([0.6, 0.9, 0.1]).reshape(d,),
                         np.array([0.11, 0.25, 0.48]).reshape(d,),
                         np.array([0.09, 0.25, 0.53]).reshape(d,),
                         np.array([0.58, 0.88, 0.09]).reshape(d,),
                         np.array([0.11, 0.27, 0.52]).reshape(d,)
                         ]
    for j in range(len(discovered_minimas)):
        if np.all(discovered_minimas[j] != None):
            for k in range(j + 1, len(discovered_minimas)):
                if np.all(discovered_minimas[k] != None):
                    if LA.norm(discovered_minimas[j] - discovered_minimas[k]) < const:
                        discovered_minimas[k] = None

    unique_minimas = []
    for n in range(len(discovered_minimas)):
        if np.all(discovered_minimas[n] != None):
            unique_minimas.append(discovered_minimas[n])
    
    unique_number_of_minimas = len(unique_minimas)
    test_unique_minimas, test_unique_number_of_minimas = mtv3.check_unique_minimas(discovered_minimas, const)
    assert(test_unique_minimas == unique_minimas)
    assert(test_unique_number_of_minimas == unique_number_of_minimas)
    assert(unique_number_of_minimas == 2)
    assert(np.all(test_unique_minimas[0] == np.array([0.1, 0.3, 0.5])))
    assert(np.all(test_unique_minimas[1] == np.array([0.6, 0.9, 0.1])))
                                   

def test_2():
    """Checking computational example
    """
    d = 2
    const = 0.1
    discovered_minimas = [np.array([0.1, 0.9]).reshape(d,),
                          np.array([0.8, 0.3]).reshape(d,),
                          np.array([0.11, 0.89]).reshape(d,),
                          np.array([0.09, 0.89]).reshape(d,),
                          np.array([0.81, 0.32]).reshape(d,),
                          np.array([0.79, 0.28]).reshape(d,)]
  
    unique_minimas, unique_number_of_minimas = mtv3.check_unique_minimas(discovered_minimas, const)
    assert(unique_number_of_minimas == 2)
    assert(np.all(unique_minimas[0] == np.array([0.1, 0.9])))
    assert(np.all(unique_minimas[1] == np.array([0.8, 0.3])))