import numpy as np
from numpy import linalg as LA
from itertools import combinations

from metod_alg import metod_algorithm_functions as mt_alg


def test_1():
    """
    Checking for loop version version of mt_alg.check_unique_minimizers()
    with computational example.
    """
    const = 0.1
    d = 3
    discovered_minimizers = [np.array([0.1, 0.3, 0.5]).reshape(d, ),
                             np.array([0.6, 0.9, 0.1]).reshape(d, ),
                             np.array([0.11, 0.25, 0.48]).reshape(d, ),
                             np.array([0.09, 0.25, 0.53]).reshape(d, ),
                             np.array([0.58, 0.88, 0.09]).reshape(d, ),
                             np.array([0.11, 0.27, 0.52]).reshape(d, )]
    for j in range(len(discovered_minimizers)):
        if np.all(discovered_minimizers[j] is not None):
            for k in range(j + 1, len(discovered_minimizers)):
                if np.all(discovered_minimizers[k] is not None):
                    if (LA.norm(discovered_minimizers[j] -
                                discovered_minimizers[k]) < const):
                        discovered_minimizers[k] = None
    unique_minimizers = []
    for n in range(len(discovered_minimizers)):
        if np.all(discovered_minimizers[n] is not None):
            unique_minimizers.append(discovered_minimizers[n])
    unique_number_of_minimizers = len(unique_minimizers)
    (test_unique_minimizers,
     test_unique_number_of_minimizers) = (mt_alg.check_unique_minimizers
                                          (discovered_minimizers,
                                           const))
    assert(test_unique_minimizers == unique_minimizers)
    assert(test_unique_number_of_minimizers == unique_number_of_minimizers)
    assert(unique_number_of_minimizers == 2)
    assert(np.all(test_unique_minimizers[0] == np.array([0.1, 0.3, 0.5])))
    assert(np.all(test_unique_minimizers[1] == np.array([0.6, 0.9, 0.1])))


def test_2():
    """Computational example for mt_alg.check_unique_minimizers()"""
    d = 2
    const = 0.1
    discovered_minimizers = [np.array([0.1, 0.9]).reshape(d, ),
                             np.array([0.8, 0.3]).reshape(d, ),
                             np.array([0.11, 0.89]).reshape(d, ),
                             np.array([0.09, 0.89]).reshape(d, ),
                             np.array([0.81, 0.32]).reshape(d, ),
                             np.array([0.79, 0.28]).reshape(d, )]
    (unique_minimizers,
     unique_number_of_minimizers) = (mt_alg.check_unique_minimizers
                                     (discovered_minimizers, const))
    assert(unique_number_of_minimizers == 2)
    assert(np.all(unique_minimizers[0] == np.array([0.1, 0.9])))
    assert(np.all(unique_minimizers[1] == np.array([0.8, 0.3])))


def test_3():
    """Ensure that all combinations are explored."""
    test_list = [1, 2, 1, 5, 7, 3, 9, 10]
    combos_total = int((len(test_list) * (len(test_list) - 1))/2)
    all_combos = np.zeros((combos_total, 2))
    pos = np.arange(len(test_list))
    index = 0
    for pos_1, pos_2 in combinations(pos, 2):
        all_combos[index, 0] = test_list[pos_1]
        all_combos[index, 1] = test_list[pos_2]
        index += 1
    assert(all_combos[combos_total - 1, 0] == test_list[len(test_list) - 2])
    assert(all_combos[combos_total - 1, 1] == test_list[len(test_list) - 1])
