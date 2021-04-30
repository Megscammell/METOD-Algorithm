import numpy as np
from numpy import linalg as LA


def check_unique_minimizers(minimizers_des, number_of_minimizers,
                            check_func, func_args):
    """
    Checks the number of unique minimizers.

    Parameters
    ----------
    minimizers_des : list
                     List of local minimizers found from applying iterations
                     of descent.
    number_of_minimizers : integer
                           Total number of unique local minimizers found.
    check_func : function
                 Function used to check local minimizers. Will be dependent
                 on the objective function used.
    func_args : tuple
                Arguments passed to check_func.
    """
    check_minimizers = np.zeros((len(minimizers_des)))
    index = 0
    for minimizer in minimizers_des:
        pos_minimizer = check_func(minimizer, *func_args)
        check_minimizers[index] = pos_minimizer
        index += 1
    assert(np.unique(check_minimizers).shape[0] == 
           number_of_minimizers)


def check_minimizers_mult_metod(unique_minimizers_metod,
                                unique_minimizers_mult):
    """
    Checks that unique local minimizers found by METOD are also found by
    applying multistart.

    Parameters
    ----------
    unique_minimizers_metod : list
                              List of unique minimizers found by METOD.
    unique_minimizers_mult : list
                             List of unique minimizers found by multistart.
    """
    store_pos_for_mult = []
    for min_pos_1 in range(len(unique_minimizers_metod)):
        for min_pos_2 in range(len(unique_minimizers_mult)):
            if (np.linalg.norm(unique_minimizers_metod[min_pos_1] -
                              unique_minimizers_mult[min_pos_2]) < 0.1):
                store_pos_for_mult.append(min_pos_2)
    assert(np.unique(store_pos_for_mult).shape[0] == len(unique_minimizers_metod))
