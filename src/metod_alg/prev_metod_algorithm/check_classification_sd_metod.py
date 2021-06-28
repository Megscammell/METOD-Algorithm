import numpy as np
from metod_alg import objective_functions as mt_obj

def check_classification_sd_metod(store_minimizer_des, class_store_x0,
                                  check_func, func_args):
    """
    Checks the proportion of times the classification of a point using the METOD
    algorithm is different to the true classification using Multistart.

    Parameters
    ----------
    store_minimizer_des : list
                          List containing local minimizers, found after
                          applying local descent at each starting point.
    class_store_x0 : 1-D array with shape (num_p,)
                     Array containing classifications for each starting
                     point from applying the METOD algorithm.
    func_args : tuple
                Arguments passed to a function f and gradient g.

    Returns
    -------
    prop_diff : float
                Proportion of times the classification of a point by the METOD
                algorithm is different to the true classification by
                Multistart.
    """

    store_class_des_mult = np.zeros((len(store_minimizer_des)))
    for j in range(len(store_minimizer_des)):
        store_class_des_mult[j] = (check_func(
                                   store_minimizer_des[j], *func_args))
    prop_diff = (np.where(store_class_des_mult != class_store_x0)[0].shape[0]
                 / len(store_class_des_mult))
    return prop_diff
