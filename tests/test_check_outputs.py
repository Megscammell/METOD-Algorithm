import numpy as np

from metod_alg import objective_functions as mt_obj


def test_1():
    d = 3
    minimizers_des = np.array([[ 2.746803, -2.903534, -2.903534],
                               [ 2.746803,  2.746803, -2.903534],
                               [-2.903534, -2.903534, -2.903534],
                               [-2.903534,  2.746803, -2.903534],
                               [ 2.746803, -2.903534,  2.746803],
                               [-2.903534,  2.746803,  2.746803],
                               [-2.903534, -2.903534,  2.746803]])
    number_of_minimizers = 7
    check_func = mt_obj.calc_minimizer_styb
    func_args = ()
    mt_obj.check_unique_minimizers(minimizers_des, number_of_minimizers,
                                   check_func, func_args)


def test_2():
    minimizers_metod = np.array([[ 2.746803, -2.903534, -2.903534],
                               [ 2.746803,  2.746803, -2.903534],
                               [-2.903534, -2.903534, -2.903534],
                               [-2.903534,  2.746803, -2.903534],
                               [ 2.746803, -2.903534,  2.746803]])

    minimizers_mult = np.array([[ 2.746803, -2.903534, -2.903534],
                                [ 2.746803,  2.746803, -2.903534],
                                [-2.903534, -2.903534, -2.903534],
                                [-2.903534,  2.746803, -2.903534],
                                [ 2.746803, -2.903534,  2.746803],
                                [-2.903534,  2.746803,  2.746803],
                                [-2.903534, -2.903534,  2.746803],
                                [ 2.746803,  2.746803,  2.746803]])                    
    mt_obj.check_minimizers_mult_metod(minimizers_metod,
                                       minimizers_mult) 