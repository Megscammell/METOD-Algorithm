import numpy as np

from metod_alg import metod_algorithm_functions as mt_alg
from metod_alg import objective_functions as mt_obj
from metod_alg import check_metod_class as prev_mt_alg


def test_1():
    """
    Checks the outputs of prev_mt_alg.check_classification_sd_metod().
    """
    np.random.seed(100)
    p = 10
    d = 50
    lambda_1 = 1
    lambda_2 = 10
    store_A = np.zeros((p, d, d))
    store_x0 = np.zeros((p, d))
    store_rotation = np.zeros((p, d, d))
    for i in range(p):
        diag_vals = np.zeros(d)
        diag_vals[:2] = np.array([lambda_1, lambda_2])
        diag_vals[2:] = np.random.uniform(lambda_1 + 1,
                                          lambda_2 - 1, (d - 2))
        store_A[i] = np.diag(diag_vals)
        store_x0[i] = np.random.uniform(0, 1, (d))
        store_rotation[i] = mt_obj.calculate_rotation_matrix(d, 3)
    matrix_test = (np.transpose(store_rotation, (0, 2, 1)) @ store_A @
                   store_rotation)
    func_args = (p, store_x0, matrix_test)

    store_minimizers = np.zeros((30, d))
    for j in range(30):
        x = np.random.uniform(0, 1, (d, ))
        projection = False
        tolerance = 0.001
        option = 'minimize_scalar'
        met = 'brent'
        initial_guess = 0.005
        bound_1, bound_2 = 0, 1
        usage = 'metod_algorithm'
        relax_sd_it = 1
        f = mt_obj.several_quad_function
        g = mt_obj.several_quad_gradient
        check_func = mt_obj.calc_minimizer_sev_quad
        (iterations_of_sd,
         its,
         store_grad_part) = (mt_alg.apply_sd_until_stopping_criteria
                             (x, d, projection, tolerance,
                              option, met, initial_guess,
                              func_args, f, g, bound_1, bound_2,
                              usage, relax_sd_it, init_grad=None))
        store_minimizers[j] = iterations_of_sd[-1]

    class_store_x0 = np.array([1., 1., 1., 3., 1., 0., 1., 9., 6., 1.,
                               5., 9., 7., 1., 4., 1., 3., 0., 1., 3.,
                               5., 4., 1., 3., 0., 4., 1., 3., 3., 1.])
    assert(prev_mt_alg.check_classification_sd_metod(store_minimizers,
                                                     class_store_x0,
                                                     check_func,
                                                     func_args) == 0)
    class_store_x0 = np.array([1., 1., 0., 0., 0, 0., 1., 9., 6., 1.,
                               5., 9., 7., 1., 4., 1., 3., 0., 1., 3.,
                               5., 4., 1., 3., 0., 4., 1., 3., 3., 1.])
    assert(prev_mt_alg.check_classification_sd_metod(store_minimizers,
                                                     class_store_x0,
                                                     check_func,
                                                     func_args) == 0.1)
    class_store_x0 = np.array([2, 2, 0., 0., 0, 0., 1., 9., 6., 1., 5.,
                               9., 7., 1., 4., 1., 3., 0., 2, 3., 5., 4.,
                               1., 3., 0., 4., 1., 3., 3., 1.])
    assert(prev_mt_alg.check_classification_sd_metod(store_minimizers,
                                                     class_store_x0,
                                                     check_func,
                                                     func_args) == 0.2)
