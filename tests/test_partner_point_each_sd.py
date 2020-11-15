import numpy as np
from hypothesis import given, settings, strategies as st

from metod import objective_functions as mt_obj
from metod import metod_algorithm_functions as mt_alg


def test_1():
    """Example to check computation of single partner point."""
    p = 2
    d = 2
    beta = 0.05
    g = mt_obj.quad_gradient
    matrix_test = np.zeros((p, d, d))
    store_x0 = np.zeros((p, d))
    matrix_test[0] = np.array([[2, 4], [4, 1]])
    matrix_test[1] = np.array([[2, 4], [4, 1]])
    x = np.array([2, 1]).reshape(1, d)
    store_x0[0] = np.array([3, 4])
    store_x0[1] = np.array([1, 0])
    func_args = p, store_x0, matrix_test
    partner_point_test = mt_alg.partner_point_each_sd(x, d, beta, 0, g,
                                                      func_args)
    assert(np.all(partner_point_test.reshape(d,) == np.array([1.7, 0.75])))


def test_2():
    """
    Check that for loop takes correct point from
    all_iterations_of_sd array and stores correctly into
    all_iterations_of_sd_test array.
    """
    iterations = 10
    d = 5
    all_iterations_of_sd = np.random.uniform(0, 1, (iterations + 1, d))
    all_iterations_of_sd_test = np.zeros((iterations + 1, d))
    for its in range(iterations + 1):
        point = all_iterations_of_sd[its, :].reshape((d, ))
        all_iterations_of_sd_test[its, :] = point.reshape(1, d)
    assert(np.all(all_iterations_of_sd_test == all_iterations_of_sd))


@settings(max_examples=50, deadline=None)
@given(st.integers(2, 10), st.integers(2, 100), st.integers(1, 30))
def test_3(p, d, iterations):
    """Ensure size of iterations_of_sd is the same as partner_points_sd."""
    beta = 0.005
    g = mt_obj.quad_function
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_test = mt_obj.function_parameters_quad(p, d, lambda_1,
                                                            lambda_2)
    func_args = p, store_x0, matrix_test
    iterations_of_sd = np.random.uniform(0, 1, (iterations + 1, d))
    partner_points_sd = mt_alg.partner_point_each_sd(iterations_of_sd, d, beta,
                                                     iterations, g, func_args)
    assert(partner_points_sd.shape[0] == iterations_of_sd.shape[0])
    assert(partner_points_sd.shape[1] == iterations_of_sd.shape[1])
