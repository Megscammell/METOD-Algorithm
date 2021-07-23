import numpy as np
from hypothesis import given, settings, strategies as st

from metod_alg import objective_functions as mt_obj
from metod_alg import metod_algorithm_functions as mt_alg


def test_1():
    """Example to check mt_alg.partner_point_each_sd()."""
    p = 2
    d = 2
    beta = 0.05
    g = mt_obj.several_quad_gradient
    matrix_test = np.zeros((p, d, d))
    store_x0 = np.zeros((p, d))
    matrix_test[0] = np.array([[2, 4], [4, 1]])
    matrix_test[1] = np.array([[2, 4], [4, 1]])
    x = np.array([2, 1]).reshape(1, d)
    store_x0[0] = np.array([3, 4])
    store_x0[1] = np.array([1, 0])
    func_args = p, store_x0, matrix_test
    store_grad = g(x.reshape(d,), *func_args).reshape(1, d)
    partner_point_test = mt_alg.partner_point_each_sd(x, beta,
                                                      store_grad)
    assert(np.all(partner_point_test.reshape(d,) == np.array([1.7, 0.75])))


def test_2():
    """
    Check numpy array addition and multiplication in partner_point_each_sd.py
    code.
    """
    store_x = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])

    store_grad = np.ones((3, 3))
    beta = 0.1
    partner_point_test = mt_alg.partner_point_each_sd(store_x, beta,
                                                      store_grad)
    assert(np.all(partner_point_test == np.array([[0.9, 1.9, 2.9],
                                                  [3.9, 4.9, 5.9],
                                                  [6.9, 7.9, 8.9]])))


@settings(max_examples=50, deadline=None)
@given(st.integers(2, 10), st.integers(2, 100), st.integers(1, 30))
def test_3(p, d, iterations):
    """
    Ensure size of iterations_of_sd is the same as partner_points_sd
    from mt_alg.partner_point_each_sd().
    """
    beta = 0.005
    g = mt_obj.several_quad_function
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_test = (mt_obj.function_parameters_several_quad
                             (p, d, lambda_1, lambda_2))
    func_args = p, store_x0, matrix_test
    iterations_of_sd = np.random.uniform(0, 1, (iterations + 1, d))

    store_grad = np.zeros((iterations + 1, d))
    for j in range(iterations + 1):
        store_grad[j] = g(iterations_of_sd[j].reshape(d,), *func_args)

    partner_points_sd = mt_alg.partner_point_each_sd(iterations_of_sd, beta,
                                                     store_grad)
    assert(partner_points_sd.shape[0] == iterations_of_sd.shape[0])
    assert(partner_points_sd.shape[1] == iterations_of_sd.shape[1])
