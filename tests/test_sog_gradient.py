import numpy as np
from hypothesis import given, settings, strategies as st

from metod import objective_functions as mt_obj


def test_1():
    """Test sog_gradient for d = 2 by coding for loop differently."""
    d = 2
    p = 3
    sigma_sq = 0.05

    store_c = np.zeros((p, ))
    store_rotation = np.zeros((p, d, d))
    store_A = np.zeros((p, d, d))
    store_x0 = np.zeros((p, d))

    store_c[0] = 0.5
    store_c[1] = 0.6
    store_c[2] = 0.7

    store_rotation[0] = np.array([[0.6, -0.3], [0.3, 0.6]])
    store_rotation[1] = np.array([[0.4, -0.2], [0.2, 0.4]])
    store_rotation[2] = np.array([[0.8, -0.6], [0.6, 0.8]])

    store_A[0] = np.array([[1, 0], [0, 10]])
    store_A[1] = np.array([[1, 0], [0, 2]])
    store_A[2] = np.array([[1, 0], [0, 3]])

    store_x0[0] = np.array([0.2, 0.3]).reshape(d, )
    store_x0[1] = np.array([0.8, 0.9]).reshape(d, )
    store_x0[2] = np.array([0.5, 0.5]).reshape(d, )

    x = np.array([0.6, 0.6]).reshape(d, )
    cumulative_gradient = 0
    for i in range(p):
        A = store_A[i]
        x0 = store_x0[i]
        rotation = store_rotation[i]
        c = store_c[i]
        matrix = rotation.T @ A @ rotation
        func_value = float((c / sigma_sq) * np.exp((-1 / (2 * sigma_sq)) *
                           ((((x - x0).T) @ matrix) @ (x - x0))))
        individual_gradient = func_value * (matrix @ (x - x0))
        cumulative_gradient += individual_gradient

    matrix_all = (np.transpose(store_rotation, (0, 2, 1)) @ store_A @
                  store_rotation)
    func_args = p, sigma_sq, store_x0, matrix_all, store_c
    gradient_test = mt_obj.sog_gradient(x, *func_args)

    assert(np.round(gradient_test[0], 5) == np.round(cumulative_gradient[0], 5)
           )
    assert(np.round(gradient_test[1], 5) == np.round(cumulative_gradient[1], 5)
           )


def test_2_f():
    """Computational example"""
    d = 2
    p = 3
    sigma_sq = 0.05
    store_c = np.zeros((p, ))
    store_rotation = np.zeros((p, d, d))
    store_A = np.zeros((p, d, d))
    store_x0 = np.zeros((p, d))

    store_c[0] = 0.5
    store_c[1] = 0.6
    store_c[2] = 0.7

    store_rotation[0] = np.array([[0.6, -0.3], [0.3, 0.6]])
    store_rotation[1] = np.array([[0.4, -0.2], [0.2, 0.4]])
    store_rotation[2] = np.array([[0.8, -0.6], [0.6, 0.8]])

    store_A[0] = np.array([[1, 0], [0, 10]])
    store_A[1] = np.array([[1, 0], [0, 2]])
    store_A[2] = np.array([[1, 0], [0, 3]])

    store_x0[0] = np.array([0.2, 0.3]).reshape(d, )
    store_x0[1] = np.array([0.8, 0.9]).reshape(d, )
    store_x0[2] = np.array([0.5, 0.5]).reshape(d, )

    x = np.array([0.6, 0.6]).reshape(d, )
    matrix_all = (np.transpose(store_rotation, (0, 2, 1)) @ store_A @
                  store_rotation)
    func_args = p, sigma_sq, store_x0, matrix_all, store_c
    gradient_test = mt_obj.sog_gradient(x, *func_args)
    el_1 = ((10 * np.exp(-9.225) * 0.99) + (12 * np.exp(-0.516) * -0.072) +
            (14 * np.exp(-0.592) * 0.268))
    el_2 = ((10 * np.exp(-9.225) * 1.755) + (12 * np.exp(-0.516) * -0.124) +
            (14 * np.exp(-0.592) * 0.324))
    assert(np.round(gradient_test[0], 5) == np.round(el_1, 5))
    assert(np.round(gradient_test[1], 5) == np.round(el_2, 5))


@settings(max_examples=50, deadline=None)
@given(st.integers(2, 10), st.integers(5, 100))
def test_3(p, d):
    """Testing size outputs for gradient
    """
    sigma_sq = 0.05
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_test, store_c = mt_obj.function_parameters_sog(p, d,
                                                                    lambda_1,
                                                                    lambda_2)
    x = np.random.uniform(0, 1, (d, ))
    gradient = mt_obj.sog_gradient(x, p, sigma_sq, store_x0, matrix_test,
                                   store_c)
    assert(gradient.shape[0] == d)
