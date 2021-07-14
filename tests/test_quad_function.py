import numpy as np
from hypothesis import assume, given, settings, strategies as st

from metod_alg import check_metod_class as prev_mt_alg
from metod_alg import objective_functions as mt_obj


def inefficient_quad_func(x, p, store_x0, matrix_test):
    """
    For loop version of prev_mt_alg.quad_function()
    """
    store_func_values = np.zeros((p))
    for i in range(p):
        store_func_values[i] = ((x - store_x0[i]).T @ matrix_test[i] @
                                (x - store_x0[i]))
    return np.min(store_func_values)


def inefficient_quad_grad(x, p, store_x0, matrix_test):
    """
    For loop version of prev_mt_alg.quad_gradient()
    """
    store_func_values = np.zeros((p))
    for i in range(p):
        store_func_values[i] = ((x - store_x0[i]).T @ matrix_test[i] @
                                (x - store_x0[i]))
    pos = np.argmin(store_func_values)
    gradient = 2 * (matrix_test[pos] @
                   (x - store_x0[pos]))
    return gradient


def test_1():
    """
    Computational test for prev_mt_alg.quad_function()
    and prev_mt_alg.calc_minimizer_quad(),
    where the minimum value is 5 and the position
    in which the minimum value is obtained is 1.
    """
    p = 2
    d = 2
    matrix_test = np.zeros((p, d, d))
    store_x0 = np.zeros((p, d))
    matrix_test[0] = np.array([[2, 4], [4, 1]])
    matrix_test[1] = np.array([[2, 4], [4, 1]])
    x = np.array([2, 1])
    store_x0[0] = np.array([3, 4])
    store_x0[1] = np.array([1, 0])
    function_parameters = p, store_x0, matrix_test
    value_of_minimum = prev_mt_alg.quad_function(x, *function_parameters)
    assert(value_of_minimum == 11)

    min_x = np.array([3.05, 3.9])
    pos_min = prev_mt_alg.calc_minimizer_quad(min_x, *function_parameters)
    assert(pos_min == 0)

    min_x = np.array([1.1, 0.1])
    pos_min = prev_mt_alg.calc_minimizer_quad(min_x, *function_parameters)
    assert(pos_min == 1)


def test_2():
    """
    Computational test for prev_mt_alg.quad_function() with
    d = 5 and p = 5.
    """
    p = 5
    store_x0 = np.array([[0.94963972,
                        0.08488167,
                        0.9188352,
                        0.12158757,
                        0.45955333],
                        [0.34542031,
                        0.7406909,
                        0.21995994,
                        0.47311855,
                        0.09100008],
                        [0.83191958,
                        0.16843706,
                        0.06743084,
                        0.82618932,
                        0.00319565],
                        [0.12135494,
                        0.55849957,
                        0.863359,
                        0.14405089,
                        0.89450105],
                        [0.25692001,
                        0.42380292,
                        0.61662887,
                        0.98868442,
                        0.39848917]])
    matrix_test = np.array([[[1.28094247, 0, 0, 0.70142069, 0],
                           [0, 5.53799473, -0.5628502, 0, 0],
                           [0, -0.5628502,  9.92900045, 0, 0],
                           [0.70142069, 0, 0, 2.75121616, 0],
                           [0, 0, 0, 0, 8.65595723]],
                           [[3.63229643, 0, 0, 0.22780732,
                            -0.25117144],
                            [0, 10, 0, 0, 0],
                            [0, 0, 8.12972135, 0, 0],
                            [0.22780732, 0, 0, 2.75296753, 1.32058744],
                            [-0.25117144, 0, 0, 1.32058744, 2.06353554]],
                           [[1, 0, 0, 0, 0],
                            [0, 5.89407958, -0.31840914, 0, -0.87443975],
                            [0, -0.31840914,  4.29748954, -0.34312367,
                             1.99964824],
                            [0, 0, -0.34312367,  5.13870628, 0.12494138],
                            [0, -0.87443975,  1.99964824,  0.12494138,
                             9.06094741]],
                           [[1, 0, 0, 0, 0],
                            [0, 8.53757755, -0.03465536, 0.18342492,
                            0.70358008],
                            [0, -0.03465536, 6.8462396, -0.03356387,
                             -0.12874409],
                            [0, 0.18342492, -0.03356387, 7.05875932,
                             0.67067621],
                            [0, 0.70358008, -0.12874409, 0.67067621,
                             9.4564879]],
                           [[1, 0, 0, 0, 0],
                            [0,  6.07244368, 0,  0.22222738, 2.02860655],
                            [0, 0, 6.8643249, 0, 0],
                            [0,  0.22222738, 0,  2.28175593, 0.72934983],
                            [0,  2.02860655, 0,  0.72934983, 8.85974054]]])
    function_parameters = p, store_x0, matrix_test
    x = np.array([0.65189308,
                 0.65894732,
                 0.33044144,
                 0.23970461,
                 0.57332117])
    value_of_minimum = (prev_mt_alg.quad_function
                        (x, *function_parameters))
    assert(np.round(value_of_minimum, 5) == 0.73306)


@settings(max_examples=10, deadline=None)
@given(st.integers(2, 50), st.integers(2, 100))
def test_3(p, d):
    """
    Checks results against different version of the quadratic function
    using for loop.
    """
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
        store_x0[i] = np.random.uniform(0, 1, (d,))
        store_rotation[i] = mt_obj.calculate_rotation_matrix(d, 3)
    matrix_test = (np.transpose(store_rotation, (0, 2, 1)) @ store_A @
                   store_rotation)
    func_args = (p, store_x0, matrix_test)
    x = np.random.uniform(0, 1, (d,))
    func_val_1 = inefficient_quad_func(x, *func_args)
    func_val_2 = (prev_mt_alg.quad_function
                  (x, *func_args))
    assert(np.round(func_val_1, 5) == np.round(func_val_2, 5))



def test_4():
    """
    Computational example for prev_mt_alg.quad_gradient()
    where the gradient is [6, 5]^T.
    """
    p = 2
    d = 2
    matrix_test = np.zeros((p, d, d))
    store_x0 = np.zeros((p, d))
    matrix_test[0] = np.array([[2, 4], [4, 1]])
    matrix_test[1] = np.array([[2, 4], [4, 1]])
    x = np.array([2, 1])
    store_x0[0] = np.array([3, 4])
    store_x0[1] = np.array([1, 0])
    function_parameters = p, store_x0, matrix_test
    grad_func = prev_mt_alg.quad_gradient(x, *function_parameters)
    assert(np.all(grad_func == np.array([12, 10])))


def test_5():
    """
    Computational example for prev_mt_alg.quad_gradient() with
    d = 5 and p = 5
    """
    p = 5
    store_x0 = np.array([[0.94963972,
                        0.08488167,
                        0.9188352,
                        0.12158757,
                        0.45955333],
                        [0.34542031,
                        0.7406909,
                        0.21995994,
                        0.47311855,
                        0.09100008],
                        [0.83191958,
                        0.16843706,
                        0.06743084,
                        0.82618932,
                        0.00319565],
                        [0.12135494,
                        0.55849957,
                        0.863359,
                        0.14405089,
                        0.89450105],
                        [0.25692001,
                        0.42380292,
                        0.61662887,
                        0.98868442,
                        0.39848917]])
    matrix_test = np.array([[[1.28094247, 0, 0, 0.70142069, 0],
                           [0, 5.53799473, -0.5628502, 0, 0],
                           [0, -0.5628502,  9.92900045, 0, 0],
                           [0.70142069, 0, 0, 2.75121616, 0],
                           [0, 0, 0, 0, 8.65595723]],
                           [[3.63229643, 0, 0, 0.22780732,
                            -0.25117144],
                            [0, 10, 0, 0, 0],
                            [0, 0, 8.12972135, 0, 0],
                            [0.22780732, 0, 0, 2.75296753, 1.32058744],
                            [-0.25117144, 0, 0, 1.32058744, 2.06353554]],
                           [[1, 0, 0, 0, 0],
                            [0, 5.89407958, -0.31840914, 0, -0.87443975],
                            [0, -0.31840914,  4.29748954, -0.34312367,
                             1.99964824],
                            [0, 0, -0.34312367,  5.13870628, 0.12494138],
                            [0, -0.87443975,  1.99964824,  0.12494138,
                             9.06094741]],
                           [[1, 0, 0, 0, 0],
                            [0, 8.53757755, -0.03465536, 0.18342492,
                            0.70358008],
                            [0, -0.03465536, 6.8462396, -0.03356387,
                             -0.12874409],
                            [0, 0.18342492, -0.03356387, 7.05875932,
                             0.67067621],
                            [0, 0.70358008, -0.12874409, 0.67067621,
                             9.4564879]],
                           [[1, 0, 0, 0, 0],
                            [0,  6.07244368, 0,  0.22222738, 2.02860655],
                            [0, 0, 6.8643249, 0, 0],
                            [0,  0.22222738, 0,  2.28175593, 0.72934983],
                            [0,  2.02860655, 0,  0.72934983, 8.85974054]]])
    function_parameters = p, store_x0, matrix_test
    x = np.array([0.65189308,
                 0.65894732,
                 0.33044144,
                 0.23970461,
                 0.57332117])
    grad_func = prev_mt_alg.quad_gradient(x, *function_parameters)
    assert(np.round(grad_func[0], 4) == np.round(1.87776, 4))
    assert(np.round(grad_func[1], 4) == np.round(-1.63487, 4))
    assert(np.round(grad_func[2], 4) == np.round(1.79637, 4))
    assert(np.round(grad_func[3], 4) == np.round(0.12837, 4))
    assert(np.round(grad_func[4], 4) == np.round(1.22013, 4))


@settings(max_examples=10, deadline=None)
@given(st.integers(2, 50), st.integers(2, 100))
def test_6(p, d):
    """
    Checks results against different version of the quadratic gradient
    using for loop.
    """
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
        store_x0[i] = np.random.uniform(0, 1, (d,))
        store_rotation[i] = mt_obj.calculate_rotation_matrix(d, 3)
    matrix_test = (np.transpose(store_rotation, (0, 2, 1)) @ store_A @
                   store_rotation)
    func_args = (p, store_x0, matrix_test)
    x = np.random.uniform(0, 1, (d,))
    grad_val_1 = inefficient_quad_grad(x, *func_args)
    grad_val_2 = (prev_mt_alg.quad_gradient
                  (x, *func_args))
    assert(np.all(np.round(grad_val_1, 5) == np.round(grad_val_2, 5)))