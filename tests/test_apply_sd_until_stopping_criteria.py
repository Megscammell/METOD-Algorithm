import numpy as np
from numpy import linalg as LA
import hypothesis
from hypothesis import assume, given, settings, strategies as st

import metod_testing as mtv3


@settings(max_examples=50, deadline=None)
@given(st.integers(5,100), st.integers(2,10))
def test_1(d, p):
    """Ensuring final iteration of steepest descent has norm of gradient smaller than tolerance.
    """
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_test = mtv3.function_parameters_quad(p, d, lambda_1,                                                            lambda_2)
    func_args = p, store_x0, matrix_test
    tolerance = 0.00001
    option = 'minimize'
    met = 'Nelder-Mead'
    initial_guess = 0.05
    f = mtv3.quad_function
    g = mtv3.quad_gradient
    projection = True
    point = np.random.uniform(0, 1, (d, ))
    initial_point = True
    sd_iterations, its, count_flag = mtv3.apply_sd_until_stopping_criteria(
                                     initial_point, point, d, projection, tolerance, option, met, initial_guess, func_args, f, g)
    
    assert(LA.norm(g(sd_iterations[its].reshape(d, ), *func_args)) < tolerance)

@settings(max_examples=50, deadline=None)
@given(st.integers(5,100), st.integers(2,10))
def test_2(d, p):
    """Ensuring shape of new iteration is (d,)
    """
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_test = mtv3.function_parameters_quad(p, d, lambda_1,                                                            lambda_2)
    func_args = p, store_x0, matrix_test
    option = 'minimize'
    met = 'Nelder-Mead'
    initial_guess = 0.05
    f = mtv3.quad_function
    g = mtv3.quad_gradient
    projection = True
    point = np.random.uniform(0, 1, (d, ))
    new_point, change_point = mtv3.sd_iteration(point, projection,                                       option, met, initial_guess,                                                func_args, f, g)
    assert(new_point.shape == (d,))

def test_3():
    """Ensuring that point is overwritten by x_iteration
    """
    point = np.array([1,2,3,4,5])
    c = 0
    while c < 5:
        x_iteration = np.arange(c, c + 5)
        point = x_iteration
        c += 1
    

    assert(np.all(point == np.array([4, 5, 6, 7, 8])))

def updating_array(d, arr):
    new_p = np.array([1, 2, 3, 4, 5])
    arr = np.vstack([arr, new_p.reshape((1, d))])
    return arr

def test_4():
    """Ensuring that array gets updated in function and new values are given to sd_iterations
    """    
    d = 5
    arr = np.zeros((1, d))
    p = np.random.uniform(0, 1, (d))
    arr[0] = p.reshape(1, d)
    arr = updating_array(d, arr)
    assert(arr.shape[0] == 2)
    assert(np.all(arr[0] == p))
    assert(np.all(arr[1] == np.array([1, 2, 3, 4, 5])))