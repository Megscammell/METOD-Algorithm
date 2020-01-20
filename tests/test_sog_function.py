import numpy as np

import metod_testing as mtv3


def test_1():
    """Test sog_function for d = 2 by using coding for loop differently. 
    """
    d = 2
    p = 3
    sigma_sq = 0.05
    store_c = np.zeros((p))
    store_rotation = np.zeros((p, d, d))
    store_A = np.zeros((p, d, d))
    store_x0 =  np.zeros((p, d))

    store_c[0] = 0.5
    store_c[1] = 0.6
    store_c[2] = 0.7

    store_rotation[0] = np.array([[0.6, -0.3], [0.3, 0.6]])
    store_rotation[1] = np.array([[0.4, -0.2], [0.2, 0.4]])
    store_rotation[2] = np.array([[0.8, -0.6], [0.6, 0.8]])

    store_A[0] = np.array([[1, 0], [0, 10]])
    store_A[1] = np.array([[1, 0], [0, 2]])
    store_A[2] = np.array([[1, 0], [0, 3]])

    store_x0[0] = np.array([0.2, 0.3]).reshape(d,)
    store_x0[1] = np.array([0.8, 0.9]).reshape(d,)
    store_x0[2] = np.array([0.5, 0.5]).reshape(d,)

    x = np.array([0.6, 0.6]).reshape(d,)

    matrix = np.transpose(store_rotation, (0, 2, 1)) @ store_A @ store_rotation

    cumulative_function = 0
    for i in range(p):
        c = store_c[i]
        matrix_test = matrix[i]
        x0 = store_x0[i]
        calculation = (-1 / (2 * sigma_sq)) * (x - x0).T @ matrix_test @ (x - x0)
        individual_function = c * np.exp(calculation)
        cumulative_function += individual_function 

    final_function_val = float(-cumulative_function)
    matrix_all = np.transpose(store_rotation, (0, 2, 1)) @ store_A @ store_rotation
    func_args = p, sigma_sq, store_x0, matrix_all, store_c
    test_function_val = mtv3.sog_function(x, *func_args)
    assert(final_function_val == test_function_val)



def test_2():
    """ Computational test where the function value has been calculated for d = 2 and p = 3. 
    """
    d = 2
    p = 3
    sigma_sq = 0.05
    store_c = np.zeros((p))
    store_rotation = np.zeros((p, d, d))
    store_A = np.zeros((p, d, d))
    store_x0 =  np.zeros((p, d))

    store_c[0] = 0.5
    store_c[1] = 0.6
    store_c[2] = 0.7

    store_rotation[0] = np.array([[0.6, -0.3], [0.3, 0.6]])
    store_rotation[1] = np.array([[0.4, -0.2], [0.2, 0.4]])
    store_rotation[2] = np.array([[0.8, -0.6], [0.6, 0.8]])

    store_A[0] = np.array([[1, 0], [0, 10]])
    store_A[1] = np.array([[1, 0], [0, 2]])
    store_A[2] = np.array([[1, 0], [0, 3]])

    store_x0[0] = np.array([0.2, 0.3]).reshape(d,)
    store_x0[1] = np.array([0.8, 0.9]).reshape(d,)
    store_x0[2] = np.array([0.5, 0.5]).reshape(d,)

    x = np.array([0.6, 0.6]).reshape(d,)
    matrix_all = np.transpose(store_rotation, (0,2,1)) @ store_A @                          store_rotation
    func_args = p, sigma_sq, store_x0, matrix_all, store_c
    function_val = mtv3.sog_function(x, *func_args)
    vals = -(0.5 * np.exp(-9.225) + 0.6 * np.exp(-0.516)+ 0.7 * np.exp(-0.592))
    assert(function_val == vals)



