import numpy as np
import pytest

from metod import objective_functions as mt_obj


def test_matrix_test():
    """ Checking np.transpose works for array of shape (p, d, d)
    """
    rotation = np.zeros((2, 3, 3))
    rotation[0] = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])
    rotation[1] = np.array([[4, 3, 1],
                           [6, 7, 2],
                           [5, 2, 6]])
    rotation_transpose = np.transpose(rotation, (0, 2, 1))
    assert(np.all(rotation_transpose[0] == np.array([[1, 4, 7],
                                                    [2, 5, 8],
                                                    [3, 6, 9]])))
    assert(np.all(rotation_transpose[1] == np.array([[4, 6, 5],
                                                    [3, 7, 2],
                                                    [1, 2, 6]])))


def test_create_function():
    """ Testing functionality of slices used in create_function and
    comparing results by using for loop.
    Have not used for loop in create_function as less efficient.
    """
    p = 4
    d = 5
    store_A = np.zeros((p, d, d))
    store_x0 = np.zeros((p, d))
    store_rotation = np.zeros((p, d, d))
    matrix_test = np.zeros((p, d, d))
    np.random.seed(90)
    for i in range(p):
        diag_vals = np.zeros(d, )
        a = 1
        diag_vals[0] = a
        b = 10
        diag_vals[1] = b
        for j in range(2, d):
            diag_vals[j] = np.random.uniform(1.1, 9.9)
        store_A[i] = np.diag(diag_vals)
        store_rotation[i] = mt_obj.calculate_rotation_matrix(d, 3)
        x0 = np.random.uniform(0, 1, (d, ))
        store_x0[i] = x0
        matrix_test[i] = store_rotation[i].T @ store_A[i] @ store_rotation[i]
    np.random.seed(90)
    store_x0_function, matrix_test_function = (mt_obj.function_parameters_quad
                                               (p, d, 1, 10))
    assert(np.all(store_x0_function == store_x0))
    assert(np.all(matrix_test_function == matrix_test))


def test_1():
    '''
    Asserts error message when num_points is not integer
    '''
    d = 20
    p = 0.01
    lambda_1 = 1
    lambda_2 = 1
    with pytest.raises(ValueError):
        mt_obj.function_parameters_quad(p, d, lambda_1, lambda_2)


def test_2():
    '''
    Asserts error message when d is not integer
    '''
    d = 0.1
    p = 2
    lambda_1 = 1
    lambda_2 = 10
    with pytest.raises(ValueError):
        mt_obj.function_parameters_quad(p, d, lambda_1, lambda_2)


def test_3():
    '''
    Asserts error message when lambda_1 is not integer
    '''
    d = 10
    p = 2
    lambda_1 = True
    lambda_2 = 10
    with pytest.raises(ValueError):
        mt_obj.function_parameters_quad(p, d, lambda_1, lambda_2)


def test_4():
    '''
    Asserts error message when lambda_2 is not integer
    '''
    d = 10
    p = 2
    lambda_1 = 1
    lambda_2 = 'test'
    with pytest.raises(ValueError):
        mt_obj.function_parameters_quad(p, d, lambda_1, lambda_2)
