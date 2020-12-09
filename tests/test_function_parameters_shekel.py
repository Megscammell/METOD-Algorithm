import numpy as np
import pytest

from metod import objective_functions as mt_obj


def test_function_parameters_shekel():
    """
     Testing functionality of slices used in function_parameters_shekel
     and comparing results by using for loop.
     Have not used for loop in function_parameters_shekel as less
     efficient.
    """
    p = 4
    d = 5
    b_val = 1
    store_A = np.zeros((p, d, d))
    C = np.zeros((d, p))
    store_b = np.zeros((p, ))
    store_rotation = np.zeros((p, d, d))
    matrix_test = np.zeros((p, d, d))
    np.random.seed(90)
    for i in range(p):
        C[:, i] = np.random.uniform(0, 1,(d)) 
        store_rotation[i] = mt_obj.calculate_rotation_matrix(d, 3)
        store_b[i] = b_val
        diag_vals = np.zeros(d, )
        a = 1
        diag_vals[0] = a
        b = 10
        diag_vals[1] = b
        for j in range(2, d):
            diag_vals[j] = np.random.uniform(1.1, 9.9)
        store_A[i] = np.diag(diag_vals)
        matrix_test[i] = store_rotation[i].T @ store_A[i] @ store_rotation[i]
    np.random.seed(90)
    t_matrix_test, t_C, t_store_b = (mt_obj.function_parameters_shekel(p, d,
                                                                       b_val,
                                                                       1, 10))
    assert(np.all(t_C == C))
    assert(np.all(t_matrix_test == matrix_test))
    assert(np.all(t_store_b == store_b))


def test_1():
    '''
    Asserts error message when num_points is not integer
    '''
    d = 20
    p = 0.01
    lambda_1 = 1
    lambda_2 = 1
    b_val = 1
    with pytest.raises(ValueError):
        mt_obj.function_parameters_shekel(p, d, b_val, lambda_1, lambda_2)


def test_2():
    '''
    Asserts error message when d is not integer
    '''
    d = 0.1
    p = 2
    lambda_1 = 1
    lambda_2 = 10
    b_val = 1
    with pytest.raises(ValueError):
        mt_obj.function_parameters_shekel(p, d, b_val, lambda_1, lambda_2)


def test_3():
    '''
    Asserts error message when lambda_1 is not integer
    '''
    d = 10
    p = 2
    lambda_1 = True
    lambda_2 = 10
    b_val = 1
    with pytest.raises(ValueError):
        mt_obj.function_parameters_shekel(p, d, b_val, lambda_1, lambda_2)


def test_4():
    '''
    Asserts error message when lambda_2 is not integer
    '''
    d = 10
    p = 2
    lambda_1 = 1
    lambda_2 = 'test'
    b_val = 1
    with pytest.raises(ValueError):
        mt_obj.function_parameters_shekel(p, d, b_val, lambda_1, lambda_2)


def test_5():
    '''
    Asserts error message when b_val is not integer or float
    '''
    d = 10
    p = 2
    lambda_1 = 1
    lambda_2 = 10
    b_val = True
    with pytest.raises(ValueError):
        mt_obj.function_parameters_shekel(p, d, b_val, lambda_1, lambda_2)