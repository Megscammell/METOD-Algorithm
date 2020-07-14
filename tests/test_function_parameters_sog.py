import numpy as np
import pytest

import metod.objective_functions as mt_obj


def test_function_parameters_sog():
    """ Testing functionality of slices used in function_parameters_sog
     and comparing results by using for loop.
     Have not used for loop in function_parameters_sog as less
     efficient.
    """
    p = 4
    d = 5
    store_A = np.zeros((p, d, d))
    store_x0 = np.zeros((p, d))
    store_c = np.zeros((p, ))
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
        store_c[i] = np.random.uniform(0.5, 1)
        store_rotation[i] = mt_obj.calculate_rotation_matrix(d, 3)
        store_x0[i] = np.random.uniform(0, 1, (d, ))
        matrix_test[i] = store_rotation[i].T @ store_A[i] @ store_rotation[i]
    np.random.seed(90)
    t_store_x0, t_matrix_test, t_store_c = (mt_obj.function_parameters_sog
                                            (p, d, 1, 10))
    assert(np.all(t_store_x0 == store_x0))
    assert(np.all(t_matrix_test == matrix_test))
    assert(np.all(t_store_c == store_c))


def test_1():
    '''
    Asserts error message when num_points is not integer
    '''
    d = 20
    p = 0.01
    lambda_1 = 1
    lambda_2 = 1
    with pytest.raises(ValueError):
        mt_obj.function_parameters_sog(p, d, lambda_1, lambda_2)


def test_2():
    '''
    Asserts error message when d is not integer
    '''
    d = 0.1
    p = 2
    lambda_1 = 1
    lambda_2 = 10
    with pytest.raises(ValueError):
        mt_obj.function_parameters_sog(p, d, lambda_1, lambda_2)


def test_3():
    '''
    Asserts error message when lambda_1 is not integer
    '''
    d = 10
    p = 2
    lambda_1 = True
    lambda_2 = 10
    with pytest.raises(ValueError):
        mt_obj.function_parameters_sog(p, d, lambda_1, lambda_2)


def test_4():
    '''
    Asserts error message when lambda_2 is not integer
    '''
    d = 10
    p = 2
    lambda_1 = 1
    lambda_2 = 'test'
    with pytest.raises(ValueError):
        mt_obj.function_parameters_sog(p, d, lambda_1, lambda_2)
