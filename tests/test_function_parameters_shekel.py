import numpy as np
import pytest

from metod import objective_functions as mt_obj


def test_1():
    '''
    Asserts error message when lambda_1 is not integer
    '''
    lambda_1 = True
    lambda_2 = 10
    with pytest.raises(ValueError):
        mt_obj.function_parameters_shekel(lambda_1, lambda_2)


def test_2():
    '''
    Asserts error message when lambda_2 is not integer
    '''
    lambda_1 = 1
    lambda_2 = 'test'
    with pytest.raises(ValueError):
        mt_obj.function_parameters_shekel(lambda_1, lambda_2)


def test_3():
    '''
    Check functionality of function_parameters_shekel.py
    '''
    lambda_1 = 1
    lambda_2 = 4
    matrix_test, C, b = mt_obj.function_parameters_shekel(lambda_1, lambda_2)
    assert(np.all(C == np.array([[4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                                 [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
                                 [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                                 [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]])))
    assert(np.all(b == np.array([0.1, 0.2, 0.2, 0.4, 0.4,
                                 0.6, 0.3, 0.7, 0.5, 0.5])))
    assert(matrix_test.shape == (10, 4, 4))
