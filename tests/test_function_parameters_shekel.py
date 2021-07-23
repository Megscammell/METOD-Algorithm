import numpy as np
import pytest

from metod_alg import objective_functions as mt_obj


def test_1():
    """
    Asserts error message when lambda_1 is not integer
    """
    lambda_1 = True
    lambda_2 = 10
    p = 5
    with pytest.raises(ValueError):
        mt_obj.function_parameters_shekel(lambda_1, lambda_2, p)


def test_2():
    """
    Asserts error message when lambda_2 is not integer
    """
    lambda_1 = 1
    lambda_2 = 'test'
    p = 5
    with pytest.raises(ValueError):
        mt_obj.function_parameters_shekel(lambda_1, lambda_2, p)


def test_3():
    """
    Asserts error message when lambda_2 is not integer
    """
    lambda_1 = 1
    lambda_2 = 10
    p = 'test'
    with pytest.raises(ValueError):
        mt_obj.function_parameters_shekel(lambda_1, lambda_2, p)


def test_4():
    """
    Check functionality of function_parameters_shekel.py
    """
    lambda_1 = 1
    lambda_2 = 4
    p = 5
    matrix_test, C, b = mt_obj.function_parameters_shekel(lambda_1, lambda_2,
                                                          p)
    assert(np.all(C == np.array([[4, 1, 8, 6, 3],
                                 [4, 1, 8, 6, 7],
                                 [4, 1, 8, 6, 3],
                                 [4, 1, 8, 6, 7]])))
    assert(np.all(b == np.array([1.25, 1.45, 1.45, 1.65, 1.7])))
    assert(matrix_test.shape == (p, 4, 4))


def test_5():
    """
    Check functionality of function_parameters_shekel.py
    """
    lambda_1 = 1
    lambda_2 = 4
    p = 7
    matrix_test, C, b = mt_obj.function_parameters_shekel(lambda_1, lambda_2,
                                                          p)
    assert(np.all(C == np.array([[4, 1, 8, 6, 3, 2, 5],
                                 [4, 1, 8, 6, 7, 9, 3],
                                 [4, 1, 8, 6, 3, 2, 5],
                                 [4, 1, 8, 6, 7, 9, 3]])))
    assert(np.all(b == np.array([1.25, 1.45, 1.45, 1.65, 1.7,
                                 1.8, 1.75])))
    assert(matrix_test.shape == (p, 4, 4))


def test_6():
    """
    Check functionality of function_parameters_shekel.py
    """
    lambda_1 = 1
    lambda_2 = 4
    p = 10
    matrix_test, C, b = mt_obj.function_parameters_shekel(lambda_1, lambda_2,
                                                          p)
    assert(np.all(C == np.array([[4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                                 [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
                                 [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                                 [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]])))
    assert(np.all(b == np.array([1.25, 1.45, 1.45, 1.65, 1.7,
                                 1.8, 1.75, 1.9, 1.7, 1.8])))
    assert(matrix_test.shape == (p, 4, 4))
