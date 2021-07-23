import numpy as np

from metod_alg import objective_functions as mt_obj


def inefficient_sog_func(point, p, sigma_sq, store_x0, matrix_test, store_c):
    """
    Compute Sum of Gaussians function at a given point with given arguments.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the function.
    p : integer
        Number of local minima.
    sigma_sq: float or integer
              Value of sigma squared.
    store_x0 : 2-D array with shape (p, d).
    matrix_test : 3-D array with shape (p, d, d).
    store_c : 1-D array with shape (p, ).

    Returns
    -------
    float(-function_val) : float
                           Function value.
    """
    function_val = 0
    for i in range(p):
        function_val += store_c[i] * np.exp((- 1 / (2 * sigma_sq)) *
                                            (np.transpose(point -
                                             store_x0[i])) @ matrix_test[i] @
                                            (point - store_x0[i]))
    return float(-function_val)


def test_1():
    """Check mt_obj.sog_function for d = 2."""
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

    x = np.array([0.6, 0.6]).reshape(d,)
    matrix = np.transpose(store_rotation, (0, 2, 1)) @ store_A @ store_rotation
    cumulative_function = 0

    for i in range(p):
        c = store_c[i]
        matrix_test = matrix[i]
        x0 = store_x0[i]
        calculation = ((-1 / (2 * sigma_sq)) * (x - x0).T @ matrix_test @ (x -
                       x0))
        individual_function = c * np.exp(calculation)
        cumulative_function += individual_function

    final_function_val = float(-cumulative_function)
    matrix_all = (np.transpose(store_rotation, (0, 2, 1)) @ store_A @
                  store_rotation)
    func_args = p, sigma_sq, store_x0, matrix_all, store_c
    test_function_val = mt_obj.sog_function(x, *func_args)
    assert(final_function_val == test_function_val)


def test_2():
    """Computational example for mt_obj.sog_function()"""
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
    function_val = mt_obj.sog_function(x, *func_args)
    vals = (-(0.5 * np.exp(-9.225) + 0.6 * np.exp(-0.516) + 0.7 *
            np.exp(-0.592)))
    assert(np.round(function_val, 10) == np.round(vals, 10))


def test_3():
    """
    Compares results of inefficient_sog_func() and mt_obj.sog_function().
    Results should be the same.
    """
    p = 10
    d = 20
    sigma_sq = 0.8
    lambda_1 = 1
    lambda_2 = 10
    store_A = np.zeros((p, d, d))
    store_x0 = np.zeros((p, d))
    store_rotation = np.zeros((p, d, d))
    store_c = np.zeros((p))
    for i in range(p):
        diag_vals = np.zeros(d)
        diag_vals[:2] = np.array([lambda_1, lambda_2])
        diag_vals[2:] = np.random.uniform(lambda_1 + 1,
                                          lambda_2 - 1, (d - 2))
        store_A[i] = np.diag(diag_vals)
        store_x0[i] = np.random.uniform(0, 1, (d))
        store_c[i] = np.random.uniform(0.5, 1)
        store_rotation[i] = mt_obj.calculate_rotation_matrix(d, 3)
    matrix_test = (np.transpose(store_rotation, (0, 2, 1)) @ store_A @
                   store_rotation)
    func_args = (p, sigma_sq, store_x0, matrix_test, store_c)

    x = np.random.uniform(0, 1, (d,))
    func_val_1 = inefficient_sog_func(x, *func_args)
    func_val_2 = (mt_obj.sog_function
                  (x, *func_args))
    assert(np.round(func_val_1, 5) == np.round(func_val_2, 5))
