import numpy as np
from hypothesis import assume, given, settings, strategies as st

from metod_alg import objective_functions as mt_obj
from metod_alg import metod_algorithm_functions as mt_alg
from metod_alg import check_metod_class as prev_mt_alg


def calc_minimizer_sev_quad(point, p, store_x0, matrix_test):
    """
    Finding the position of the local minimizer which point is closest
    to, using the minimum of several Quadratic forms function.

    Parameters
    ----------
    point : 1-D array with shape (d, )
            A point used to evaluate the function.
    p : integer
        Number of local minima.
    store_x0 : 2-D arrays with shape (p, d).
    matrix_test : 3-D arrays with shape (p, d, d).

    Returns
    -------
    position_minimum : integer
                       Position of the local minimizer which produces the
                       smallest distance between point and all p local
                       minimizers.
    """
    store_func_values = np.zeros((p))
    for i in range(p):
        store_func_values[i] = 0.5 * (np.transpose(point - store_x0[i]) @
                                      matrix_test[i] @ (point - store_x0[i]))
    position_minimum = np.argmin(store_func_values)
    norm_with_minimizer = np.linalg.norm(point - store_x0[position_minimum])
    return position_minimum


@settings(max_examples=10, deadline=None)
@given(st.integers(2, 20), st.integers(10, 50))
def test_1(p, d):
    """
    Check whether a local minimizer has already been identified by the METOD
    algorithm by applying prev_mt_alg.check_des_points().
    The position of the identified local minimizer is c = 0.
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
        store_x0[i] = np.random.uniform(0, 1, (d))
        store_rotation[i] = mt_obj.calculate_rotation_matrix(d, 3)
    matrix_test = (np.transpose(store_rotation, (0, 2, 1)) @ store_A @
                store_rotation)
    func_args = (p, store_x0, matrix_test)

    x = np.random.uniform(0,1,(d,))
    projection = False
    tolerance = 0.001
    option = 'minimize_scalar'
    met = 'brent'
    initial_guess = 0.005
    const = 0.1
    bound_1, bound_2 = 0,1 
    usage = 'metod_algorithm'
    relax_sd_it = 1
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    check_func = mt_obj.calc_minimizer_sev_quad
    (iterations_of_sd,
    its,
    store_grad_part) = (mt_alg.apply_sd_until_stopping_criteria
                        (x, d, projection, tolerance,
                        option, met, initial_guess,
                        func_args, f, g, bound_1, bound_2,
                        usage, relax_sd_it, init_grad=None))
    
    assert(calc_minimizer_sev_quad(x, p, store_x0, matrix_test) ==
           check_func(iterations_of_sd[-1], p,
                      store_x0, matrix_test))
    pos = calc_minimizer_sev_quad(iterations_of_sd[-1], p,
                                  store_x0, matrix_test)                             
    discovered_minimizers = [store_x0[pos]]
    for j in range(int(p/2)):
        if j != pos:
            discovered_minimizers.append(j)

    c = prev_mt_alg.check_des_points(iterations_of_sd, discovered_minimizers, const)
    assert(c == 0)
    

@settings(max_examples=10, deadline=None)
@given(st.integers(2, 20), st.integers(10, 50))
def test_2(p, d):
    """
    Check whether a local minimizer has already been identified by the METOD
    algorithm by applying prev_mt_alg.check_des_points(). The position of the identified
    local minimizer is c = len(discovered_minimizers)-1.
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
        store_x0[i] = np.random.uniform(0, 1, (d))
        store_rotation[i] = mt_obj.calculate_rotation_matrix(d, 3)
    matrix_test = (np.transpose(store_rotation, (0, 2, 1)) @ store_A @
                   store_rotation)
    func_args = (p, store_x0, matrix_test)

    x = np.random.uniform(0,1,(d,))
    projection = False
    tolerance = 0.001
    option = 'minimize_scalar'
    met = 'brent'
    initial_guess = 0.005
    const = 0.1
    bound_1, bound_2 = 0,1 
    usage = 'metod_algorithm'
    relax_sd_it = 1
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    check_func = mt_obj.calc_minimizer_sev_quad
    (iterations_of_sd,
    its,
    store_grad_part) = (mt_alg.apply_sd_until_stopping_criteria
                        (x, d, projection, tolerance,
                        option, met, initial_guess,
                        func_args, f, g, bound_1, bound_2,
                        usage, relax_sd_it, init_grad=None))
    
    assert(calc_minimizer_sev_quad(x, p, store_x0, matrix_test) ==
           check_func(iterations_of_sd[-1], p,
                      store_x0, matrix_test))
    pos = calc_minimizer_sev_quad(iterations_of_sd[-1], p,
                                  store_x0, matrix_test)                             
    discovered_minimizers = []
    for j in range(int(p/2)):
        if j != pos:
            discovered_minimizers.append(j)
    discovered_minimizers.append(store_x0[pos])
    c = prev_mt_alg.check_des_points(iterations_of_sd, discovered_minimizers, const)
    assert(c == len(discovered_minimizers)-1)


@settings(max_examples=10, deadline=None)
@given(st.integers(2, 20), st.integers(10, 50))
def test_3(p, d):
    """
    Check whether a local minimizerhas already been identified by the METOD
    algorithm by applying prev_mt_alg.check_des_points().
    The position of the identified local minimizer is c = None.
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
        store_x0[i] = np.random.uniform(0, 1, (d))
        store_rotation[i] = mt_obj.calculate_rotation_matrix(d, 3)
    matrix_test = (np.transpose(store_rotation, (0, 2, 1)) @ store_A @
                store_rotation)
    func_args = (p, store_x0, matrix_test)

    x = np.random.uniform(0,1,(d,))
    projection = False
    tolerance = 0.001
    option = 'minimize_scalar'
    met = 'brent'
    initial_guess = 0.005
    const = 0.1
    bound_1, bound_2 = 0,1 
    usage = 'metod_algorithm'
    relax_sd_it = 1
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    check_func = mt_obj.calc_minimizer_sev_quad
    (iterations_of_sd,
    its,
    store_grad_part) = (mt_alg.apply_sd_until_stopping_criteria
                        (x, d, projection, tolerance,
                        option, met, initial_guess,
                        func_args, f, g, bound_1, bound_2,
                        usage, relax_sd_it, init_grad=None))
    
    assert(calc_minimizer_sev_quad(x, p, store_x0, matrix_test) ==
           check_func(iterations_of_sd[-1], p,
                      store_x0, matrix_test))
    pos = calc_minimizer_sev_quad(iterations_of_sd[-1], p,
                                  store_x0, matrix_test)                             
    discovered_minimizers = []
    for j in range(int(p/2)):
        if j != pos:
            discovered_minimizers.append(j)
    c = prev_mt_alg.check_des_points(iterations_of_sd, discovered_minimizers, const)
    assert(c == None)


@settings(max_examples=10, deadline=None)
@given(st.integers(2, 20), st.integers(10, 50))
def test_4(p, d):
    """
    Check whether a local minimizer has already been identified by the METOD
    algorithm by applying prev_mt_alg.check_des_points(). The position of the identified
    local minimizer can be either c = len(discovered_minimizers)-1 or
    c = len(discovered_minimizers) - 2.
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
        store_x0[i] = np.random.uniform(0, 1, (d))
        store_rotation[i] = mt_obj.calculate_rotation_matrix(d, 3)
    matrix_test = (np.transpose(store_rotation, (0, 2, 1)) @ store_A @
                store_rotation)
    func_args = (p, store_x0, matrix_test)

    x = np.random.uniform(0,1,(d,))
    projection = False
    tolerance = 0.001
    option = 'minimize_scalar'
    met = 'brent'
    initial_guess = 0.005
    const = 0.1
    bound_1, bound_2 = 0,1 
    usage = 'metod_algorithm'
    relax_sd_it = 1
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    check_func = mt_obj.calc_minimizer_sev_quad
    (iterations_of_sd,
    its,
    store_grad_part) = (mt_alg.apply_sd_until_stopping_criteria
                        (x, d, projection, tolerance,
                        option, met, initial_guess,
                        func_args, f, g, bound_1, bound_2,
                        usage, relax_sd_it, init_grad=None))
    
    assert(calc_minimizer_sev_quad(x, p, store_x0, matrix_test) ==
           check_func(iterations_of_sd[-1], p,
                      store_x0, matrix_test))
    pos = calc_minimizer_sev_quad(iterations_of_sd[-1], p,
                                  store_x0, matrix_test)                             
    discovered_minimizers = []
    for j in range(int(p/2)):
        if j != pos:
            discovered_minimizers.append(j)
    discovered_minimizers.append(store_x0[pos])
    discovered_minimizers.append(store_x0[pos])
    c = prev_mt_alg.check_des_points(iterations_of_sd, discovered_minimizers, const)
    assert(c == len(discovered_minimizers)-1 or c == len(discovered_minimizers)-2)