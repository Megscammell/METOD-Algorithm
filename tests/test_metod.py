import numpy as np
import hypothesis
from hypothesis import assume, given, settings, strategies as st

import metod_testing as mtv3


def test_1():
    """Test that all parameters are checked before being applied in algorithm. 
    """
    num_points = 0.001
    ans_1 = True
    if isinstance(num_points, int) == False:
        ans_1 = False
    
    assert(ans_1 == False)

    num_points = 100
    ans_2 = True
    if isinstance(num_points, int) == False:
        ans_2 = False
    
    assert(ans_2 == True)


    proj = 0.999
    ans_3 = True
    if isinstance(proj, bool) == False:
        ans_3 = False
    
    assert(ans_3 == False)

    proj = True
    ans_4 = True
    if isinstance(proj, bool) == False:
        ans_4 = False
    
    assert(ans_4 == True)


@settings(max_examples=10, deadline=None)
@given(st.integers(2,20), st.integers(0,3), st.integers(2,100))
def test_2(p, m, d):
    """ Test warm up, m, is being applied correctly in metod when computing distances """
    np.random.seed(p)
    x = np.random.uniform(0, 1,(d,))
    tolerance = 0.00001
    projection = True
    initial_guess = 0.05
    option = 'minimize'
    met = 'Nelder-Mead'
    beta = 0.095
    matrix_test = np.zeros((p, d, d))
    store_x0 = np.random.uniform(0, 1, (p, d))

    diag_vals = np.zeros(d)
    diag_vals[:2] = np.array([1, 10])
    diag_vals[2:] = np.random.uniform(2, 9, (d - 2))
    matrix_test[0] = np.diag(diag_vals)   
    diag_vals = np.zeros(d)
    diag_vals[:2] = np.array([1, 10])
    diag_vals[2:] = np.random.uniform(2, 9, (d - 2))
    matrix_test[1] = np.diag(diag_vals)   

    func_args = p, store_x0, matrix_test
    f = mtv3.quad_function
    g = mtv3.quad_gradient
    initial_point = True
    iterations_of_sd, its = mtv3.apply_sd_until_stopping_criteria(
                            initial_point, x, d, projection, tolerance, option, met, initial_guess, func_args, f, g)

    #METOD algorithm checks the below
    assume(its > m)

    sd_iterations_partner_points = mtv3.partner_point_each_sd(iterations_of_sd, d, beta, its, g, func_args)

    test_x = np.random.uniform(0,1,(d,))
    
    original_shape = iterations_of_sd.shape[0]
    test_x = np.random.uniform(0,1,(d,))
    #Checking correct warm up applied when checking distances
    set_dist = mtv3.distances(iterations_of_sd, test_x, m, d)
    assert(set_dist.shape == (original_shape - m,))
    assert(set_dist.shape == (its + 1 - m,))
    assert(sd_iterations_partner_points.shape[0] == iterations_of_sd.shape[0])


@settings(max_examples=10, deadline=None)
@given(st.integers(2,20), st.integers(5,100), st.integers(50,1000))
def test_3(p, d, num_points_t):
    """ Check ouputs of algorithm with minimum of several quadratic forms function and gradient """ 
    np.random.seed(p)
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_test = mtv3.function_parameters_quad(p, d, lambda_1, lambda_2)
    func_args = p, store_x0, matrix_test 
    f = mtv3.quad_function
    g = mtv3.quad_gradient
    discovered_minimas, number_minimas, func_vals_of_minimas, number_excessive_descents  = mtv3.metod(f, g, func_args, d, num_points=num_points_t)

    #Check outputs are as expected
    assert(len(discovered_minimas) == number_minimas)
    assert(number_minimas == len(func_vals_of_minimas))

    norms_with_minima = np.zeros((number_minimas))
    pos_list = np.zeros((number_minimas))
    for j in range(number_minimas):
        pos, norm_minima = mtv3.calc_pos(discovered_minimas[j].reshape(d,), *func_args)
        pos_list[j] = pos
        norms_with_minima[j] = norm_minima

    #Ensures discovered minima is very close to actual minima
    assert(np.max(norms_with_minima) < 0.0001)

    #Ensure that each region of attraction discovered is unique
    assert(np.unique(pos_list).shape[0] == number_minimas)

def test_4():
    """ Checks ouputs of algorithm with Sum of Gaussians function and gradient""" 
    np.random.seed(11)
    d = 100
    p = 10
    sigma_sq = 2
    lambda_1 = 1
    lambda_2 = 10
    matrix_test = np.zeros((p, d, d))
    store_x0, matrix_test, store_c = mtv3.function_parameters_sog(
                                     p, d, lambda_1,lambda_2)
    args = p, sigma_sq, store_x0, matrix_test, store_c 
    f = mtv3.sog_function
    g = mtv3.sog_gradient
    discovered_minimas, number_minimas, func_vals_of_minimas, number_excessive_descents  = mtv3.metod(f, g, args, d)

    #Check outputs are as expected
    assert(len(discovered_minimas) == number_minimas)
    assert(number_minimas == len(func_vals_of_minimas))

    norms_with_minima = np.zeros((number_minimas))
    pos_list = np.zeros((number_minimas))
    for j in range(number_minimas):
        pos, min_dist = mtv3.calc_minima(discovered_minimas[j], *args)
        pos_list[j] = pos
        norms_with_minima[j] = min_dist

    #Ensures discovered minima is very close to actual minima
    assert(np.max(norms_with_minima) < 0.0001)

    #Ensure that each region of attraction discovered is unique
    assert(np.unique(pos_list).shape[0] == number_minimas)
