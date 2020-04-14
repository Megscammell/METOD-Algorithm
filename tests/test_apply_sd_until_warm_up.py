import numpy as np
import hypothesis
from hypothesis import assume, given, settings, strategies as st

import metod_testing as mtv3


def test_1():
    """Check that while count < m, produces 3 points from np.random.uniform(0,1,(d,1)) when m = 3.
    """
    m = 3
    d = 10
    count = 0
    test = []
    while count < m:
        count += 1
        x = np.random.uniform(0,1,(d,))
        test.append(x)
    assert(len(test) == 3)

@settings(deadline=None)
@given(st.integers(2,20), st.integers(2,100))
def test_2(p, d):
    """Check that apply_sd_until_warm_up produces points and corresponding partner points with m - 1 and m iterations of steepest descent applied
    """
    beta = 0.099
    tolerance = 0.00001
    initial_guess = 0.05
    projection = False
    lambda_1 = 1
    lambda_2 = 10
    option = 'minimize'
    met = 'Nelder-Mead'
    f = mtv3.quad_function
    g = mtv3.quad_gradient
    m = 3

    #Create objective function parameters
    store_x0, matrix_test = mtv3.function_parameters_quad(p, d, lambda_1, lambda_2)  
    func_args = p, store_x0, matrix_test

    #Generate random starting point
    bound_1 = 0
    bound_2 = 1
    x = np.random.uniform(bound_1,bound_2,(d,))
    #Apply one iteration of steepest descent
    x_1 = mtv3.sd_iteration(x, projection, option, met, initial_guess, func_args, f, g, bound_1,bound_2)
    #Apply second iteration of steepest descent
    x_2 = mtv3.sd_iteration(x_1, projection, option, met, initial_guess, func_args, f, g, bound_1,bound_2)
    #Apply third iteration of steepest descent
    x_3 = mtv3.sd_iteration(x_2, projection, option, met, initial_guess, func_args, f, g, bound_1,bound_2)
    #Compute corresponding partner point for x_2
    z_2 = mtv3.partner_point(x_2, beta, d, g, func_args)
    #Compute corresponding partner point for x_3
    z_3 = mtv3.partner_point(x_3, beta, d, g, func_args)
    
    sd_iterations, sd_iterations_partner_points = mtv3.apply_sd_until_warm_up(x, d, m, beta, projection, option, met, initial_guess, func_args, f, g, bound_1,bound_2)

    assert(np.all(x_2 == sd_iterations[m - 1]))
    assert(np.all(x_3 == sd_iterations[m]))
    assert(np.all(z_2 == sd_iterations_partner_points[m - 1]))
    assert(np.all(z_3 == sd_iterations_partner_points[m]))

@settings(max_examples=10, deadline=None)
@given(st.integers(2,20), st.integers(1,5), st.integers(2,100))
def test_3(p, m, d):
    """Check that continued iterations from x_2 to a minimizer, (iterations_of_sd_part), joined with the initial warm up points (warm_up_sd), has the same points and shape as when initial point x has steepest descent iterations applied to find the minimizer.
    """
    beta = 0.099
    tolerance = 0.00001
    initial_guess = 0.05
    projection = False
    lambda_1 = 1
    lambda_2 = 10
    option = 'minimize'
    met = 'Nelder-Mead'
    f = mtv3.quad_function
    g = mtv3.quad_gradient

    #Create objective function parameters
    store_x0, matrix_test = mtv3.function_parameters_quad(p, d, lambda_1, lambda_2)  
    func_args = p, store_x0, matrix_test
    #Generate random starting point
    bound_1 = 0
    bound_2 = 1
    x = np.random.uniform(bound_1,bound_2,(d,))  
    warm_up_sd, warm_up_sd_partner_points = mtv3.apply_sd_until_warm_up(
                                            x, d, m, beta, projection, option, met, initial_guess, func_args, f, g, bound_1,bound_2)

    x_1 = warm_up_sd[m - 1].reshape(d,)
    z_1 = warm_up_sd_partner_points[m - 1].reshape(d,)
    x_2 = warm_up_sd[m].reshape(d,)
    z_2 = warm_up_sd_partner_points[m].reshape(d,) 

    iterations_of_sd_part, its = mtv3.apply_sd_until_stopping_criteria(x_2, d, projection, tolerance, option, met, initial_guess, func_args, f, g, bound_1,bound_2)

    iterations_of_sd = np.vstack([warm_up_sd,iterations_of_sd_part[1:,]])                        

    sd_iterations_partner_points = mtv3.partner_point_each_sd(iterations_of_sd, d, beta, its + m, g, func_args)

    iterations_of_sd_test, its_test = mtv3.apply_sd_until_stopping_criteria ( x, d, projection, tolerance, option, met, initial_guess, func_args, f, g, bound_1,bound_2)


    sd_iterations_partner_points_test = mtv3.partner_point_each_sd(iterations_of_sd_test, d,beta, its_test, g, func_args)

    assert(np.all(np.round(iterations_of_sd_test,4) == np.round(iterations_of_sd,4)))
    assert(np.all(np.round(sd_iterations_partner_points_test,4) == np.round(sd_iterations_partner_points,4)))

    assert(iterations_of_sd_test.shape[0] == iterations_of_sd.shape[0])
    assert(sd_iterations_partner_points_test.shape[0] == sd_iterations_partner_points.shape[0])
    assert(its_test == its + m)


