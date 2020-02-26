import numpy as np
import hypothesis
from hypothesis import assume, given, settings, strategies as st

import metod_testing as mtv3

def test_1():
    """Test that for loop takes correct point from all_iterations_of_sd array and stores correctly into 
    all_iterations_of_sd_test array
    """
    iterations = 10
    d = 5
    all_iterations_of_sd = np.random.uniform(0, 1, (iterations + 1, d))
    all_iterations_of_sd_test = np.zeros((iterations + 1, d))
    for its in range(iterations + 1):
        point = all_iterations_of_sd[its, :].reshape((d,))
        all_iterations_of_sd_test[its, :] = point.reshape(1, d)
    
    assert(np.all(all_iterations_of_sd_test == all_iterations_of_sd))

@settings(max_examples=50, deadline=None)
@given(st.integers(2, 10), st.integers(2, 100), st.integers(1,30))
def test_2(p, d, iterations):
    """Ensure size of iterations_of_sd is the same as partner_points_sd
    """
    beta = 0.005 
    g = mtv3.quad_function
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_test = mtv3.function_parameters_quad(p, d, lambda_1, lambda_2)
    func_args = p, store_x0, matrix_test
    iterations_of_sd = np.random.uniform(0, 1, (iterations + 1, d))
    partner_points_sd = mtv3.partner_point_each_sd(iterations_of_sd, d, beta, iterations, g, func_args)
    assert(partner_points_sd.shape[0] == iterations_of_sd.shape[0])
    assert(partner_points_sd.shape[1] == iterations_of_sd.shape[1])


# def test_3():
#     np.random.seed(90)
#     lambda_1 = 1
#     lambda_2 = 10
#     p = 10
#     d = 10
#     store_x0, matrix_test = mtv3.function_parameters_quad(p, d, lambda_1, lambda_2)
#     func_args = p, store_x0, matrix_test
#     tolerance = 0.00001
#     option = 'minimize'
#     met = 'Nelder-Mead'
#     initial_guess = 0.05
#     f = mtv3.quad_function
#     g = mtv3.quad_gradient
#     projection = True

#     point = np.random.uniform(0,1,(d,))
#     sd_iterations, iters = mtv3.apply_sd_until_stopping_criteria(
#                          point, d, projection, tolerance, option, met, initial_guess, func_args, f, g)

#     all_iterations_of_sd_partner_points = np.zeros((iters + 1, d))
#     beta = 0.005


#     val = 10

#     for its in range(iters + 1):
#         if its > 0  and its % val == 0:
#             beta = beta / 2
        
#         if its == 10:
#             assert beta == 0.005/2

#         if its == 20:
#             assert beta == 0.005/4

#         if its == 30:
#             assert beta == 0.005/8
        
#         if its == 40:
#             assert beta == 0.005/16
        
#         if its == 50:
#             assert beta == 0.005/32

#         point = sd_iterations[iters, :].reshape((d,))
#         partner_point = mtv3.partner_point(point, beta, d, g, func_args)
#         all_iterations_of_sd_partner_points[iters,:] = partner_point.reshape(1,d)






# def test_4():
#     np.random.seed(90)
#     lambda_1 = 1
#     lambda_2 = 10
#     p = 10
#     d = 20
#     store_x0, matrix_test = mtv3.function_parameters_quad(p, d, lambda_1, lambda_2)
#     func_args = p, store_x0, matrix_test
#     tolerance = 0.00001
#     option = 'minimize'
#     met = 'Nelder-Mead'
#     initial_guess = 0.05
#     f = mtv3.quad_function
#     g = mtv3.quad_gradient
#     projection = True
#     beta = 0.005

#     point = np.random.uniform(0,1,(d,))
#     sd_iterations, its = mtv3.apply_sd_until_stopping_criteria(
#                          point, d, projection, tolerance, option, met, initial_guess, func_args, f, g)

#     all_iterations_of_sd_partner_points = mtv3.partner_point_each_sd(sd_iterations, d, beta, its, g, func_args)

#     assert(beta == 0.005)