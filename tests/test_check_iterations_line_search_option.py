import numpy as np

import metod_testing as mtv3


def test_1():
    """Particular case for line_search where solver doesn't converge. Will produce a warning message but want to check that the point is replaced and iterations are started again from the new point. Checks that the starting point is different to that in sd_iterations[0,:]. Also checks that sd_iterations[its,:] is the minima corresponding to sd_iterations[0,:].
    """    
    d = 20
    p = 10
    lambda_1 = 1
    lambda_2 = 10
    tolerance = 0.00001
    projection = True
    option = 'line_search'
    initial_guess = 0.05
    met = 'None'
    f = mtv3.quad_function
    g = mtv3.quad_gradient
    np.random.seed(2*222+1)
    store_x0, matrix_test = mtv3.function_parameters_quad(p, d, lambda_1, lambda_2)
    func_args = p, store_x0, matrix_test
    x = np.random.uniform(0,1,(d,))
    initial_point = True
    sd_iterations, its = mtv3.apply_sd_until_stopping_criteria                                      (initial_point, x, d, projection,                                           tolerance, option, met, initial_guess,                                     func_args, f, g)
    assert(np.any(x != sd_iterations[0,:]))

    assert(mtv3.calc_pos(x, *func_args)[0] != mtv3.calc_pos(sd_iterations[its,:], *func_args)[0])

    assert(mtv3.calc_pos(sd_iterations[0,:], *func_args)[0] == mtv3.calc_pos                                                              (sd_iterations                                                              [its, :],                                                                  *func_args)[0])

