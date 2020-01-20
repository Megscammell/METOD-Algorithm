import numpy as np
import hypothesis
from hypothesis import assume, given, settings, strategies as st

import metod_testing as mtv3


@settings(max_examples=10, deadline=None)
@given(st.integers(2,20), st.integers(5,100), st.integers(50,1000))
def test_1(p, d, num_points_t):
    """ Check warm up within metod_indepth only applied when checking distances.""" 
    np.random.seed(d + 10)
    lambda_1 = 1
    lambda_2 = 10
    
    store_x0, matrix_test = mtv3.function_parameters_quad(p, d, lambda_1, lambda_2)
    func_args = p, store_x0, matrix_test 
    f = mtv3.quad_function
    g = mtv3.quad_gradient
    discovered_minimas, number_minimas, func_vals_of_minimas, number_excessive_descents, store_its, des_x_points, des_z_points, starting_points  = mtv3.metod_indepth(f, g, func_args, d, num_points=num_points_t)

    assert(len(discovered_minimas) == number_minimas)
    assert(number_minimas == len(func_vals_of_minimas))
    assert(starting_points.shape[0] == num_points_t)
    norms_with_minima = np.zeros((number_minimas))
    pos_list = np.zeros((number_minimas))
    for j in range(number_minimas):
        pos, norm_minima = mtv3.calc_pos(discovered_minimas[j].reshape(d,), *func_args)
        pos_list[j] = pos
        norms_with_minima[j] = norm_minima

    assert(np.max(norms_with_minima) < 0.0001)
    assert(np.unique(pos_list).shape[0] == number_minimas)

    #Extra assert statements compared to metod
    assert(len(des_x_points) == len(store_its))
    for j in range(len(des_x_points)):
        assert(des_x_points[j].shape[0] - 1 == store_its[j])
        assert(des_x_points[j].shape[0] == des_z_points[j].shape[0])
