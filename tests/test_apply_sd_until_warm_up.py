import numpy as np
from hypothesis import given, settings, strategies as st

from metod_alg import metod_algorithm_functions as mt_alg
from metod_alg import objective_functions as mt_obj


def test_1():
    """
    Checks that 'while count < m', produces 3 points from
    np.random.uniform(0, 1, (d, 1)) when m = 3.
    """
    m = 3
    d = 10
    count = 0
    test = []
    while count < m:
        count += 1
        x = np.random.uniform(0, 1, (d, ))
        test.append(x)
    assert(len(test) == 3)


@settings(deadline=None)
@given(st.integers(2, 20), st.integers(2, 100))
def test_2(p, d):
    """
    Check that apply_sd_until_warm_up.py produces points and
    corresponding partner points with m=3 iterations of steepest descent
    applied.
    """
    beta = 0.099
    projection = False
    lambda_1 = 1
    lambda_2 = 10
    option = 'minimize_scalar'
    met = 'Brent'
    initial_guess = 0.005
    store_grad = np.zeros((4, d))
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    m = 3
    """Create objective function parameters"""
    store_x0, matrix_test = (mt_obj.function_parameters_several_quad
                             (p, d, lambda_1, lambda_2))
    func_args = p, store_x0, matrix_test
    """Generate random starting point"""
    bound_1 = 0
    bound_2 = 1
    relax_sd_it = 1
    x = np.random.uniform(bound_1, bound_2, (d, ))
    """Apply one iteration of steepest descent"""
    store_grad[0] = g(x, *func_args)
    x_1 = mt_alg.sd_iteration(x, projection, option, met, initial_guess,
                              func_args, f, store_grad[0], bound_1, bound_2,
                              relax_sd_it)
    """Apply second iteration of steepest descent"""
    store_grad[1] = g(x_1, *func_args)
    x_2 = mt_alg.sd_iteration(x_1, projection, option, met, initial_guess,
                              func_args, f, store_grad[1], bound_1, bound_2,
                              relax_sd_it)
    """Apply third iteration of steepest descent"""
    store_grad[2] = g(x_2, *func_args)
    x_3 = mt_alg.sd_iteration(x_2, projection, option, met, initial_guess,
                              func_args, f, store_grad[2], bound_1, bound_2,
                              relax_sd_it)
    store_grad[3] = g(x_3, *func_args)
    store_x_2_x_3 = np.zeros((2, d))
    store_x_2_x_3[0, :] = x_2
    store_x_2_x_3[1, :] = x_3
    """Compute corresponding partner points"""
    partner_points = mt_alg.partner_point_each_sd(store_x_2_x_3,
                                                  beta,
                                                  store_grad[-2:, :])
    z_2 = partner_points[0, :].reshape(d, )
    z_3 = partner_points[1, :].reshape(d, )

    init_grad = store_grad[0]
    (sd_iterations,
     sd_iterations_partner_points,
     store_grad_test) = (mt_alg.apply_sd_until_warm_up
                         (x, d, m, beta, projection,
                          option, met, initial_guess,
                          func_args, f, g, bound_1,
                          bound_2, relax_sd_it,
                          init_grad))
    assert(np.all(x_2 == sd_iterations[m - 1]))
    assert(np.all(x_3 == sd_iterations[m]))
    assert(np.all(z_2 == sd_iterations_partner_points[m - 1]))
    assert(np.all(z_3 == sd_iterations_partner_points[m]))
    assert(sd_iterations.shape[0] == m + 1)
    assert(sd_iterations.shape[1] == d)
    assert(np.all(store_grad_test == store_grad))


@settings(max_examples=10, deadline=None)
@given(st.integers(2, 20), st.integers(1, 5), st.integers(2, 100))
def test_3(p, m, d):
    """
    Consider sd_iterations returned by apply_sd_until_warm_up.py. In order to
    continue steepest descent iterations until some stopping condition is met,
    we take the final point of sd_iterations and run
    apply_sd_until_stopping_criteria.py.
    Test checks that steepest descent iterations from an initial point
    (apply_sd_until_stopping_criteria.py) are the same as when
    apply_sd_until_warm_up.py and apply_sd_until_stopping_criteria.py are
    applied.
    """
    beta = 0.099
    tolerance = 0.00001
    projection = False
    lambda_1 = 1
    lambda_2 = 10
    option = 'minimize_scalar'
    met = 'Brent'
    initial_guess = 0.005
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    """Create objective function parameters"""
    store_x0, matrix_test = (mt_obj.function_parameters_several_quad
                             (p, d, lambda_1, lambda_2))
    func_args = p, store_x0, matrix_test
    """Generate random starting point"""
    bound_1 = 0
    bound_2 = 1
    usage = 'metod_algorithm'
    relax_sd_it = 1
    x = np.random.uniform(bound_1, bound_2, (d, ))
    init_grad = g(x, *func_args)
    (warm_up_sd,
     warm_up_sd_partner_points,
     store_grad_test) = (mt_alg.apply_sd_until_warm_up
                         (x, d, m, beta, projection,
                          option, met, initial_guess,
                          func_args, f, g, bound_1,
                          bound_2, relax_sd_it,
                          init_grad))
    x_2 = warm_up_sd[m].reshape(d, )
    (iterations_of_sd_part,
     its,
     store_grad_part) = (mt_alg.apply_sd_until_stopping_criteria
                         (x_2, d, projection, tolerance, option, met,
                          initial_guess, func_args, f, g, bound_1,
                          bound_2, usage, relax_sd_it, store_grad_test[-1]))
    iterations_of_sd = np.vstack([warm_up_sd, iterations_of_sd_part[1:, ]])

    iterations_of_sd_part_partner_point = (mt_alg.partner_point_each_sd
                                           (iterations_of_sd_part, beta,
                                            store_grad_part))
    sd_iterations_partner_points = (np.vstack(
                                    [warm_up_sd_partner_points,
                                     iterations_of_sd_part_partner_point[1:, ]
                                     ]))

    all_grad_store = np.vstack([store_grad_test,
                                store_grad_part[1:, ]])

    (iterations_of_sd_test,
     its_test,
     store_grad) = (mt_alg.apply_sd_until_stopping_criteria
                    (x, d, projection, tolerance, option,
                     met, initial_guess, func_args, f, g,
                     bound_1, bound_2, usage, relax_sd_it, g(x, *func_args)))
    sd_iterations_partner_points_test = (mt_alg.partner_point_each_sd
                                         (iterations_of_sd_test, beta,
                                          store_grad))

    assert(np.all(np.round(iterations_of_sd_test, 4) == np.round
           (iterations_of_sd, 4)))

    assert(np.all(np.round(sd_iterations_partner_points_test, 4) == np.round
           (sd_iterations_partner_points, 4)))

    assert(iterations_of_sd_test.shape[0] == iterations_of_sd.shape[0])

    assert(sd_iterations_partner_points_test.shape[0] ==
           sd_iterations_partner_points.shape[0])

    assert(its_test == its + m)

    assert(np.all(all_grad_store == store_grad))
