import numpy as np

from metod_alg import objective_functions as mt_obj
from metod_alg import metod_algorithm_functions as mt_alg


def test_1():
    """
    Test for mt_alg.forward_tracking() - check that when flag=True,
    track is updated.
    """
    np.random.seed(90)
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    d = 10
    P = 5
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_combined = (mt_obj.function_parameters_several_quad
                                 (P, d, lambda_1, lambda_2))
    func_args = P, store_x0, matrix_combined
    point = np.random.uniform(0, 1, (d, ))
    const_forward = 1.1
    forward_tol = 1000000000
    step = 0.0001
    grad = g(point, *func_args)
    f_old = f(np.copy(point), *func_args)
    f_new = f(np.copy(point) - step * grad, *func_args)
    assert(f_old > f_new)
    track, flag = (mt_alg.forward_tracking(
                   point, step, f_old, f_new, grad,
                   const_forward, forward_tol, f,
                   func_args))
    assert(flag == True)
    assert(track[0][0] == 0)
    for j in range(1, len(track)):
        assert(track[j][0] == step)
        step = step * const_forward
        if j < len(track) - 1:
            assert(track[j][1] < track[j-1][1])
        else:
            assert(track[j][1] > track[j-1][1])


def test_2():
    """
    Test for mt_alg.forward_tracking() - check for flag=False.
    """
    np.random.seed(90)
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    d = 10
    P = 5
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_combined = (mt_obj.function_parameters_several_quad
                                 (P, d, lambda_1, lambda_2))
    func_args = P, store_x0, matrix_combined
    point = np.random.uniform(0, 1, (d, ))
    const_forward = 1.1
    forward_tol = 0.001
    step = 0.0001
    grad = g(point, *func_args)
    f_old = f(np.copy(point), *func_args)
    f_new = f(np.copy(point) - step * grad, *func_args)
    assert(f_old > f_new)
    track, flag = (mt_alg.forward_tracking(
                   point, step, f_old, f_new, grad,
                   const_forward, forward_tol, f,
                   func_args))
    assert(flag == False)
    for j in range(1, len(track)):
        assert(track[j][0] == step)
        step = step * const_forward
        assert(track[j][1] < track[j-1][1])


def test_3():
    """Test for mt_alg.backward_tracking() - back_tol is met"""
    np.random.seed(90)
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    d = 10
    P = 5
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_combined = (mt_obj.function_parameters_several_quad
                                 (P, d, lambda_1, lambda_2))
    func_args = P, store_x0, matrix_combined
    point = np.random.uniform(0, 1, (d, ))
    const_back = 0.9
    back_tol = 0.4
    step = 0.9
    grad = g(point, *func_args)
    f_old = f(np.copy(point), *func_args)
    f_new = f(np.copy(point) - step * grad, *func_args)
    assert(f_old < f_new)
    track = (mt_alg.backward_tracking
             (point, step, f_old, f_new,
              grad, const_back, back_tol,
              f, func_args))
    assert(track[0][0] == 0)
    assert(track[0][1] == f_old)
    assert(track[1][0] == step)
    assert(track[1][1] == f_new)


def test_4():
    """Test for mt_alg.backward_tracking() - back tol is not met"""
    np.random.seed(90)
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    d = 10
    P = 5
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_combined = (mt_obj.function_parameters_several_quad
                                 (P, d, lambda_1, lambda_2))
    func_args = P, store_x0, matrix_combined
    point = np.random.uniform(0, 1, (d, ))
    const_back = 0.9
    back_tol = 0.000001
    step = 1
    grad = g(point, *func_args)
    f_old = f(np.copy(point), *func_args)
    f_new = f(np.copy(point) - step * grad, *func_args)
    assert(f_old < f_new)
    track = (mt_alg.backward_tracking
             (point, step, f_old, f_new,
              grad, const_back, back_tol,
              f, func_args))

    assert(track[0][0] == 0)
    assert(track[0][1] == f_old)
    for j in range(1, len(track)):
        assert(track[j][0] == step)
        step = step * const_back
        if j < len(track) - 1:
            assert(track[0][1] < track[j][1])
        else:
            assert(track[j][1] < track[0][1])


def test_5():
    """Checks computation in mt_alg.compute_coeffs()"""
    track_y = np.array([100, 200, 50])
    track_t = np.array([0, 1, 0.5])
    opt_t = mt_alg.compute_coeffs(track_y, track_t)
    OLS_polyfit = np.polyfit(track_t, track_y, 2)

    check = -OLS_polyfit[1] / (2 * OLS_polyfit[0])
    assert(np.all(np.round(check, 5) == np.round(opt_t, 5)))


def test_6():
    """
    Test for mt_alg.combine_tracking() - check that correct step size is
    returned when forward_tol is not met.
    """
    np.random.seed(90)
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    d = 10
    P = 5
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_combined = (mt_obj.function_parameters_several_quad
                                 (P, d, lambda_1, lambda_2))
    func_args = P, store_x0, matrix_combined
    point = np.random.uniform(0, 1, (d, ))
    const_back = 0.9
    const_forward = 1.1
    forward_tol = 1000000000
    back_tol = 0.000001
    step = 0.0001
    f_old = f(np.copy(point), *func_args)
    grad = g(point, *func_args)
    opt_t = (mt_alg.combine_tracking
             (point, f_old, grad, step,
              const_back, back_tol,
              const_forward, forward_tol,
              f, func_args))
    assert(opt_t >= 0)
    upd_point = point - opt_t * grad
    assert(f(upd_point, *func_args) < f_old)


def test_7():
    """
    Test for mt_alg.combine_tracking() - check that correct step size is
    returned, when forward_tol is met.
    """
    np.random.seed(90)
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    d = 10
    P = 5
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_combined = (mt_obj.function_parameters_several_quad
                                 (P, d, lambda_1, lambda_2))
    func_args = P, store_x0, matrix_combined
    point = np.random.uniform(0, 1, (d, ))
    const_back = 0.9
    const_forward = 1.1
    forward_tol = 0.001
    back_tol = 0.000001
    step = 0.0001
    grad = g(point, *func_args)
    f_old = f(np.copy(point), *func_args)
    opt_t = (mt_alg.combine_tracking
             (point, f_old, grad, step,
              const_back, back_tol,
              const_forward, forward_tol,
              f, func_args))
    assert(opt_t >= 0)
    upd_point = point - opt_t * grad
    assert(f(upd_point, *func_args) < f_old)


def test_8():
    """
    Test for mt_alg.combine_tracking() - check that correct step size is
    returned, when back_tol is met.
    """
    np.random.seed(90)
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    d = 10
    P = 5
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_combined = (mt_obj.function_parameters_several_quad
                                 (P, d, lambda_1, lambda_2))
    func_args = P, store_x0, matrix_combined
    point = np.random.uniform(0, 1, (d, ))
    const_back = 0.9
    const_forward = 1.1
    back_tol = 0.4
    forward_tol = 100000000
    step = 0.9
    grad = g(point, *func_args)
    f_old = f(np.copy(point), *func_args)
    opt_t = (mt_alg.combine_tracking
             (point, f_old, grad, step,
              const_back, back_tol,
              const_forward, forward_tol,
              f, func_args))
    assert(opt_t == 0)
    upd_point = point - opt_t * grad
    assert(f(upd_point, *func_args) == f_old)


def test_9():
    """
    Test for mt_alg.combine_tracking() - check that correct step size is
    returned, when back_tol is not met.
    """
    np.random.seed(90)
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    d = 10
    P = 5
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_combined = (mt_obj.function_parameters_several_quad
                                 (P, d, lambda_1, lambda_2))
    func_args = P, store_x0, matrix_combined
    point = np.random.uniform(0, 1, (d, ))
    const_back = 0.9
    const_forward = 1.1
    back_tol = 0.000001
    forward_tol = 100000000
    step = 1
    grad = g(point, *func_args)
    f_old = f(np.copy(point), *func_args)
    opt_t = (mt_alg.combine_tracking
             (point, f_old, grad, step,
              const_back, back_tol,
              const_forward, forward_tol,
              f, func_args))
    assert(opt_t >= 0)
    upd_point = point - opt_t * grad
    assert(f(upd_point, *func_args) < f_old)


def test_10():
    """Test that mt_alg.arrange_track_y_t produces expected outputs"""
    track = np.array([[0, 100],
                      [1, 80],
                      [2, 160],
                      [4, 40],
                      [8, 20],
                      [16, 90]])
    track_method = 'Forward'
    track_y, track_t = mt_alg.arrange_track_y_t(track, track_method)
    assert(np.all(track_y == np.array([40, 20, 90])))
    assert(np.all(track_t == np.array([4, 8, 16])))


def test_11():
    """Test that mt_alg.arrange_track_y_t produces expected outputs"""
    track = np.array([[0, 100],
                      [1, 80],
                      [2, 70],
                      [4, 90]])
    track_method = 'Forward'
    track_y, track_t = mt_alg.arrange_track_y_t(track, track_method)
    assert(np.all(track_y == np.array([80, 70, 90])))
    assert(np.all(track_t == np.array([1, 2, 4])))


def test_12():
    """Test that mt_alg.arrange_track_y_t produces expected outputs"""
    track = np.array([[0, 100],
                      [1, 120],
                      [0.5, 110],
                      [0.25, 90]])
    track_method = 'Backward'
    track_y, track_t = mt_alg.arrange_track_y_t(track, track_method)
    assert(np.all(track_y == np.array([100, 90, 110])))
    assert(np.all(track_t == np.array([0, 0.25, 0.5])))


def test_13():
    """
    Test for mt_alg.check_func_val_coeffs() when func_val < track_y[1].
    """
    np.random.seed(90)
    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    d = 10
    P = 5
    lambda_1 = 1
    lambda_2 = 10
    store_x0, matrix_combined = (mt_obj.function_parameters_several_quad
                                 (P, d, lambda_1, lambda_2))
    func_args = P, store_x0, matrix_combined
    step = 0.00001
    point = np.array([0.5525204, 0.8256308, 0.5034502, 0.68755988,
                      0.75954891, 0.64230399, 0.38500431, 0.0801039,
                      0.80748984, 0.81147401])
    grad = g(point, *func_args)

    f_old = f(np.copy(point), *func_args)
    f_new = f(np.copy(point) - step * grad, *func_args)
    assert(f_old > f_new)
    forward_tol = 100000000
    const_forward = 1.1
    track_method = 'Forward'
    track, flag = (mt_alg.forward_tracking
                   (point, step, f_old, f_new, grad,
                    const_forward, forward_tol, f, func_args))
    opt_t = mt_alg.check_func_val_coeffs(track, track_method, point, grad, f,
                                         func_args)
    assert(f(point - opt_t * grad, *func_args) < np.min(track[:, 1]))


def test_14():
    """
    Test for mt_alg.check_func_val_coeffs() when func_val > track_y[1].
    """
    np.random.seed(34272212)
    f = mt_obj.sog_function
    g = mt_obj.sog_gradient
    d = 20
    P = 10
    lambda_1 = 1
    lambda_2 = 4
    sigma_sq = 0.8
    store_x0, matrix_combined, store_c = (mt_obj.function_parameters_sog
                                          (P, d, lambda_1, lambda_2))
    func_args = P, sigma_sq, store_x0, matrix_combined, store_c
    point = np.random.uniform(0, 1, (d,))
    f_old = f(point, *func_args)
    grad = g(point, *func_args)
    step = 0.1
    const_forward = 1.5
    forward_tol = 1000000000
    f_new = f(np.copy(point) - step * grad, *func_args)
    assert(f_old > f_new)
    track_method = 'Forward'
    track, flag = (mt_alg.forward_tracking
                   (point, step, f_old, f_new, grad,
                    const_forward, forward_tol, f, func_args))
    opt_t = mt_alg.check_func_val_coeffs(track, track_method, point, grad, f,
                                         func_args)
    pos = np.argmin(track[:, 1])
    step_length = track[pos][0]
    assert(step_length == opt_t)
