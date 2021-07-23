import numpy as np

from metod_alg import objective_functions as mt_obj
from metod_alg import metod_algorithm_functions as mt_alg


def test_1():
    """Computational test for mt_obj.hartmann6_func() with d = 3."""
    a = np.array([[1, 2, 3],
                  [4, 5, 6]])
    c = np.array([1, 1.5])
    p = np.array([[0.1, 0.2, 0.5],
                  [0.9, 0.8, 0.1]])
    x = np.array([1, 1, 1])
    d = 3

    func_val = mt_obj.hartmann6_func(x, d, a, c, p)
    assert(np.round(func_val, 5) == -0.06757)


def test_2():
    """Computational test for mt_obj.hartmann6_grad() with d = 3."""
    a = np.array([[1, 2, 3],
                  [4, 5, 6]])
    c = np.array([1, 1.5])
    p = np.array([[0.1, 0.2, 0.5],
                  [0.9, 0.8, 0.1]])
    x = np.array([1, 1, 1])
    d = 3

    grad_val = mt_obj.hartmann6_grad(x, d, a, c, p)
    assert(np.all(np.round(grad_val, 5) ==
           np.array([0.11248, 0.20525, 0.27404])))


def test_3():
    """
    Computational test for mt_obj.hartmann6_func_params(),
    mt_obj.hartmann6_func() and mt_obj.hartmann6_grad(),
    with d = 6.
    """
    d = 6
    x = np.array([1, 1, 1, 1, 1, 1])
    a, c, p = mt_obj.hartmann6_func_params()
    func_val = mt_obj.hartmann6_func(x, d, a, c, p)
    assert(isinstance(func_val, float))

    grad_val = mt_obj.hartmann6_grad(x, d, a, c, p)
    assert(grad_val.shape == (d,))


def test_4():
    """
    Computational test for mt_obj.hartmann6_func_params(),
    mt_obj.hartmann6_func() and mt_obj.hartmann6_grad(),
    with d = 6.
    """
    d = 6
    x = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    a, c, p = mt_obj.hartmann6_func_params()
    func_val = mt_obj.hartmann6_func(x, d, a, c, p)
    assert(np.round(func_val, 5) == -3.32237)
    assert(np.linalg.norm(mt_obj.hartmann6_grad(x, d, a, c, p)) < 0.0001)


def test_5():
    """
    Computational test for mt_obj.calc_minimizer_hartmann6(),
    with d = 6.
    """
    d = 6
    x = np.random.uniform(0, 1, (d, ))
    a, c, p = mt_obj.hartmann6_func_params()
    f = mt_obj.hartmann6_func
    g = mt_obj.hartmann6_grad
    tolerance = 0.01
    projection = False
    option = 'forward_backward_tracking'
    met = 'None'
    initial_guess = 0.005
    bound_1 = 0
    bound_2 = 1
    relax_sd_it = 1
    usage_choice = 'metod_algorithm'
    func_args = (d, a, c, p)
    (iterations_of_sd,
     its, store_grad) = (mt_alg.apply_sd_until_stopping_criteria
                         (x, d, projection,
                          tolerance, option,
                          met, initial_guess,
                          func_args, f, g,
                          bound_1,
                          bound_2,
                          usage_choice,
                          relax_sd_it, None))
    pos = mt_obj.calc_minimizer_hartmann6(iterations_of_sd[its])
    assert(pos in [0, 1])


def test_6():
    """
    Computational test for mt_obj.calc_minimizer_hartmann6(),
    with d = 6. That is, check that x is not near any local
    minimizers.
    """
    d = 6
    x = np.random.uniform(0, 1, (d, ))
    pos = mt_obj.calc_minimizer_hartmann6(x)
    assert(pos is None)
