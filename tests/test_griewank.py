import numpy as np

from metod_alg import objective_functions as mt_obj


def test_1():
    d = 3
    x = np.array([1, 1.5, 2])

    func_val = mt_obj.griewank_func(x, d)
    assert(np.round(func_val, 6) == 0.895175)


def test_2():
    d = 3
    x = np.array([1, 1.5, 2])

    grad_val = mt_obj.griewank_grad(x, d)
    assert(np.all(np.round(grad_val, 6) == np.array([0.166577,
                                                     0.135511,
                                                     0.140324])))


def test_3():
    d = 3
    x = np.zeros((d))
    func_val = mt_obj.griewank_func(x, d)
    assert(func_val == 0)
    assert(np.all(mt_obj.griewank_grad(x, d) == np.zeros((d))))
