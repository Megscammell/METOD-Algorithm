import numpy as np

from metod_alg import objective_functions as mt_obj


def test_1():
    """Computational test for mt_obj.zakharov_func() with d = 3."""
    d = 3
    x = np.array([2, 3, 4])

    func_val = mt_obj.zakharov_func(x, d)
    assert(np.round(func_val, 1) == 10129.0)


def test_2():
    """Computational test for mt_obj.zakharov_grad() with d = 3."""
    d = 3
    x = np.array([2, 3, 4])

    grad_val = mt_obj.zakharov_grad(x, d)
    assert(np.all(np.round(grad_val, 1) == np.array([2014.0,
                                                     4026.0,
                                                     6038.0])))


def test_3():
    """
    Computational test for mt_obj.zakharov_func() and
    mt_obj.zakharov_grad() with d = 3.
    """
    d = 3
    x = np.zeros((d))
    assert(mt_obj.zakharov_func(x, d) == 0)
    assert(np.all(mt_obj.zakharov_grad(x, d) == np.zeros((d))))
