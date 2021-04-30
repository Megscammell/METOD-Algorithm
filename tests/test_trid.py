import numpy as np

from metod_alg import objective_functions as mt_obj


def test_1():
    d = 4
    x = np.array([1, 5, 2, 8])

    func_val = mt_obj.trid_func(x, d)
    assert(np.round(func_val, 1) == 35.0)


def test_2():
    d = 4
    x = np.array([1, 5, 2, 8])

    grad_val = mt_obj.trid_grad(x, d)
    assert(np.all(np.round(grad_val, 1) == np.array([-5.0,
                                                     5.0,
                                                     -11.0,
                                                     12])))


def test_3():
    d = 4
    x_min = np.zeros((d))
    for i in range(d):
        x_min[i] = (i + 1) * (d + 1 - (i + 1))
    
    func_val_min = (-d * (d + 4) * (d - 1)) / 6 
    assert(np.round(mt_obj.trid_func(x_min, d), 1) == func_val_min)
    assert(np.all(mt_obj.trid_grad(x_min, d) == np.zeros((d))))
