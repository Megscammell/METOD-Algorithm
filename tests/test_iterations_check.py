import numpy as np


def test_1():
    """Checking functionality of np.vstack and ensuring it stores points as expected.
    """
    d = 10
    store_x = np.zeros((2, d))
    x = np.arange(1, 11).reshape(d, 1)
    store_x[0] = x.reshape(1, d)
    x = np.arange(11, 21).reshape(d, 1)
    store_x[1] = x.reshape(1, d)
    for j in range(2, 8):
        x = np.arange((j * 10) + 1, ((j * 10) + 11)).reshape(d, 1)
        store_x = np.vstack([store_x, x.reshape(1, d)])
    print(store_x)
    assert(np.all(store_x[2] == np.arange(21, 31)))
    assert(np.all(store_x[3] == np.arange(31, 41)))
    assert(np.all(store_x[4] == np.arange(41, 51)))
    assert(np.all(store_x[5] == np.arange(51, 61)))
    assert(np.all(store_x[6] == np.arange(61, 71)))
    assert(np.all(store_x[7] == np.arange(71, 81)))

def test_2():
    """Checking new points generated due to change_point == True are stored correctly i.e points_x is a new
    array that includes the starting point and one iteration of steepest descent.
    """
    d = 5
    its = 9
    points_x = np.random.uniform(0, 1, (10, d))
    change_point = True
    while change_point == True:
        its = 0
        points_x = np.zeros((1, d))
        point = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        points_x[0] = point.reshape(1, d)
        change_point = False
    
    assert(its == 0)
    assert(points_x.shape[0] == 1)
    assert(points_x.shape[1] == d)
    assert(np.all(points_x[0, :] == np.array([0.1, 0.1, 0.1, 0.1, 0.1])))
    
