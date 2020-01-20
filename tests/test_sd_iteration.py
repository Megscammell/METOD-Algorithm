import numpy as np


def test_1():
    """Testing np.clip method with for loop to ensure points projected correctly.
    """
    projection = True
    x = np.array([0.1, 0.5, 0.9, -0.9, 1.1, 0.9, -0.2, 1.1, 0.1, -0.5])
    old_x = np.copy(x)
    if projection == True:
        for j in range(10):
            if x[j] > 1:
                x[j] = 1
            if x[j] < 0:
                x[j] = 0
    
    assert(np.all(x == np.clip(old_x, 0, 1)))
    
