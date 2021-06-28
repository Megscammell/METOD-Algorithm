import numpy as np
import SALib
from SALib.sample import sobol_sequence
from hypothesis import assume, given, settings, strategies as st

import metod_alg as mt
from metod_alg import objective_functions as mt_obj
from metod_alg import metod_algorithm_functions as mt_alg


@settings(max_examples=10, deadline=None)
@given(st.integers(1, 10000), st.integers(2, 100))
def test_1(num_points, d):
    """
    Checks that np.random.seed reproduces the same shuffle.
    """    

    bounds_set_x = (0, 1)
    np.random.seed(90)
    diff = bounds_set_x[1] - bounds_set_x[0]
    temp_sobol_points = sobol_sequence.sample(num_points, d)
    sobol_points = temp_sobol_points * (-diff) + bounds_set_x[1]
    np.random.shuffle(sobol_points)

    np.random.seed(90)
    sobol_points_test = mt_alg.create_sobol_sequence_points(bounds_set_x[0],
                                                            bounds_set_x[1], d, num_points)
    assert(np.all(sobol_points_test == sobol_points))
    assert(sobol_points_test.shape == (num_points, d))


@settings(max_examples=10, deadline=None)
@given(st.integers(1, 10000), st.integers(2, 100), st.integers(-5, -1),
       st.integers(0, 5))
def test_2(num_points, d, a, b):
    """
    Checks that points within bounds are correct.
    """    

    bounds_set_x = (a, b)
    np.random.seed(90)
    sobol_points = mt_alg.create_sobol_sequence_points(bounds_set_x[0],
                                                       bounds_set_x[1], d, 
                                                       num_points)
    assert(sobol_points.shape == (num_points, d))
    assert(np.all(sobol_points >= a))
    assert(np.all(sobol_points <= b))
