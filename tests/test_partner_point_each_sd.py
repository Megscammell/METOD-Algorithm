import numpy as np
import hypothesis
from hypothesis import assume, given, settings, strategies as st

import metod_testing as mtv3

def test_1():
    """Test that for loop takes correct point from all_iterations_of_sd array and stores correctly into 
    all_iterations_of_sd_test array
    """
    iterations = 10
    d = 5
    all_iterations_of_sd = np.random.uniform(0, 1, (iterations + 1, d))
    all_iterations_of_sd_test = np.zeros((iterations + 1, d))
    for its in range(iterations + 1):
        point = all_iterations_of_sd[its, :].reshape((d,))
        all_iterations_of_sd_test[its, :] = point.reshape(1, d)
    
    assert(np.all(all_iterations_of_sd_test == all_iterations_of_sd))

@settings(max_examples=50, deadline=None)
@given(st.integers(2, 100), st.integers(1,30))
def test_2(d, iterations):
    """Ensure size of iterations_of_sd is the same as partner_points_sd
    """
    beta = 0.005 
    g = mtv3.quad_function
    iterations_of_sd = np.random.uniform(0, 1, (iterations + 1, d))
    partner_points_sd = mtv3.partner_point_each_sd(iterations_of_sd, d, beta, iterations, g, func_args)
    assert(partner_points_sd.shape[0] == all_iterations_of_sd.shape[0])
    assert(partner_points_sd.shape[1] == all_iterations_of_sd.shape[1])