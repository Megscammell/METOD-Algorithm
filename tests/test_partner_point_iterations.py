import numpy as np

def test_1():
    """Test that for loop takes correct point from all_iterations_of_sd array and stores correctly into all_iterations_of_sd_test array
    """
    iterations = 10
    d = 5
    all_iterations_of_sd = np.random.uniform(0, 1, (iterations + 1, d))
    all_iterations_of_sd_test = np.zeros((iterations + 1, d))
    for its in range(iterations + 1):
        point = all_iterations_of_sd[its, :].reshape((d,))
        all_iterations_of_sd_test[its, :] = point.reshape(1, d)
    
    assert(np.all(all_iterations_of_sd_test == all_iterations_of_sd))
