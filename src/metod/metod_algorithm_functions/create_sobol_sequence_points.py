import SALib
from SALib.sample import sobol_sequence
import numpy as np


def create_sobol_sequence_points(bound_min, bound_max, d, num_points):
    """
    Create Sobol sequence points (see [1]).

    Parameters
    ----------
    bound_min : integer
                Smallest bound
    bound_max : integer 
                Largest bound.
    d : integer
        Size of dimension.
    num_points : integer
                 Number of random points generated.

    Returns
    -------
    sobol_points : 2-D array
                   Array with shape (num_points * 5, d), computed using [1].Each row has been randomly shuffled.

    References
    ----------
    1) Herman et al, (2017), SALib: An open-source Python library for 
       Sensitivity Analysis, Journal of Open Source Software, 2(9), 97, doi:10.
       21105/joss.00097
    """

    diff = bound_max - bound_min
    temp_sobol_points = sobol_sequence.sample(num_points * 5, d)
    sobol_points = temp_sobol_points * (-diff) + bound_max
    np.random.shuffle(sobol_points)
    return sobol_points
